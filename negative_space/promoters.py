"""Behavioural promoters defined by constructing their negative space.

General mechanism
-----------------
A **promoter** names a desired behaviour. Its *positive samples* are model states
(activations) where the behaviour is *not* needed or already holds; for the
knowledge promoter these are items the model demonstrably knows. From those
positives we construct the promoter's **negative space**, meaning the boundary,
the holes and the outside of the positive region, and attach behaviours to those
regions. When a new state falls in a region, the attached behaviour is promoted:
a keko routing decision, an activation-steering intervention, or a training signal.

This module keeps the API promoter-agnostic (``Promoter``, ``Region``, ``Behaviour``,
``NegativePromoterEvent``). One promoter is implemented: ``KNOWLEDGE_PROMOTER``
(uncertainty / knowledge -> abstain, clarify, retrieve, grow). The events it emits
are **negative promoters**, the mirror image of the *grounding promoters* of
KEKO v4.2 (Executive Summary, §1): grounding promoters are low-uncertainty
sensorial evidence that *reduce* uncertainty and build beliefs; negative promoters
are signals derived from the model's negative space that *promote developmental
behaviour* in the inferential stream.

Keko compatibility
------------------
* ``StreamType`` mirrors KEKO v4.2 §1 (values identical).
* Stream invariants (v4.2 §10): negative promoters act only on inferential /
  epistemic streams; a sensorial observation or telemetry event can never produce one
  (``StreamRoutingError``), and only training-eligible streams may be written to the
  uncertainty buffer (UB write guard).
* ``NegativePromoterEvent.to_ub_row()`` returns exactly the columns of the v4.2 §15
  ``uncertainty_buffer`` table.
* ``to_clarification_record()`` builds ``uncertain.ClarificationRecord``.
* ``suggested_k`` maps into v4.2 §11 ``LANGUAGE_K_MIN..LANGUAGE_K_MAX`` (1..5).
* The scalar baseline is the keko window ``0.3 < p < 0.7`` of ``uncertain.py``
  (``UncertaintyPNN.uncertainty_lower/upper``; H7).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class StreamType(Enum):
    """Mirror of KEKO v4.2 §1 ``StreamType`` (the spec is markdown; values match)."""

    SENSORIAL_OBSERVATION = "sensorial_observation"
    GROUNDING_EVIDENCE = "grounding_evidence"
    WORLD_STATE_BELIEF = "world_state_belief"
    RAW_PERCEPTION = "raw_perception"
    PERCEPTUAL_INFERENCE = "perceptual_inference"
    LANGUAGE_INFERENCE = "language_inference"
    RESOURCE_TELEMETRY = "resource_telemetry"
    DREAM_REPLAY = "dream_replay"


# v4.2 §10 UB write guard.
TRAINING_ELIGIBLE_STREAMS = {StreamType.LANGUAGE_INFERENCE, StreamType.PERCEPTUAL_INFERENCE,
                             StreamType.DREAM_REPLAY}
# Negative promoters are computed from model activations on inferential inputs only.
PROMOTER_ELIGIBLE_STREAMS = {StreamType.LANGUAGE_INFERENCE, StreamType.PERCEPTUAL_INFERENCE}

LANGUAGE_K_MIN, LANGUAGE_K_MAX = 1, 5          # v4.2 §11
KEKO_WINDOW = (0.3, 0.7)                        # uncertain.py UncertaintyPNN


class StreamRoutingError(RuntimeError):
    """Raised when a stream reaches a path its invariants forbid (v4.2 §10)."""


class Region(str, Enum):
    INTERIOR = "interior"    # known territory
    BOUNDARY = "boundary"    # fragile: flips at low noise / inconsistent / probe undecided
    HOLE = "hole"            # unknown pocket surrounded by known territory, model confident
    OUTSIDE = "outside"      # far from the known region


class Behaviour(str, Enum):
    NONE = "frozen_core_answers"
    CLARIFY_RETRIEVE = "clarify_and_adaptive_k_retrieval"
    GROW_REPLAY = "targeted_column_growth_and_ub_replay"
    ABSTAIN_GROUND = "abstain_or_request_grounding"


@dataclass
class Promoter:
    """Spec for one behavioural promoter.

    name            identifier
    positive_source description of where positive samples come from
    label_fn        record -> True if the record is a positive sample
    behaviours      region -> behaviour promoted when a state enters that region
    uncertainty_type value written to the UB ``uncertainty_type`` column
    """

    name: str
    positive_source: str
    label_fn: Callable[[dict], bool]
    behaviours: dict[Region, Behaviour]
    uncertainty_type: str
    description: str = ""

    def positives(self, records: list[dict]) -> list[bool]:
        return [bool(self.label_fn(r)) for r in records]


KNOWLEDGE_PROMOTER = Promoter(
    name="knowledge",
    positive_source="answerable items labelled 'known' (greedy correct and all k samples correct)",
    label_fn=lambda r: r.get("label") == "known",
    behaviours={
        Region.INTERIOR: Behaviour.NONE,
        Region.BOUNDARY: Behaviour.CLARIFY_RETRIEVE,
        Region.HOLE: Behaviour.GROW_REPLAY,
        Region.OUTSIDE: Behaviour.ABSTAIN_GROUND,
    },
    uncertainty_type="epistemic_negative_space",
    description="First instance: the negative space of what the model knows.",
)

# Planned (N6, not implemented): a clarification promoter whose positives are prompts
# with a single clear interpretation; its negative space is where interpretations diverge.


@dataclass
class RegionThresholds:
    tau_outside: float          # distance above which a state is outside the known region
    tau_flip: float             # predicted flip level at or below which a state is on the boundary
    p_boundary: float = 0.5     # probe P(not known) at or above which a state is on the boundary
    confidence_high: float = 0.7


def assign_region(distance: float, pred_flip: float, p_not_known: float, in_hole: bool,
                  confidence: float, th: RegionThresholds) -> Region:
    """Precedence: HOLE > OUTSIDE > BOUNDARY > INTERIOR.

    HOLE requires the model to be *confident* (first-token max prob >= confidence_high):
    a confident state inside an unknown pocket is the "confident but wrong" case.
    """
    if in_hole and confidence >= th.confidence_high:
        return Region.HOLE
    if distance > th.tau_outside:
        return Region.OUTSIDE
    if p_not_known >= th.p_boundary or pred_flip <= th.tau_flip:
        return Region.BOUNDARY
    return Region.INTERIOR


def keko_window_behaviour(max_prob: float, window: tuple[float, float] = KEKO_WINDOW) -> Behaviour:
    """What keko's scalar window would do: route 0.3 < p < 0.7 to columns/clarification."""
    lo, hi = window
    return Behaviour.CLARIFY_RETRIEVE if lo < max_prob < hi else Behaviour.NONE


def severity(score: float) -> str:
    return "high" if score >= 0.66 else "medium" if score >= 0.33 else "low"


def suggested_k(score: float) -> int:
    s = min(max(score, 0.0), 1.0)
    return int(round(LANGUAGE_K_MIN + s * (LANGUAGE_K_MAX - LANGUAGE_K_MIN)))


@dataclass
class NegativePromoterEvent:
    event_id: str
    timestamp: float
    stream_type: StreamType
    source_id: str
    promoter: str
    region: Region
    behaviour: Behaviour
    severity: str
    uncertainty_score: float        # the negative score in [0, 1]
    uncertainty_type: str
    context_vector_ref: Optional[str] = None
    payload_ref: Optional[str] = None
    column_id: Optional[int] = None
    session_id: Optional[str] = None
    node_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def enqueue_for_training(self) -> bool:
        return self.behaviour in (Behaviour.GROW_REPLAY, Behaviour.CLARIFY_RETRIEVE)

    def replay_weight(self) -> float:
        """UB sampling weight: holes are weighted by the persistence of their hole."""
        if self.region == Region.HOLE:
            return float(self.uncertainty_score * (1.0 + self.metadata.get("hole_persistence", 0.0)))
        return float(self.uncertainty_score)

    def to_ub_row(self) -> dict[str, Any]:
        """Row for the v4.2 §15 ``uncertainty_buffer`` table (UB write guard applied)."""
        if self.stream_type not in TRAINING_ELIGIBLE_STREAMS:
            raise StreamRoutingError(f"{self.stream_type} reached UB write path")
        meta = dict(self.metadata, promoter=self.promoter, region=self.region.value,
                    behaviour=self.behaviour.value, severity=self.severity,
                    replay_weight=self.replay_weight())
        return dict(event_id=self.event_id, timestamp=self.timestamp,
                    stream_type=self.stream_type.value, source_id=self.source_id,
                    uncertainty_score=float(self.uncertainty_score),
                    uncertainty_type=self.uncertainty_type,
                    context_vector_ref=self.context_vector_ref, payload_ref=self.payload_ref,
                    column_id=self.column_id, session_id=self.session_id, node_id=self.node_id,
                    metadata_json=json.dumps(meta, default=str))

    def to_clarification_record(self, original_input: str, clarifying_question: str,
                                mode: str = "autonomous"):
        """Build keko's ``uncertain.ClarificationRecord`` for boundary events."""
        from uncertain import ClarificationRecord  # keko root module (imports torch/transformers)

        return ClarificationRecord(timestamp=self.timestamp, original_input=original_input,
                                   clarifying_question=clarifying_question, clarified_response=None,
                                   uncertainty_before=float(self.uncertainty_score),
                                   uncertainty_after=None, clarification_gain=None, mode=mode)


def make_event(promoter: Promoter, region: Region, score: float, *,
               stream_type: StreamType = StreamType.LANGUAGE_INFERENCE,
               source_id: str = "user_prompt", hole_persistence: float = 0.0,
               context_vector_ref: str | None = None, payload_ref: str | None = None,
               session_id: str | None = None, extra: dict | None = None) -> NegativePromoterEvent | None:
    """Create the negative-promoter event for a state, or None for the interior."""
    if stream_type not in PROMOTER_ELIGIBLE_STREAMS:
        raise StreamRoutingError(
            f"negative promoters act on inferential streams only; got {stream_type.value}")
    behaviour = promoter.behaviours[region]
    if behaviour == Behaviour.NONE:
        return None
    meta = dict(extra or {})
    if region == Region.HOLE:
        meta["hole_persistence"] = float(hole_persistence)
    if behaviour == Behaviour.CLARIFY_RETRIEVE:
        meta["suggested_k"] = suggested_k(score)
    return NegativePromoterEvent(
        event_id=str(uuid.uuid4()), timestamp=time.time(), stream_type=stream_type,
        source_id=source_id, promoter=promoter.name, region=region, behaviour=behaviour,
        severity=severity(score), uncertainty_score=float(score),
        uncertainty_type=promoter.uncertainty_type, context_vector_ref=context_vector_ref,
        payload_ref=payload_ref, session_id=session_id, metadata=meta)
