# KEKO Uncertainty Mechanism Specification v4.2
### *Grounded Telemetry, Resource Governance, and Wake/Sleep Persistence Edition*

**Author:** Oscar Liguori ([@Divagnz](https://github.com/Divagnz))  
**Repository:** https://github.com/champi-ai/keko  
**Supersedes:** v4.1 — Dual-Stream Architecture Edition  
**Status:** Active Research Specification  
**Version Goal:** Convert v4.1 from a dual-stream uncertainty router into a grounded, resource-aware, persistent, wake/sleep-capable cognitive system.

---

## Changelog from v4.1

| Section | Change | Reason |
|---------|--------|--------|
| Global terminology | Replaces ambiguous “entropy window” wording with explicit `H`, `C`, `M`, and `U` definitions | v4.1 used entropy in prose but max-probability confidence in code |
| §1 | Replaces strict dual-stream with a layered stream taxonomy | Sensor data can be low-uncertainty grounding evidence, not merely bypass data |
| §2 | Adds Grounding Evidence Layer | Sensor diversity, correlation, and causal confirmation reduce belief uncertainty |
| §3 | Adds World-State Belief Graph | The system now maintains grounded beliefs such as `garage_light_is_on` |
| §4 | Adds Telemetry Resource Governor | Internal telemetry controls compute metabolism: training, columns, retrieval, precision, sleep |
| §5 | Revises `DualStreamUncertaintyMonitor` into `LayeredUncertaintyMonitor` | Different uncertainty types require different handling |
| §6 | Revises MemoryBank retrieval | Retrieval now depends on stream class and belief state, not only sensorial/inferential |
| §7 | Corrects loss function semantics | KL-to-base regularizes drift; it does not encourage divergence when minimized |
| §8 | Adds Wake/Sleep Persistence Architecture | Supports awake realtime ingestion and sleep/dream consolidation |
| §9 | Adds Rule 30 memory transformation | Rule 30 is used for sleep-phase memory perturbation/replay, not as raw truth generation |
| §10 | Adds resource-controlled scheduling policies | Telemetry determines active resource mode: normal, conserve, degraded, sleep-training, emergency |
| §11 | Updates stream invariants | Raw sensors do not directly train epistemic columns, but they do update grounding and resource state |
| §12 | Extends metrics | Adds grounding, causal, resource, sleep, and contamination metrics |
| §13 | Updates hypotheses | Adds H16–H22 for grounding, telemetry governance, wake/sleep, and Rule 30 replay |
| §14 | Updates roadmap | Prioritizes persistence, stream tagging, resource governor, and replayable validation before PNN training |
| Appendix | Adds implementation contracts and DB schemas | Makes v4.2 directly implementable |

All valid concepts from v4.1 are preserved: uncertainty as a developmental signal, frozen base model, progressive columns, uncertainty buffers, clarification buffers, adaptive-k memory retrieval, stream isolation guards, background training, meta-supervision, and hypothesis-driven validation.

---

## Executive Summary

KEKO is an uncertainty-aware adaptive architecture. Its core thesis remains unchanged:

> **Uncertainty is not merely an error signal. It is a developmental signal.**

v4.1 correctly identified that not all inputs have the same epistemic status. A CPU temperature reading, a relay state, a camera frame, and a philosophical user question should not enter the same uncertainty pipeline.

v4.2 refines that insight.

The main correction is that sensor inputs are not just “bypass data.” They are **grounding promoters**. A relay state, a camera observation, a microphone impulse, a power draw reading, and a door sensor can jointly reduce uncertainty about the world. The system should not train directly on raw sensor events as if they were language uncertainty, but it should use them to build grounded beliefs.

The second correction is that internal telemetry is not only another input stream. It is the system’s **body-state**. GPU temperature, VRAM usage, latency, queue depth, CPU pressure, and battery/power state must govern what the system is allowed to do. Telemetry controls compute metabolism.

v4.2 therefore introduces three architectural pillars missing or underdeveloped in v4.1:

1. **Grounding Evidence Layer** — uses diverse sensors to substantiate world-state beliefs through correlation and cause/effect confirmation.
2. **Telemetry Resource Governor** — uses internal telemetry to control training, inference, column activation, retrieval depth, and sleep/dream replay.
3. **Wake/Sleep Persistence Layer** — awake mode receives realtime feed; sleep mode consolidates memory, applies Rule 30 transformations, and trains from replay.

The result is no longer merely a dual-stream uncertainty pipeline. It is a grounded, resource-aware, persistent adaptive system.

---

## Core Philosophy

> *"Knowing what you don't know is worth more than all the internet's data."*

The architecture treats uncertainty as a first-class computational object. However, v4.2 distinguishes multiple kinds of uncertainty:

| Type | Example | Meaning | Action |
|------|---------|---------|--------|
| Operational uncertainty | GPU temp sensor missing | System/body-state ambiguity | Resource governor alert |
| Measurement uncertainty | Camera brightness noisy | Sensor reliability issue | Update reliability score |
| Grounding uncertainty | Relay ON but camera dark | World-state contradiction | Diagnostic inference |
| Perceptual uncertainty | Is that a person or shadow? | Derived perception ambiguity | Perceptual uncertainty event |
| Epistemic uncertainty | User asks ambiguous question | Meaning/reasoning ambiguity | Clarification/columns/tools |
| Meta-uncertainty | System unsure why it is unsure | Self-monitoring ambiguity | MetaSupervisor analysis |

The important distinction:

> Raw observations are not automatically training samples.  
> Derived contradictions, unresolved beliefs, and failed predictions can become uncertainty events.

---

## Relationship to Consciousness / SEPAS

v4.2 keeps the SEPAS relationship:

1. **Presence** — the system exists in a state.
2. **Acknowledgement** — it detects gaps, conflicts, and uncertainty.
3. **Self-Emergent** — specialization emerges from resolving uncertainty.
4. **System** — coherence is maintained through persistent memory, frozen core, adaptive periphery, and resource governance.

v4.2 adds a more embodied interpretation:

- External sensors ground the system in the world.
- Internal telemetry grounds the system in its own computational body.
- Wake mode receives experience.
- Sleep mode consolidates experience.
- Dreams are structured perturbations of memory traces used for training and integration.

---

# §1 — Layered Input Taxonomy

v4.1 used two streams:

```text
SENSORIAL
INFERENTIAL
```

v4.2 replaces this with a layered taxonomy.

```python
from enum import Enum

class StreamType(Enum):
    # External world observations
    SENSORIAL_OBSERVATION = "sensorial_observation"

    # Evidence derived from raw observations
    GROUNDING_EVIDENCE = "grounding_evidence"

    # Current grounded beliefs about world/system state
    WORLD_STATE_BELIEF = "world_state_belief"

    # Raw perceptual data that should be persisted but not directly reasoned over
    RAW_PERCEPTION = "raw_perception"

    # Derived perception outputs: objects, faces, speech, motion, brightness, etc.
    PERCEPTUAL_INFERENCE = "perceptual_inference"

    # User prompts, dialogue, planning, ambiguity, synthesis
    LANGUAGE_INFERENCE = "language_inference"

    # Internal telemetry: GPU, CPU, RAM, queue depth, latency, thermal state
    RESOURCE_TELEMETRY = "resource_telemetry"

    # System-generated replay/dream samples during sleep
    DREAM_REPLAY = "dream_replay"
```

## Stream Roles

| Stream | Example | Primary Role | Direct Epistemic Training? |
|--------|---------|--------------|-----------------------------|
| `SENSORIAL_OBSERVATION` | relay ON, door open | Observed external event | No |
| `GROUNDING_EVIDENCE` | relay ON supports light ON | Belief support | No, but updates belief graph |
| `WORLD_STATE_BELIEF` | garage light probably ON | Current grounded belief | Only if belief conflict occurs |
| `RAW_PERCEPTION` | video frame, audio chunk | Persistent raw feed | No |
| `PERCEPTUAL_INFERENCE` | person detected, light increased | Derived model-dependent claim | Yes, separate perceptual path |
| `LANGUAGE_INFERENCE` | user prompt or reasoning task | Epistemic reasoning | Yes |
| `RESOURCE_TELEMETRY` | GPU 82°C, p95 latency 3.1s | Compute control | No, controls governor |
| `DREAM_REPLAY` | transformed memory trace | Sleep training sample | Yes, gated |

## Key v4.2 Principle

```text
Sensorial observations are low-uncertainty promoters.
They do not directly train epistemic columns.
They update grounding, reliability, causal links, and belief confidence.
```

---

# §2 — Input Classifier and Event Envelope

Every input is wrapped in a durable event envelope before routing.

```python
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time
import uuid

@dataclass
class EventEnvelope:
    event_id: str
    timestamp: float
    source_id: str
    stream_type: StreamType
    content: Any
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_event_ids: list[str] = field(default_factory=list)
    session_id: Optional[str] = None
    node_id: Optional[str] = None

    @staticmethod
    def create(
        source_id: str,
        stream_type: StreamType,
        content: Any,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        parent_event_ids: Optional[list[str]] = None,
        session_id: Optional[str] = None,
        node_id: Optional[str] = None,
    ):
        return EventEnvelope(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            source_id=source_id,
            stream_type=stream_type,
            content=content,
            confidence=confidence,
            metadata=metadata or {},
            parent_event_ids=parent_event_ids or [],
            session_id=session_id,
            node_id=node_id,
        )
```

## Source Registry

The classifier should not infer source type from text. Source identity is declared by ingestion.

```python
class SourceRegistry:
    SOURCE_STREAMS = {
        # External sensors
        "garage_relay_light": StreamType.SENSORIAL_OBSERVATION,
        "garage_front_camera": StreamType.RAW_PERCEPTION,
        "garage_door_sensor": StreamType.SENSORIAL_OBSERVATION,
        "power_meter_garage": StreamType.SENSORIAL_OBSERVATION,
        "microphone_garage": StreamType.RAW_PERCEPTION,

        # Derived perception
        "camera_brightness_detector": StreamType.PERCEPTUAL_INFERENCE,
        "object_detector": StreamType.PERCEPTUAL_INFERENCE,
        "speech_to_text": StreamType.PERCEPTUAL_INFERENCE,

        # Internal telemetry
        "gpu_metrics": StreamType.RESOURCE_TELEMETRY,
        "cpu_metrics": StreamType.RESOURCE_TELEMETRY,
        "ram_metrics": StreamType.RESOURCE_TELEMETRY,
        "queue_metrics": StreamType.RESOURCE_TELEMETRY,
        "latency_metrics": StreamType.RESOURCE_TELEMETRY,

        # User/system reasoning
        "user_prompt": StreamType.LANGUAGE_INFERENCE,
        "tool_result": StreamType.LANGUAGE_INFERENCE,
        "dream_replay": StreamType.DREAM_REPLAY,
    }

    def classify(self, raw_input: Any, source_id: str, **kwargs) -> EventEnvelope:
        stream_type = self.SOURCE_STREAMS.get(source_id, StreamType.LANGUAGE_INFERENCE)
        return EventEnvelope.create(
            source_id=source_id,
            stream_type=stream_type,
            content=raw_input,
            confidence=kwargs.get("confidence", 1.0),
            metadata=kwargs.get("metadata", {}),
            session_id=kwargs.get("session_id"),
            node_id=kwargs.get("node_id"),
        )
```

---

# §3 — Grounding Evidence Layer

## Purpose

The Grounding Evidence Layer converts low-uncertainty observations into belief support.

It answers questions like:

```text
Did the garage light actually turn on?
Is the relay working?
Is the camera mapped to the correct physical location?
Is the system seeing the expected effect after an action?
```

## Example: Relay + Camera Confirmation

```text
Event A:
garage_relay_light = ON
confidence = 0.98

Event B:
garage_front_camera brightness increased
confidence = 0.82

Temporal relation:
B occurred 250 ms after A

Expected causal window:
0–1000 ms

Derived belief:
garage_light_is_on = true
belief confidence increases
```

## Evidence Atom

```python
@dataclass
class EvidenceAtom:
    evidence_id: str
    belief_key: str
    source_event_id: str
    supports: bool
    strength: float
    source_reliability: float
    independence_weight: float
    timestamp: float
    causal_role: str | None = None  # cause, effect, context, contradiction
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## Belief Confidence Update

A simple first version:

```python
def combine_evidence(evidence_atoms: list[EvidenceAtom]) -> float:
    """
    Combines independent support using noisy-OR style aggregation.
    This is intentionally simple for v4.2.
    """
    remaining_uncertainty = 1.0

    for e in evidence_atoms:
        if not e.supports:
            continue

        contribution = e.strength * e.source_reliability * e.independence_weight
        contribution = max(0.0, min(1.0, contribution))
        remaining_uncertainty *= (1.0 - contribution)

    return 1.0 - remaining_uncertainty
```

## Independence Matters

Two cameras are useful. But a camera plus relay plus power draw plus microphone click is stronger.

| Evidence | Modality | Shared Failure Risk |
|----------|----------|---------------------|
| Camera A bright | Vision | Lighting/camera noise |
| Camera B bright | Vision | Similar visual noise |
| Relay ON | Electrical/control | Different failure path |
| Power draw increased | Electrical/measurement | Different failure path |
| Microphone click | Audio | Different failure path |

Diverse sensors reduce uncertainty because they reduce shared failure modes.

---

# §4 — Causal Grounding Layer

Correlation is not enough. KEKO should distinguish correlation from causal confirmation.

## Correlation

```text
relay ON and camera bright happen together
```

## Causal Confirmation

```text
relay changed first
camera brightness changed after
delay was inside expected physical window
relation repeated across events
camera brightness usually did not change without relay event
relay ON without brightness created anomaly
```

## Causal Link Model

```python
@dataclass
class CausalLink:
    cause_key: str
    effect_key: str
    expected_min_delay_ms: int
    expected_max_delay_ms: int
    confidence: float
    observations: int = 0
    confirmations: int = 0
    contradictions: int = 0

    def update(self, cause_time: float, effect_time: float, effect_seen: bool):
        self.observations += 1

        delay_ms = int((effect_time - cause_time) * 1000)

        if effect_seen and self.expected_min_delay_ms <= delay_ms <= self.expected_max_delay_ms:
            self.confirmations += 1
        else:
            self.contradictions += 1

        total = max(1, self.confirmations + self.contradictions)
        self.confidence = self.confirmations / total
```

## Causal Contradiction

Contradictions become diagnostic uncertainty events.

Example:

```text
relay says ON
camera remains dark
power draw is zero
```

Derived inferential event:

```text
Possible causes:
- bulb failure
- relay fault
- wrong camera mapping
- camera obstruction
- power meter failure
- latency/window mismatch
```

This event is eligible for inferential processing. The raw sensor values are not.

---

# §5 — World-State Belief Graph

The belief graph is the persistent state of grounded reality.

```python
@dataclass
class WorldStateBelief:
    belief_key: str
    value: Any
    confidence: float
    uncertainty: float
    last_updated: float
    supporting_evidence_ids: list[str]
    contradicting_evidence_ids: list[str]
    causal_links: list[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## Example Belief

```json
{
  "belief_key": "garage.light.on",
  "value": true,
  "confidence": 0.94,
  "uncertainty": 0.06,
  "supporting_evidence": [
    "relay_on_event",
    "camera_brightness_event",
    "power_draw_event"
  ],
  "contradicting_evidence": [],
  "causal_links": [
    "relay_on_causes_brightness_increase"
  ]
}
```

## Belief Graph Update Path

```text
Sensorial Observation
    ↓
Evidence Atom
    ↓
Causal Link Update
    ↓
World-State Belief Update
    ↓
Contradiction Detection
    ↓
Inferential Event if unresolved
```

---

# §6 — Resource Telemetry as Compute Metabolism

v4.1 mentioned GPU temperature, queue load, latency, and scheduler behavior, but telemetry was not modeled as a first-class resource-control subsystem.

v4.2 adds the **Telemetry Resource Governor**.

## Principle

```text
External sensors ground the world.
Internal telemetry grounds the system's own body-state.
```

Resource telemetry does not enter the uncertainty buffer. It controls what the system is allowed to do.

## Resource Signals

| Signal | Source | Meaning |
|--------|--------|---------|
| GPU temperature | NVML/ROCm/driver | Thermal headroom |
| GPU utilization | driver metrics | Compute saturation |
| VRAM used | driver metrics | Memory pressure |
| CPU load | OS metrics | Host pressure |
| RAM pressure | OS metrics | Memory risk |
| Disk I/O | OS metrics | Persistence pressure |
| Queue depth | orchestration | Pending work |
| p95 user latency | serving layer | User-facing degradation |
| timeout rate | serving layer | Reliability degradation |
| battery / power mode | OS/hardware | Energy constraint |
| active input load | ingestion | Awake pressure |
| sleep eligibility | scheduler | Consolidation opportunity |

---

# §7 — Telemetry Resource Governor

## Resource Modes

```python
from enum import Enum

class ResourceMode(Enum):
    NORMAL = "normal"
    CONSERVE = "conserve"
    DEGRADED = "degraded"
    SLEEP_TRAINING = "sleep_training"
    EMERGENCY = "emergency"
```

## Telemetry Snapshot

```python
@dataclass
class TelemetrySnapshot:
    timestamp: float
    gpu_temp_c: float | None = None
    gpu_util: float | None = None
    vram_used_ratio: float | None = None
    cpu_load: float | None = None
    ram_pressure: float | None = None
    disk_io_pressure: float | None = None
    user_queue_depth: int = 0
    background_queue_depth: int = 0
    user_latency_p95: float = 0.0
    timeout_rate: float = 0.0
    battery_saver: bool = False
    active_realtime_feed: bool = False
```

## Governor Evaluation

```python
class TelemetryResourceGovernor:
    def evaluate(self, telemetry: TelemetrySnapshot) -> ResourceMode:
        if (
            (telemetry.gpu_temp_c is not None and telemetry.gpu_temp_c > 85)
            or (telemetry.vram_used_ratio is not None and telemetry.vram_used_ratio > 0.95)
            or telemetry.timeout_rate > 0.20
        ):
            return ResourceMode.EMERGENCY

        if telemetry.user_latency_p95 > 2.0 or telemetry.user_queue_depth > 5:
            return ResourceMode.DEGRADED

        if (
            (telemetry.gpu_util is not None and telemetry.gpu_util > 0.80)
            or (telemetry.ram_pressure is not None and telemetry.ram_pressure > 0.85)
            or telemetry.battery_saver
        ):
            return ResourceMode.CONSERVE

        if (
            telemetry.user_queue_depth == 0
            and not telemetry.active_realtime_feed
            and telemetry.background_queue_depth > 0
            and (telemetry.gpu_temp_c is None or telemetry.gpu_temp_c < 65)
        ):
            return ResourceMode.SLEEP_TRAINING

        return ResourceMode.NORMAL
```

## Policy Map

```python
RESOURCE_POLICIES = {
    ResourceMode.NORMAL: {
        "max_columns": 4,
        "memory_k_max": 5,
        "background_training": True,
        "verification_async": True,
        "tool_fanout": True,
        "dream_replay": False,
        "precision": "bf16",
    },

    ResourceMode.CONSERVE: {
        "max_columns": 2,
        "memory_k_max": 3,
        "background_training": "throttled",
        "verification_async": True,
        "tool_fanout": "limited",
        "dream_replay": False,
        "precision": "bf16",
    },

    ResourceMode.DEGRADED: {
        "max_columns": 1,
        "memory_k_max": 1,
        "background_training": False,
        "verification_async": "minimal",
        "tool_fanout": False,
        "dream_replay": False,
        "precision": "int8_or_cached",
    },

    ResourceMode.SLEEP_TRAINING: {
        "max_columns": 0,
        "memory_k_max": 5,
        "background_training": True,
        "verification_async": False,
        "tool_fanout": False,
        "dream_replay": True,
        "rule30_memory_transform": True,
        "precision": "bf16",
    },

    ResourceMode.EMERGENCY: {
        "max_columns": 0,
        "memory_k_max": 1,
        "background_training": False,
        "verification_async": False,
        "tool_fanout": False,
        "dream_replay": False,
        "only_cached_or_base_response": True,
        "precision": "lowest_safe",
    },
}
```

## Governor Effects

| Resource Mode | Behavior |
|---------------|----------|
| `NORMAL` | Full inferential pipeline available |
| `CONSERVE` | Fewer columns, smaller retrieval, throttled training |
| `DEGRADED` | User latency protected; background tasks paused |
| `SLEEP_TRAINING` | No user pressure; replay/consolidation allowed |
| `EMERGENCY` | Stop training, stop replay, use cached/base responses |

---

# §8 — Corrected Uncertainty Definitions

v4.1 used entropy language but implemented a max-probability mask. v4.2 separates the concepts.

## Token-Level Measures

```python
import torch
import torch.nn.functional as F

def normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    max_entropy = torch.log(torch.tensor(probs.shape[-1], device=probs.device))
    return entropy / max_entropy

def max_confidence(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return probs.max(dim=-1).values

def top2_margin(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    top2 = probs.topk(k=2, dim=-1).values
    return top2[..., 0] - top2[..., 1]

def uncertainty_score(
    logits: torch.Tensor,
    w_entropy: float = 0.50,
    w_confidence: float = 0.35,
    w_margin: float = 0.15,
) -> torch.Tensor:
    H = normalized_entropy(logits)
    C = max_confidence(logits)
    M = top2_margin(logits)

    # High entropy, low confidence, and low margin all increase uncertainty.
    return (
        w_entropy * H
        + w_confidence * (1.0 - C)
        + w_margin * (1.0 - M)
    )
```

## Definitions

| Symbol | Meaning | Direction |
|--------|---------|-----------|
| `H` | Normalized entropy | Higher = more distributed probability |
| `C` | Max probability confidence | Higher = more confident top token |
| `M` | Top-2 margin | Higher = clearer winner |
| `U` | Combined uncertainty score | Higher = more uncertain |

Thresholds should be defined against `U`, not vaguely against entropy.

---

# §9 — Layered Uncertainty Monitor

```python
class LayeredUncertaintyMonitor:
    """
    Handles uncertainty according to stream type.

    RESOURCE_TELEMETRY:
      - routed to Resource Governor
      - no UB write
      - no column activation

    SENSORIAL_OBSERVATION:
      - persisted as observation
      - converted to evidence atoms
      - no epistemic UB write

    RAW_PERCEPTION:
      - persisted
      - sent to perception models
      - no direct epistemic UB write

    PERCEPTUAL_INFERENCE:
      - may enter perceptual uncertainty buffer
      - may update belief graph

    WORLD_STATE_BELIEF:
      - normally updates state
      - contradictions can generate inferential events

    LANGUAGE_INFERENCE:
      - full uncertainty pipeline

    DREAM_REPLAY:
      - sleep-phase training candidate
      - gated by resource mode and validation
    """

    LANGUAGE_WINDOW = (0.35, 0.75)
    PERCEPTUAL_WINDOW = (0.30, 0.80)
    BELIEF_CONFLICT_THRESHOLD = 0.35

    def route(self, event: EventEnvelope, logits=None):
        if event.stream_type == StreamType.RESOURCE_TELEMETRY:
            return "resource_governor"

        if event.stream_type == StreamType.SENSORIAL_OBSERVATION:
            return "grounding_evidence"

        if event.stream_type == StreamType.RAW_PERCEPTION:
            return "persist_and_perception_extract"

        if event.stream_type == StreamType.PERCEPTUAL_INFERENCE:
            return "perceptual_uncertainty_or_belief_update"

        if event.stream_type == StreamType.WORLD_STATE_BELIEF:
            return "belief_graph_update"

        if event.stream_type == StreamType.DREAM_REPLAY:
            return "sleep_training_candidate"

        if event.stream_type == StreamType.LANGUAGE_INFERENCE:
            if logits is None:
                return "language_inference_no_logits"

            U = uncertainty_score(logits)
            low, high = self.LANGUAGE_WINDOW
            mask = (U > low) & (U < high)
            return {
                "route": "language_uncertainty_pipeline",
                "uncertainty": U,
                "mask": mask,
            }

        return "unknown"
```

---

# §10 — Revised Stream Invariants

## Hard Invariants

| Invariant | Description |
|----------|-------------|
| Resource telemetry never enters UB/CB | It controls scheduling, not epistemic learning |
| Sensor observations never directly train language columns | They first become evidence/belief updates |
| Raw perception is persisted but not directly treated as reasoning | Derived perceptual outputs handle uncertainty |
| Contradictions may generate inferential events | Derived conflicts are valid uncertainty signals |
| World-state beliefs are auditable | Each belief must trace to supporting/contradicting evidence |
| Sleep training is resource-gated | Dream replay cannot run during emergency/degraded user pressure |
| Raw sensor data and derived claims must not be conflated | Observed event ≠ interpreted meaning |

## UB Write Guard

```python
TRAINING_ELIGIBLE_STREAMS = {
    StreamType.LANGUAGE_INFERENCE,
    StreamType.PERCEPTUAL_INFERENCE,
    StreamType.DREAM_REPLAY,
}

def write_to_ub(sample):
    if sample.stream_type not in TRAINING_ELIGIBLE_STREAMS:
        raise StreamRoutingError(
            f"{sample.stream_type} reached UB write path. "
            "Only language, perceptual inference, and gated dream replay are eligible."
        )

    ub_local.append(sample)
```

## Grounding Store Write

```python
GROUNDING_ELIGIBLE_STREAMS = {
    StreamType.SENSORIAL_OBSERVATION,
    StreamType.GROUNDING_EVIDENCE,
    StreamType.WORLD_STATE_BELIEF,
    StreamType.PERCEPTUAL_INFERENCE,
}

def write_to_grounding_store(event_or_evidence):
    grounding_store.append(event_or_evidence)
```

---

# §11 — Memory Bank and Retrieval Strategy

v4.2 keeps adaptive retrieval but makes it stream-aware at a finer level.

```python
class MemoryBank:
    RESOURCE_K = 0
    SENSORIAL_K = 1
    BELIEF_K = 3
    PERCEPTUAL_K_MIN = 1
    PERCEPTUAL_K_MAX = 5
    LANGUAGE_K_MIN = 1
    LANGUAGE_K_MAX = 5
    DREAM_K_MAX = 8

    RELEVANCE_THRESHOLD = 0.6

    def retrieve(self, query_embedding, stream_type: StreamType, resource_policy=None):
        if stream_type == StreamType.RESOURCE_TELEMETRY:
            return []

        if stream_type == StreamType.SENSORIAL_OBSERVATION:
            return self._retrieve(query_embedding, k=self.SENSORIAL_K)

        if stream_type == StreamType.WORLD_STATE_BELIEF:
            return self._retrieve(query_embedding, k=self.BELIEF_K)

        if stream_type == StreamType.PERCEPTUAL_INFERENCE:
            return self._adaptive_retrieve(
                query_embedding,
                k_min=self.PERCEPTUAL_K_MIN,
                k_max=self.PERCEPTUAL_K_MAX,
            )

        if stream_type == StreamType.LANGUAGE_INFERENCE:
            k_max = self.LANGUAGE_K_MAX
            if resource_policy is not None:
                k_max = min(k_max, resource_policy.get("memory_k_max", k_max))

            return self._adaptive_retrieve(
                query_embedding,
                k_min=self.LANGUAGE_K_MIN,
                k_max=k_max,
            )

        if stream_type == StreamType.DREAM_REPLAY:
            return self._retrieve(query_embedding, k=self.DREAM_K_MAX)

        return self._retrieve(query_embedding, k=1)

    def _adaptive_retrieve(self, query_embedding, k_min: int, k_max: int):
        results = self._retrieve(query_embedding, k=k_min)
        if results and results[0].relevance >= self.RELEVANCE_THRESHOLD:
            return results
        return self._retrieve(query_embedding, k=k_max)

    def _retrieve(self, query_embedding, k: int):
        if k <= 0:
            return []
        distances, indices = self.vector_index.search(query_embedding, k)
        return [self.memory_store[i] for i in indices[0] if i >= 0]
```

---

# §12 — Corrected Loss Function

v4.1 said KL divergence encourages column divergence from the base. That is backwards if KL is minimized.

v4.2 defines the training loss as:

```text
L = L_task
  + α U_after
  + β temporal_drift
  + λ KL(p_col || p_base)
  + ρ contamination_penalty
```

Where:

| Term | Meaning |
|------|---------|
| `L_task` | Task/clarification/replay objective |
| `α U_after` | Penalizes unresolved uncertainty after processing |
| `β temporal_drift` | Penalizes unstable uncertainty oscillation |
| `λ KL(p_col || p_base)` | Regularizes column drift against frozen base |
| `ρ contamination_penalty` | Penalizes invalid stream contamination |

The base model is the frozen anchor. Columns specialize, but the KL term prevents destructive divergence.

```python
def compute_column_loss(sample, column_output, base_output):
    task_loss = sample.task_loss(column_output) if sample.has_task_target else 0.0
    uncertainty_after = sample.uncertainty_after(column_output)
    temporal_drift = (sample.U_t - sample.U_t_minus_1) ** 2

    kl_to_base = kl_divergence(
        log_probs(column_output),
        probs(base_output),
    )

    contamination = 1.0 if sample.stream_type not in TRAINING_ELIGIBLE_STREAMS else 0.0

    return (
        task_loss
        + alpha * uncertainty_after
        + beta * temporal_drift
        + lambda_kl * kl_to_base
        + rho * contamination
    )
```

---

# §13 — Awake/Sleep Persistence Architecture

v4.2 adds a full persistence lifecycle.

## Awake Mode

During awake mode, the system receives realtime feed.

```text
Awake Mode:
- ingest raw audio/video/sensor/resource events
- persist all event envelopes
- extract perceptual features
- update grounding evidence
- update world-state belief graph
- answer user requests
- queue uncertainty events
- protect latency through Resource Governor
```

## Sleep Mode

During sleep mode, the system reduces realtime interaction and consolidates memory.

```text
Sleep Mode:
- pause noncritical active inference
- keep critical telemetry/resource monitoring active
- sample memory traces
- apply Rule 30 transformations to compressed memory planes
- generate dream/replay batches
- train columns/adapters on replay
- validate against held-out memory and grounding consistency
- checkpoint successful updates
- reject unstable updates
```

## Cycle Policy

The user-proposed cycle:

```text
awake: 4 hours
sleep: 3 hours
```

A configurable scheduler:

```python
@dataclass
class WakeSleepPolicy:
    awake_hours: float = 4.0
    sleep_hours: float = 3.0
    allow_interrupt_sleep_for_user: bool = True
    allow_interrupt_sleep_for_emergency: bool = True
    min_gpu_cool_temp_for_sleep_training: float = 65.0
    max_user_queue_for_sleep: int = 0
```

## Wake/Sleep State Machine

```python
class CognitiveState(Enum):
    AWAKE = "awake"
    SLEEP_PREP = "sleep_prep"
    SLEEP_REPLAY = "sleep_replay"
    SLEEP_TRAINING = "sleep_training"
    WAKE_REINTEGRATION = "wake_reintegration"
    INTERRUPTED = "interrupted"
```

```text
AWAKE
  ↓ if schedule + resource eligible
SLEEP_PREP
  ↓
SLEEP_REPLAY
  ↓
SLEEP_TRAINING
  ↓
WAKE_REINTEGRATION
  ↓
AWAKE

Any sleep state
  ↓ if user/emergency pressure
INTERRUPTED
  ↓
AWAKE
```

---

# §14 — Rule 30 Memory Transformation

Rule 30 is used as a sleep-phase perturbation mechanism, not as a source of truth.

## Purpose

```text
Rule 30 creates deterministic-chaotic variations of compressed memory traces.
These variations are used to generate dream/replay samples that expose columns to
nearby-but-novel memory states.
```

## Why Rule 30

Rule 30 is simple, deterministic, chaotic, and cheap. It can perturb memory planes without requiring a large generative model.

## Memory Plane Representation

```text
Memory trace → compressed binary/quantized plane → Rule 30 evolution → replay candidate
```

## Rule 30 Step

```python
def rule30_step(bits: list[int]) -> list[int]:
    n = len(bits)
    out = [0] * n

    for i in range(n):
        left = bits[(i - 1) % n]
        center = bits[i]
        right = bits[(i + 1) % n]

        pattern = (left << 2) | (center << 1) | right

        # Rule 30 binary: 00011110
        out[i] = 1 if pattern in (1, 2, 3, 4) else 0

    return out
```

## Replay Generation

```python
def generate_rule30_replay(memory_bits, steps: int = 8):
    states = [memory_bits]
    current = memory_bits

    for _ in range(steps):
        current = rule30_step(current)
        states.append(current)

    return states
```

## Safety Gate

Rule 30 outputs cannot directly update long-term memory.

They must pass:

```text
1. Reconstruction validity check
2. Grounding consistency check
3. Uncertainty reduction check
4. Catastrophic drift check
5. Resource governor permission
```

## Dream Replay Contract

```python
@dataclass
class DreamReplaySample:
    replay_id: str
    source_memory_ids: list[str]
    rule30_steps: int
    transformed_payload: Any
    reconstruction_score: float
    grounding_consistency_score: float
    uncertainty_gain_estimate: float
    eligible_for_training: bool
```

---

# §15 — Persistence Layer

v4.2 requires persistence from the beginning.

## Required Stores

| Store | Purpose |
|-------|---------|
| `event_log` | Append-only raw event envelopes |
| `raw_media_store` | Video/audio/image chunks |
| `perceptual_feature_store` | Derived features from media |
| `grounding_store` | Evidence atoms and causal links |
| `belief_graph_store` | Current and historical world-state beliefs |
| `resource_telemetry_store` | Internal body-state history |
| `UB_local` | Local uncertainty buffer |
| `CB_local` | Local clarification buffer |
| `dream_replay_store` | Sleep replay candidates and outcomes |
| `column_registry` | Column lifecycle, freeze/unfreeze/fuse metadata |
| `metrics_store` | Operational, uncertainty, grounding, and sleep metrics |

## Event Log Schema

```sql
CREATE TABLE event_log (
    event_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    source_id TEXT NOT NULL,
    stream_type TEXT NOT NULL,
    content_ref TEXT,
    content_json TEXT,
    confidence REAL NOT NULL,
    metadata_json TEXT,
    parent_event_ids_json TEXT,
    session_id TEXT,
    node_id TEXT
);
```

## Grounding Evidence Schema

```sql
CREATE TABLE grounding_evidence (
    evidence_id TEXT PRIMARY KEY,
    belief_key TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    supports INTEGER NOT NULL,
    strength REAL NOT NULL,
    source_reliability REAL NOT NULL,
    independence_weight REAL NOT NULL,
    causal_role TEXT,
    timestamp REAL NOT NULL,
    metadata_json TEXT
);
```

## Belief Graph Schema

```sql
CREATE TABLE world_state_beliefs (
    belief_key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    uncertainty REAL NOT NULL,
    last_updated REAL NOT NULL,
    supporting_evidence_ids_json TEXT,
    contradicting_evidence_ids_json TEXT,
    causal_links_json TEXT,
    metadata_json TEXT
);
```

## Resource Telemetry Schema

```sql
CREATE TABLE resource_telemetry (
    id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    gpu_temp_c REAL,
    gpu_util REAL,
    vram_used_ratio REAL,
    cpu_load REAL,
    ram_pressure REAL,
    disk_io_pressure REAL,
    user_queue_depth INTEGER,
    background_queue_depth INTEGER,
    user_latency_p95 REAL,
    timeout_rate REAL,
    resource_mode TEXT NOT NULL
);
```

## Uncertainty Buffer Schema

```sql
CREATE TABLE uncertainty_buffer (
    event_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    stream_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    uncertainty_score REAL NOT NULL,
    uncertainty_type TEXT NOT NULL,
    context_vector_ref TEXT,
    payload_ref TEXT,
    column_id INTEGER,
    session_id TEXT,
    node_id TEXT,
    metadata_json TEXT,

    CHECK (stream_type IN (
        'language_inference',
        'perceptual_inference',
        'dream_replay'
    ))
);
```

---

# §16 — Async Orchestration v4.2

## Task Priorities

```python
class TaskPriority(IntEnum):
    USER_WAITING = 0
    RESOURCE_GOVERNOR = 1
    GROUNDING_UPDATE = 2
    CLARIFICATION = 3
    TOOL_RESEARCH = 4
    COLUMN_QUERY = 5
    VERIFICATION = 6
    PERCEPTION_EXTRACTION = 7
    BACKGROUND_TRAIN = 8
    DREAM_REPLAY = 9
```

Resource governor tasks must be high priority because they affect whether other tasks are allowed.

## Main Ingestion Flow

```python
async def ingest(raw_input, source_id: str, session_id: str | None = None):
    event = source_registry.classify(
        raw_input=raw_input,
        source_id=source_id,
        session_id=session_id,
    )

    await event_log.write(event)

    if event.stream_type == StreamType.RESOURCE_TELEMETRY:
        telemetry = parse_telemetry(event)
        mode = resource_governor.evaluate(telemetry)
        await resource_state.update(mode, telemetry)
        return

    if event.stream_type == StreamType.SENSORIAL_OBSERVATION:
        evidence = grounding_engine.to_evidence(event)
        await grounding_store.write(evidence)
        await belief_graph.update(evidence)
        return

    if event.stream_type == StreamType.RAW_PERCEPTION:
        await raw_media_store.write(event)
        await task_queue.enqueue(
            PerceptionExtractionTask(event),
            priority=TaskPriority.PERCEPTION_EXTRACTION,
        )
        return

    if event.stream_type == StreamType.PERCEPTUAL_INFERENCE:
        await perceptual_pipeline.process(event)
        return

    if event.stream_type == StreamType.LANGUAGE_INFERENCE:
        await inference_pipeline.process(event)
        return

    if event.stream_type == StreamType.DREAM_REPLAY:
        await sleep_pipeline.process(event)
        return
```

---

# §17 — Inference Pipeline v4.2

```python
async def process_language_inference(event: EventEnvelope):
    policy = await resource_state.current_policy()

    if policy.get("only_cached_or_base_response"):
        return await cached_or_base_response(event)

    # Token hunger only applies to language inference.
    if token_hunger.needs_more_context(event.content):
        return clarification_engine.generate_inquiry(event.content)

    # Initial base inference.
    base_response, logits = await base_model.quick_inference(event.content)

    U = uncertainty_score(logits)
    should_route = uncertainty_router.should_route_language(U)

    if not should_route:
        return base_response

    # Respect resource governor.
    max_columns = policy.get("max_columns", 1)
    memory_k_max = policy.get("memory_k_max", 1)

    memory_context = memory_bank.retrieve(
        query_embedding=embed(event.content),
        stream_type=StreamType.LANGUAGE_INFERENCE,
        resource_policy=policy,
    )

    column_results = []
    if max_columns > 0:
        column_results = await column_router.query(
            event=event,
            memory_context=memory_context,
            max_columns=max_columns,
        )

    tool_results = []
    if policy.get("tool_fanout"):
        tool_results = await tool_orchestrator.query_relevant(event.content)

    response = response_synthesizer.synthesize(
        base_response=base_response,
        column_results=column_results,
        tool_results=tool_results,
        uncertainty=U,
    )

    await maybe_queue_training_sample(event, response, U)

    return response
```

---

# §18 — Background Training and Sleep Training

## Background Training

Background training runs only when the resource policy allows it.

```python
async def background_training_step():
    policy = await resource_state.current_policy()

    if not policy.get("background_training"):
        return

    samples = global_UB.sample(
        batch_size=32,
        weight="entropy_delta_time",
        allowed_streams={
            StreamType.LANGUAGE_INFERENCE,
            StreamType.PERCEPTUAL_INFERENCE,
        },
    )

    for sample in samples:
        if sample.stream_type not in TRAINING_ELIGIBLE_STREAMS:
            log_routing_error(sample)
            continue

        column = column_router.assign(sample.context_vector)
        loss = compute_column_loss(sample, column.output, base_model.output)
        loss.backward()
        optimizer.step()
```

## Sleep Training

```python
async def sleep_training_step():
    policy = await resource_state.current_policy()

    if not policy.get("dream_replay"):
        return

    memory_traces = memory_sampler.sample_for_sleep()
    replay_samples = []

    for trace in memory_traces:
        transformed = rule30_transform(trace)
        replay = dream_replay_builder.build(trace, transformed)

        if replay.eligible_for_training:
            replay_samples.append(replay)

    for replay in replay_samples:
        loss = compute_replay_loss(replay)
        loss.backward()
        optimizer.step()

    await sleep_validator.validate_and_checkpoint()
```

---

# §19 — Metrics and Monitoring

## Existing Metrics Preserved

| Metric | Definition |
|--------|------------|
| AUT | Average uncertainty per token |
| URR | Uncertainty resolution rate |
| TSM | Temporal stability metric |
| Clarification Efficiency | Tokens resolved / tokens requested |
| Column Complementarity | Diversity between frozen column outputs |
| SSS | Stream separation score |
| SCR | Sensorial contamination rate |
| SAR | Sensorial anomaly rate |
| Inferential k-utilization | % language queries expanding beyond k=1 |

## New v4.2 Metrics

| Metric | Definition | Direction |
|--------|------------|-----------|
| GCS — Grounding Consistency Score | % causal expectations confirmed by sensors | Maximize |
| BCR — Belief Contradiction Rate | % beliefs with active contradictory evidence | Monitor/minimize |
| SRS — Source Reliability Score | rolling reliability per sensor/source | Monitor |
| CIS — Causal Integrity Score | causal confirmations / total causal observations | Maximize |
| RGM — Resource Governor Mode Time | % time spent in each resource mode | Monitor |
| BTP — Background Training Pause Rate | % training cycles paused by governor | Monitor |
| ULP — User Latency Protection | latency reduction after governor action | Maximize |
| DRE — Dream Replay Eligibility | % Rule 30 replay samples passing safety gates | Monitor |
| DRS — Dream Replay Stability | post-sleep regression/drift score | Maximize |
| WSI — Wake/Sleep Integrity | sleep completed without emergency interruption | Monitor |
| PUC — Perceptual Uncertainty Count | unresolved perception events | Monitor |
| RSC — Resource-Safety Compliance | % tasks respecting current policy | Must be 100% |

## Logging Example

```python
wandb.log({
    # Uncertainty
    "AUT": avg_uncertainty,
    "URR": uncertainty_resolution_rate,
    "TSM": temporal_stability_metric,

    # Stream isolation
    "SSS": stream_separation_score,
    "SCR": sensorial_contamination_rate,
    "SAR": sensorial_anomaly_rate,

    # Grounding
    "GCS": grounding_consistency_score,
    "BCR": belief_contradiction_rate,
    "CIS": causal_integrity_score,

    # Resource governance
    "ResourceMode": current_resource_mode.value,
    "RGM_NORMAL": resource_mode_time["normal"],
    "BTP": background_training_pause_rate,
    "ULP": user_latency_protection,

    # Sleep/replay
    "DRE": dream_replay_eligibility,
    "DRS": dream_replay_stability,
    "WSI": wake_sleep_integrity,
})
```

---

# §20 — Updated Hypothesis Set

v4.2 preserves H1–H15 from v4.1, with corrected terminology and validation caveats.

## H1 — Uncertainty-Activation Interaction

High `U` prompts should trigger more column activations than low `U` prompts in language inference. Raw telemetry and raw sensorial observations should not trigger language columns.

Status: supported only as synthetic sanity check until live model validation.

## H3 — Stream-Separated Memory Retrieval

Adaptive-k retrieval should outperform fixed k=1 for language/perceptual inference, while sensorial observations use precise references and belief graph lookup.

## H7 — Layered Threshold Strategy

Thresholds should be separately derived for language inference, perceptual inference, belief contradiction, and resource anomaly detection.

## H8 — Stream Separation

Language/perceptual uncertainty distributions should be statistically separable from raw sensor/resource telemetry distributions.

## New v4.2 Hypotheses

### H16 — Multi-Sensor Grounding Reduces Belief Uncertainty

Diverse sensor confirmation reduces world-state belief uncertainty more than single-sensor confirmation.

- X = number and modality diversity of evidence sources
- Y = belief uncertainty reduction
- Z = source reliability
- Expected: relay + camera + power draw produces lower uncertainty than camera alone

### H17 — Causal Ordering Improves Grounding Reliability

Temporal cause/effect confirmation improves belief confidence more than correlation alone.

- X = causal window confirmation rate
- Y = belief confidence accuracy
- Z = contradiction rate
- Expected: causally ordered evidence produces fewer false beliefs than co-occurrence-only evidence

### H18 — Resource Governor Protects User Latency

Resource governor policy changes reduce p95 user latency during high system load.

- X = governor mode
- Y = p95 user latency
- Z = background training pause rate
- Expected: degraded/conserve mode lowers latency compared to unrestricted execution

### H19 — Resource-Aware Column Budgeting Preserves Quality Under Load

Reducing max active columns under resource pressure preserves acceptable answer quality while improving latency.

- X = max active columns allowed
- Y = response quality / uncertainty reduction
- Z = latency and VRAM pressure
- Expected: max_columns=1 under degraded mode beats unrestricted columns under overload

### H20 — Wake/Sleep Replay Improves Long-Term Stability

Sleep replay reduces recurring uncertainty without increasing catastrophic drift.

- X = sleep replay enabled/disabled
- Y = recurring uncertainty rate
- Z = post-sleep regression score
- Expected: replay improves recurring uncertainty with stable regression metrics

### H21 — Rule 30 Replay Adds Useful Variation

Rule 30 transformed memory traces produce useful training variation when gated by reconstruction and grounding consistency checks.

- X = Rule 30 steps
- Y = replay eligibility and uncertainty gain
- Z = drift/regression score
- Expected: low-to-moderate Rule 30 steps improve generalization; excessive steps degrade reconstruction

### H22 — Sensor Reliability Weighting Reduces False Grounding

Belief updates weighted by source reliability produce fewer false world-state beliefs than equal-weight evidence.

- X = reliability weighting enabled/disabled
- Y = false belief rate
- Z = contradiction recovery time
- Expected: reliability weighting lowers false belief rate

---

# §21 — Development Roadmap v4.2

## Phase 0 — Persistence Foundation

Before PNN training or advanced uncertainty learning, implement durable persistence.

Deliverables:

```text
- EventEnvelope
- event_log
- raw media references
- stream_type tagging
- append-only ingestion
- replayable event loader
```

Milestone:

```text
All input events can be replayed deterministically from storage.
```

## Phase 1 — Source Registry and Stream Routing

Deliverables:

```text
- SourceRegistry
- StreamType taxonomy
- routing tests
- UB write guard
- grounding store write path
- resource telemetry path
```

Milestone:

```text
Invalid streams cannot enter UB/CB.
```

## Phase 2 — Telemetry Resource Governor

Deliverables:

```text
- TelemetrySnapshot
- ResourceMode evaluation
- RESOURCE_POLICIES
- task scheduler integration
- latency protection tests
```

Milestone:

```text
Background training pauses automatically when user latency/resource pressure rises.
```

## Phase 3 — Grounding Evidence and Belief Graph

Deliverables:

```text
- EvidenceAtom
- WorldStateBelief
- CausalLink
- noisy-OR confidence update
- contradiction detection
- garage relay/camera/power-draw demo
```

Milestone:

```text
System can infer garage_light_is_on from cross-sensor confirmation and flag contradictions.
```

## Phase 4 — Corrected Uncertainty Monitor

Deliverables:

```text
- H, C, M, U computation
- LayeredUncertaintyMonitor
- language/perception threshold config
- live SmolLM-360M validation harness
```

Milestone:

```text
Uncertainty thresholds are derived from real model outputs, not synthetic assumptions.
```

## Phase 5 — Memory Retrieval v4.2

Deliverables:

```text
- stream-aware retrieval policies
- adaptive-k with resource governor cap
- belief graph retrieval
- retrieval metrics
```

Milestone:

```text
Language inference, perception, and belief lookup use different retrieval strategies.
```

## Phase 6 — Async Orchestration

Deliverables:

```text
- priority queue v4.2
- resource-aware scheduling
- perception extraction queue
- grounding update queue
- background train queue
```

Milestone:

```text
Resource governor controls actual execution, not just metrics.
```

## Phase 7 — Wake/Sleep

Deliverables:

```text
- WakeSleepPolicy
- cognitive state machine
- sleep eligibility checks
- interrupt handling
- wake reintegration checks
```

Milestone:

```text
System alternates awake/replay modes without corrupting active service.
```

## Phase 8 — Rule 30 Replay

Deliverables:

```text
- memory compression to binary/quantized planes
- Rule 30 transform
- replay sample builder
- reconstruction/grounding/drift gates
- replay training loop
```

Milestone:

```text
Rule 30 dream replay produces gated training candidates without direct memory corruption.
```

## Phase 9 — Progressive Columns and Training

Deliverables:

```text
- column activation budgeting
- corrected loss function
- background training
- EWC integration
- freeze/unfreeze/fuse logic
- column registry
```

Milestone:

```text
Columns specialize on uncertainty events while base remains frozen and latency protected.
```

## Phase 10 — Full Hypothesis Harness

Deliverables:

```text
- H1–H22 experiment runner
- synthetic vs live split
- statistical report generator
- reproducible artifacts
```

Milestone:

```text
Claims are supported by replayable experiments, not hand-tuned examples.
```

---

# §22 — Minimal POC Target

The fastest meaningful v4.2 demo is not full model training.

It is:

```text
Garage Grounding + Resource Governor Demo
```

## Inputs

```text
relay state
camera brightness
power draw
GPU temperature
queue depth
mock user prompt
```

## Demonstrates

```text
1. relay ON is persisted as sensor observation
2. camera brightness is persisted as raw/perceptual observation
3. power draw confirms causal effect
4. belief graph updates garage.light.on confidence
5. contradiction generates diagnostic uncertainty
6. GPU temp / queue depth changes resource mode
7. resource mode reduces column budget / pauses training
8. all events are replayable from persistence
```

## Why This POC Is Correct

It tests the architectural novelty before expensive training.

If this does not work, the PNN layer will only hide bugs.

---

# §23 — Publication / Claim Discipline

v4.2 should be strict about evidence.

Allowed language:

```text
Synthetic sanity check supports the proposed mechanism.
Live validation is pending.
The architecture predicts X.
The next experiment will test Y.
```

Avoid until validated:

```text
Empirically proven.
Publishable confirmation.
Confirmed on model behavior.
Production threshold.
```

Thresholds from synthetic Arbor results may be used as initial defaults, but not final values.

---

# Appendix A — Configuration Example

```yaml
keko:
  streams:
    default_stream: language_inference

  wake_sleep:
    awake_hours: 4
    sleep_hours: 3
    allow_interrupt_sleep_for_user: true
    allow_interrupt_sleep_for_emergency: true

  resource_governor:
    gpu_temp_emergency_c: 85
    gpu_temp_sleep_max_c: 65
    vram_emergency_ratio: 0.95
    latency_degraded_p95_s: 2.0
    user_queue_degraded: 5
    ram_conserve_ratio: 0.85

  grounding:
    default_source_reliability: 0.8
    contradiction_threshold: 0.35
    causal_confirmation_min: 0.7

  uncertainty:
    language_window: [0.35, 0.75]
    perceptual_window: [0.30, 0.80]
    relevance_threshold: 0.6

  memory:
    language_k_max: 5
    perceptual_k_max: 5
    dream_k_max: 8

  rule30:
    enabled: true
    max_steps: 8
    min_reconstruction_score: 0.75
    min_grounding_consistency_score: 0.80
    max_drift_score: 0.20

  training:
    background_batch_size: 32
    base_frozen: true
    bf16: true
```

---

# Appendix B — Recommended File Layout

```text
keko/
├── streams/
│   ├── types.py
│   ├── envelope.py
│   └── registry.py
├── persistence/
│   ├── event_log.py
│   ├── grounding_store.py
│   ├── belief_store.py
│   ├── telemetry_store.py
│   ├── ub_store.py
│   └── schema.sql
├── grounding/
│   ├── evidence.py
│   ├── causal.py
│   ├── belief_graph.py
│   └── contradiction.py
├── telemetry/
│   ├── snapshot.py
│   ├── governor.py
│   └── policies.py
├── uncertainty/
│   ├── measures.py
│   ├── monitor.py
│   └── routing.py
├── memory/
│   ├── bank.py
│   ├── retrieval_policy.py
│   └── replay_sampler.py
├── sleep/
│   ├── state_machine.py
│   ├── rule30.py
│   ├── dream_replay.py
│   └── validator.py
├── orchestration/
│   ├── priority.py
│   ├── queue.py
│   └── scheduler.py
├── columns/
│   ├── pnn.py
│   ├── router.py
│   ├── lifecycle.py
│   └── loss.py
└── experiments/
    ├── h1_uncertainty_activation.py
    ├── h16_grounding.py
    ├── h18_resource_governor.py
    └── h21_rule30_replay.py
```

---

# Appendix C — The v4.2 Core Thesis

The most compact version of v4.2:

```text
KEKO should not treat all uncertainty equally.

External sensors provide low-uncertainty grounding evidence.
Internal telemetry governs computational metabolism.
Raw perception is persisted and interpreted through perception models.
Contradictions become inferential uncertainty.
Language uncertainty drives clarification, tools, columns, and training.
Sleep consolidates memory using gated replay.
Rule 30 creates dream-like perturbations of memory traces.
The frozen base remains stable while columns adapt around uncertainty.
```

---

# End of Document

**KEKO Uncertainty Mechanism Specification v4.2**  
*Grounded Telemetry, Resource Governance, and Wake/Sleep Persistence Edition*  
Oscar Liguori — Independent Researcher — https://github.com/champi-ai/keko
