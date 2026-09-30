import json

import pytest

from negative_space.promoters import (KNOWLEDGE_PROMOTER, Behaviour, Region, RegionThresholds,
                                      StreamRoutingError, StreamType, assign_region,
                                      keko_window_behaviour, make_event, suggested_k)

TH = RegionThresholds(tau_outside=10.0, tau_flip=0.1, p_boundary=0.5, confidence_high=0.7)
UB_COLUMNS = {"event_id", "timestamp", "stream_type", "source_id", "uncertainty_score",
              "uncertainty_type", "context_vector_ref", "payload_ref", "column_id", "session_id",
              "node_id", "metadata_json"}   # KEKO v4.2 §15 uncertainty_buffer


def test_region_precedence():
    assert assign_region(1, 1.0, 0.1, False, 0.9, TH) == Region.INTERIOR
    assert assign_region(1, 0.05, 0.1, False, 0.9, TH) == Region.BOUNDARY
    assert assign_region(1, 1.0, 0.8, False, 0.9, TH) == Region.BOUNDARY
    assert assign_region(20, 1.0, 0.1, False, 0.9, TH) == Region.OUTSIDE
    assert assign_region(20, 1.0, 0.1, True, 0.9, TH) == Region.HOLE
    assert assign_region(20, 1.0, 0.1, True, 0.5, TH) == Region.OUTSIDE  # not confident -> not a hole


def test_behaviour_mapping_and_keko_window():
    b = KNOWLEDGE_PROMOTER.behaviours
    assert b[Region.INTERIOR] == Behaviour.NONE
    assert b[Region.BOUNDARY] == Behaviour.CLARIFY_RETRIEVE
    assert b[Region.HOLE] == Behaviour.GROW_REPLAY
    assert b[Region.OUTSIDE] == Behaviour.ABSTAIN_GROUND
    assert keko_window_behaviour(0.5) == Behaviour.CLARIFY_RETRIEVE
    assert keko_window_behaviour(0.9) == Behaviour.NONE
    assert keko_window_behaviour(0.2) == Behaviour.NONE


def test_events_and_ub_row():
    assert make_event(KNOWLEDGE_PROMOTER, Region.INTERIOR, 0.1) is None
    ev = make_event(KNOWLEDGE_PROMOTER, Region.HOLE, 0.8, hole_persistence=0.5)
    row = ev.to_ub_row()
    assert set(row) == UB_COLUMNS
    assert row["stream_type"] == "language_inference"
    meta = json.loads(row["metadata_json"])
    assert meta["region"] == "hole" and meta["replay_weight"] == pytest.approx(0.8 * 1.5)
    assert ev.severity == "high"
    ev = make_event(KNOWLEDGE_PROMOTER, Region.BOUNDARY, 0.5)
    assert ev.metadata["suggested_k"] == suggested_k(0.5) and 1 <= suggested_k(0.5) <= 5


@pytest.mark.parametrize("st", [StreamType.SENSORIAL_OBSERVATION, StreamType.RESOURCE_TELEMETRY,
                                StreamType.RAW_PERCEPTION, StreamType.GROUNDING_EVIDENCE])
def test_stream_invariants(st):
    with pytest.raises(StreamRoutingError):
        make_event(KNOWLEDGE_PROMOTER, Region.OUTSIDE, 0.9, stream_type=st)
