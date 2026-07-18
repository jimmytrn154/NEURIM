import numpy as np
import pytest

from src.acquisition.acquisition import Acquisition, MockBackend
from src.acquisition.ring_buffer import Epoch
from src.experiment.protocol import (
    HeadlessPresenter,
    ProtocolConfig,
    run_pairwise_trial,
    run_satisfaction_trial,
)
from src.experiment.trial_store import TrialStore

CHANNELS = ["AF3", "AF4", "F3", "F4", "P7", "P8", "O1", "O2"]


def _acq(**kw):
    backend = MockBackend(CHANNELS, sample_rate_hz=128, **kw)
    acq = Acquisition(backend, CHANNELS, 128)
    acq.connect()
    acq.pump_seconds(1.0)  # warm the buffer
    return acq


# ---- acquisition runner -----------------------------------------------------
def test_pump_fills_ring_and_tracks_contact_and_motion():
    acq = _acq()
    acq.pump_seconds(2.0)
    assert acq.ring.n_pushed > 300
    assert set(acq.latest_contact()) == set(CHANNELS)
    t0, t1 = acq.ring.span()
    assert acq.motion_rms(t0, t1) is not None


def test_extract_epoch_and_quality_valid():
    acq = _acq()
    onset = acq.now()
    acq.pump_seconds(3.0)
    ep = acq.extract_epoch(onset, 0.5, 2.5)
    assert isinstance(ep, Epoch)
    assert ep.data.shape[0] == len(CHANNELS)
    res = acq.evaluate(ep)
    assert res.status == "valid", res.reasons


def test_injected_bad_contact_makes_quality_not_valid():
    acq = _acq()
    acq.backend.inject_bad_contact = set(CHANNELS[:6])  # most channels no contact
    onset = acq.now()
    acq.pump_seconds(3.0)
    res = acq.evaluate(acq.extract_epoch(onset, 0.5, 2.5))
    assert res.status in ("retry", "invalid")
    assert not res.is_valid or res.status == "retry"


# ---- protocol driver --------------------------------------------------------
def test_satisfaction_trial_produces_valid_candidate_epoch():
    acq = _acq(signal_gain=0.8)
    cfg = ProtocolConfig()
    res = run_satisfaction_trial(acq, HeadlessPresenter(), cfg, true_satisfaction=0.8,
                                 rng=np.random.default_rng(0))
    assert "candidate" in res.epochs
    ep = res.epochs["candidate"]
    assert ep.duration_s == pytest.approx(2.0)  # eval_end - eval_start
    assert res.quality["candidate"].status == "valid"
    # markers include a CANDIDATE_ONSET
    assert any(m.name == "CANDIDATE_ONSET" for m in res.markers)


def test_injected_signal_separates_satisfied_from_dissatisfied():
    """The mock's satisfaction modulation must show up in the candidate epoch."""
    def mean_parietal_power(true_s):
        acq = _acq(signal_gain=0.9, seed=3)
        res = run_satisfaction_trial(acq, HeadlessPresenter(), ProtocolConfig(),
                                     true_satisfaction=true_s, rng=np.random.default_rng(1))
        ep = res.epochs["candidate"]
        idx = [ep.channels.index(c) for c in ("P7", "P8", "O1", "O2")]
        return float(np.mean(np.var(ep.data[idx], axis=1)))

    assert mean_parietal_power(0.9) > mean_parietal_power(-0.9)


def test_pairwise_trial_produces_two_epochs():
    acq = _acq()
    res = run_pairwise_trial(acq, HeadlessPresenter(), ProtocolConfig(),
                             sat_a=0.8, sat_b=-0.8, rng=np.random.default_rng(0))
    assert set(res.epochs) == {"A", "B"}
    assert all(res.quality[k].status == "valid" for k in ("A", "B"))


# ---- trial store ------------------------------------------------------------
def test_trial_store_roundtrip_and_resume(tmp_path):
    acq = _acq()
    store = TrialStore(tmp_path, "sub-001", "ses-001")
    store.write_session({"protocol": "satisfaction"})
    res = run_satisfaction_trial(acq, HeadlessPresenter(), ProtocolConfig(),
                                 true_satisfaction=0.8, rng=np.random.default_rng(0))
    tid = store.next_trial_id()
    assert tid == 1
    store.save_trial(tid, {"true_label": "satisfied", "usable_label": True}, res.epochs)

    # Round-trip: raw array is recoverable with channel order intact.
    arrays = store.load_trial_arrays(1)
    assert arrays["candidate_eeg"].shape[0] == len(CHANNELS)
    assert list(arrays["channels"]) == CHANNELS
    manifest = store.load_manifest()
    assert manifest[0]["true_label"] == "satisfied"
    assert manifest[0]["sample_rate_hz"] == 128

    # Resume: a new store instance continues at the next id, no overwrite.
    store2 = TrialStore(tmp_path, "sub-001", "ses-001")
    assert store2.next_trial_id() == 2


def test_invalid_trial_saved_but_not_usable_label(tmp_path):
    acq = _acq()
    acq.backend.inject_bad_contact = set(CHANNELS)  # all dead -> invalid
    store = TrialStore(tmp_path, "sub-x", "ses-001")
    res = run_satisfaction_trial(acq, HeadlessPresenter(), ProtocolConfig(),
                                 true_satisfaction=-0.8, rng=np.random.default_rng(0))
    q = res.quality["candidate"]
    store.save_trial(1, {"quality_status": q.status, "usable_label": q.is_valid}, res.epochs)
    row = store.load_manifest()[0]
    # Raw is still saved for later reanalysis, but it is not a usable label.
    assert store.load_trial_arrays(1)["candidate_eeg"].size > 0
    assert row["usable_label"] is False
