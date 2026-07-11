import numpy as np

from src.generator.latent_morph import LatentMorpher, morph_path


def test_step_never_exceeds_max_step():
    m = LatentMorpher(np.zeros(8), max_step=0.1)
    target = np.full(8, 5.0)  # far away
    prev = m.z.copy()
    for _ in range(50):
        cur = m.step(target)
        assert np.linalg.norm(cur - prev) <= 0.1 + 1e-9  # bounded per-frame motion
        prev = cur


def test_follower_reaches_target():
    m = LatentMorpher(np.zeros(4), max_step=0.2)
    target = np.array([1.0, -1.0, 0.5, -0.5])
    for _ in range(200):
        m.step(target)
    assert m.at_target(target, tol=1e-2)


def test_retarget_midflight_has_no_jump():
    m = LatentMorpher(np.zeros(3), max_step=0.1)
    a = np.array([2.0, 0.0, 0.0])
    prev = m.z.copy()
    for _ in range(5):
        prev = m.step(a)
    # Abruptly change the target; the very next step must still be bounded.
    b = np.array([0.0, -3.0, 1.0])
    cur = m.step(b)
    assert np.linalg.norm(cur - prev) <= 0.1 + 1e-9


def test_smoothing_eases_arrival():
    # With smoothing, the final approach steps shrink (ease-in), so the last
    # step near the target is smaller than the cruise step.
    m = LatentMorpher(np.zeros(2), max_step=0.2, smoothing=0.3)
    target = np.array([1.0, 0.0])
    steps = []
    prev = m.z.copy()
    for _ in range(60):
        cur = m.step(target)
        steps.append(np.linalg.norm(cur - prev))
        prev = cur
    assert max(steps) <= 0.2 + 1e-9
    assert steps[-1] < steps[0]  # arrival slower than cruise


def test_frames_to_target():
    m = LatentMorpher(np.zeros(2), max_step=0.1)
    assert m.frames_to_target(np.array([1.0, 0.0])) == 10


def test_morph_path_endpoints():
    z0 = np.zeros(3)
    z1 = np.array([1.0, 2.0, -1.0])
    path = morph_path(z0, z1, n=5)
    assert path.shape == (5, 3)
    assert np.allclose(path[-1], z1)          # ends exactly at z_new
    assert not np.allclose(path[0], z0)       # first frame already moved off z_old
    # monotonically increasing distance from z0 (smooth advance)
    dists = np.linalg.norm(path - z0, axis=1)
    assert np.all(np.diff(dists) > 0)
