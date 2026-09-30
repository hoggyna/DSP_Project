import numpy as np
import pytest
import scipy.signal as ss

from src import config as cfg
from src import psd

FS = 1000


def assignment_signal(N=1024, sigma=0.5, seed=42):
    """Same test signal as Assignment 1: 100 Hz + 110 Hz + weak 300 Hz + noise."""
    n = np.arange(N)
    rng = np.random.default_rng(seed)
    return (np.sin(2 * np.pi * 100 * n / FS)
            + np.sin(2 * np.pi * 110 * n / FS)
            + 0.01 * np.sin(2 * np.pi * 300 * n / FS)
            + rng.normal(0, sigma, N))


@pytest.fixture
def x():
    return assignment_signal()


# ------------------------------------------------ agreement with scipy

def test_window_matches_scipy():
    for name in ("hann", "hamming", "boxcar"):
        np.testing.assert_allclose(psd.get_window(name, 256), ss.get_window(name, 256), atol=1e-12)


@pytest.mark.parametrize("window,nfft", [("rect", None), ("rect", 2048), ("hann", None), ("hann", 2048)])
def test_periodogram_matches_scipy(x, window, nfft):
    f, P = psd.periodogram(x, FS, window, nfft)
    fs_, Ps = ss.periodogram(x, FS, window="boxcar" if window == "rect" else window,
                             nfft=nfft, detrend=False, scaling="density")
    np.testing.assert_allclose(f, fs_)
    np.testing.assert_allclose(P, Ps, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("nfft", [None, 1024])
def test_bartlett_matches_scipy(x, nfft):
    f, P = psd.bartlett(x, FS, 128, nfft)
    fs_, Ps = ss.welch(x, FS, window="boxcar", nperseg=128, noverlap=0, nfft=nfft,
                       detrend=False, scaling="density")
    np.testing.assert_allclose(f, fs_)
    np.testing.assert_allclose(P, Ps, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("nfft", [None, 1024])
def test_welch_matches_scipy(x, nfft):
    f, P = psd.welch(x, FS, 128, 0.5, "hann", nfft)
    fs_, Ps = ss.welch(x, FS, window="hann", nperseg=128, noverlap=64, nfft=nfft,
                       detrend=False, scaling="density")
    np.testing.assert_allclose(f, fs_)
    np.testing.assert_allclose(P, Ps, rtol=1e-10, atol=1e-14)


def test_bartlett_uses_whole_signal():
    """Notebook bug: K was fixed at 8. K must grow with the signal."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=4096)
    _, P_full = psd.bartlett(x, FS, 128)
    _, P_head = psd.bartlett(x[:1024], FS, 128)
    assert not np.allclose(P_full, P_head)


# ------------------------------------------------------- Blackman-Tukey

def test_blackman_tukey_full_lag_rect_equals_periodogram(x):
    """With every lag and no lag window, the correlogram is the periodogram."""
    N = len(x)
    nfft = 2 * N
    _, P_bt = psd.blackman_tukey(x, FS, M=N, lag="rect", nfft=nfft)
    _, P_per = psd.periodogram(x, FS, "rect", nfft)
    np.testing.assert_allclose(P_bt, P_per, rtol=1e-8, atol=1e-12)


def test_blackman_tukey_bartlett_lag_is_nonnegative(x):
    _, P = psd.blackman_tukey(x, FS, M=200, lag="bartlett", nfft=1024)
    assert P.min() > -1e-12 * P.max()


def test_blackman_tukey_rejects_small_nfft(x):
    with pytest.raises(ValueError):
        psd.blackman_tukey(x, FS, M=200, nfft=256)


def test_biased_autocorr_matches_direct(x):
    N, M = len(x), 50
    direct = np.array([x[: N - m] @ x[m:] / N for m in range(M + 1)])
    np.testing.assert_allclose(psd.biased_autocorr(x, M), direct, atol=1e-10)


# --------------------------------------------------------------- AR

def test_ar_recovers_known_coefficients():
    """x[n] = 1.5 x[n-1] - 0.9 x[n-2] + w[n]  ->  A(z) = 1 - 1.5 z^-1 + 0.9 z^-2."""
    rng = np.random.default_rng(1)
    w = rng.normal(0, 1, 50_000)
    x = ss.lfilter([1], [1, -1.5, 0.9], w)[1000:]
    a, sigma2 = psd.ar_coeffs(x, 2)
    np.testing.assert_allclose(a, [-1.5, 0.9], atol=0.02)
    assert sigma2 == pytest.approx(1.0, rel=0.05)


def test_ar_silent_frame_gives_zero():
    f, P = psd.ar_psd(np.zeros(1024), FS, 18, 1024)
    assert np.all(P == 0)


# ------------------------------------------------- behaviour vs theory

@pytest.mark.parametrize("method", cfg.PSD_METHODS)
def test_peak_at_sine_frequency(x, method):
    """Every method must put its strongest peak at the 100/110 Hz pair."""
    f, P = {
        "periodogram": lambda: psd.periodogram(x, FS, "rect", 1024),
        "bartlett": lambda: psd.bartlett(x, FS, 256, 1024),
        "welch": lambda: psd.welch(x, FS, 256, 0.5, "hann", 1024),
        "blackman_tukey": lambda: psd.blackman_tukey(x, FS, 256, "bartlett", 1024),
        "ar": lambda: psd.ar_psd(x, FS, 18, 1024),
    }[method]()
    assert 95 <= f[np.argmax(P)] <= 115


@pytest.mark.parametrize("method", cfg.PSD_METHODS)
def test_parseval_total_power(method):
    """Area under the one-sided PSD ~= signal power."""
    rng = np.random.default_rng(2)
    x = ss.lfilter([1], [1, -0.5], rng.normal(size=1024))
    nfft = 8192
    f, P = {
        "periodogram": lambda: psd.periodogram(x, FS, "rect", nfft),
        "bartlett": lambda: psd.bartlett(x, FS, 256, nfft),
        "welch": lambda: psd.welch(x, FS, 256, 0.5, "hann", nfft),
        "blackman_tukey": lambda: psd.blackman_tukey(x, FS, 256, "bartlett", nfft),
        "ar": lambda: psd.ar_psd(x, FS, 18, nfft),
    }[method]()
    power = np.sum(P) * FS / nfft
    exact = method in ("periodogram", "blackman_tukey", "ar")   # all built from r[0]
    assert power == pytest.approx(np.mean(x ** 2), rel=1e-6 if exact else 0.1)


def test_averaging_reduces_variance():
    """White noise has a flat PSD. Periodogram spread stays ~100 %, Bartlett
    with K segments drops to ~1/sqrt(K) (CH02-03)."""
    rng = np.random.default_rng(3)
    spread = {"periodogram": [], "bartlett": [], "welch": []}
    for _ in range(50):
        x = rng.normal(size=1024)
        for name, (f, P) in {
            "periodogram": psd.periodogram(x, FS, "rect"),
            "bartlett": psd.bartlett(x, FS, 256),
            "welch": psd.welch(x, FS, 256, 0.5, "hann"),
        }.items():
            P = P[1:-1]                        # drop DC / Nyquist
            spread[name].append(P.std() / P.mean())
    s = {k: np.mean(v) for k, v in spread.items()}
    assert s["periodogram"] > 0.85
    assert 0.4 < s["bartlett"] < 0.6          # K = 4 -> 1/sqrt(4) = 0.5
    assert s["welch"] < s["bartlett"]         # 7 overlapped segments


# ------------------------------------------------------ project setup

def test_batch_equals_loop():
    rng = np.random.default_rng(4)
    frames = rng.normal(size=(5, cfg.FRAME_LEN))
    for method in cfg.PSD_METHODS:
        _, Pb = psd.estimate(method, frames, cfg.SR, cfg)
        for i in range(len(frames)):
            _, Pi = psd.estimate(method, frames[i], cfg.SR, cfg)
            np.testing.assert_allclose(Pb[i], Pi, rtol=1e-10, atol=1e-18)


def test_all_methods_share_one_grid():
    rng = np.random.default_rng(5)
    frame = rng.normal(size=cfg.FRAME_LEN)
    grids = [psd.estimate(m, frame, cfg.SR, cfg)[0] for m in cfg.PSD_METHODS]
    for g in grids:
        np.testing.assert_array_equal(g, grids[0])
    assert len(grids[0]) == cfg.NFFT // 2 + 1


def test_shortest_utterance_is_usable():
    """Shortest file is 0.44 s -> 7040 samples at 16 kHz."""
    n = int(0.44 * cfg.SR)
    frames = psd.sliding_window_view(np.random.default_rng(6).normal(size=n),
                                     cfg.FRAME_LEN)[::cfg.HOP]
    assert len(frames) >= 10
    for method in cfg.PSD_METHODS:
        _, P = psd.estimate(method, frames, cfg.SR, cfg)
        assert np.all(np.isfinite(P))


def test_too_short_signal_raises():
    with pytest.raises(ValueError):
        psd.bartlett(np.ones(100), FS, 256)
    with pytest.raises(ValueError):
        psd.blackman_tukey(np.ones(100), FS, 256)
