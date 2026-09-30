"""Power spectral density estimators (CH02-03, CH05-01).

Rewritten from example_code/Assignment_DSP.ipynb with its bugs fixed.

Conventions (match scipy.signal with scaling="density", detrend=False):
- every function works on the last axis, so x may be one signal (N,) or a
  batch of frames (..., N)
- returns (f, P): one-sided PSD in power/Hz on the grid rfftfreq(nfft, 1/fs)
- sum(P) * fs / nfft ~= mean(x**2)  (Parseval)
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.linalg import solve_toeplitz


# ---------------------------------------------------------------- helpers

def get_window(window, L):
    """Periodic window of length L (same as scipy.signal.get_window default)."""
    if isinstance(window, np.ndarray):
        if window.shape != (L,):
            raise ValueError(f"window length {window.shape} != {L}")
        return window.astype(float)
    n = np.arange(L)
    if window in ("rect", "boxcar"):
        return np.ones(L)
    if window == "hann":
        return 0.5 - 0.5 * np.cos(2 * np.pi * n / L)
    if window == "hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / L)
    raise ValueError(f"unknown window: {window}")


def _one_sided(P2, nfft):
    """Fold a two-sided density sampled on rfft bins into a one-sided one."""
    P = P2.copy()
    if nfft % 2 == 0:
        P[..., 1:-1] *= 2          # DC and Nyquist appear once
    else:
        P[..., 1:] *= 2
    return P


def _check_nfft(nfft, need):
    if nfft < need:
        raise ValueError(f"nfft={nfft} must be >= {need}")


def biased_autocorr(x, max_lag):
    """r[m] = (1/N) sum_n x[n] x[n+m], m = 0..max_lag (FFT based)."""
    x = np.asarray(x, dtype=float)
    N = x.shape[-1]
    if not 0 <= max_lag < N:
        raise ValueError(f"max_lag={max_lag} must be in [0, {N - 1}]")
    n_ac = 1 << int(np.ceil(np.log2(2 * N - 1)))   # avoid circular wrap
    X = np.fft.rfft(x, n=n_ac)
    r = np.fft.irfft(np.abs(X) ** 2, n=n_ac)[..., : max_lag + 1]
    return r / N


# ------------------------------------------------------- nonparametric

def periodogram(x, fs, window="rect", nfft=None):
    """Periodogram (rect window) or modified periodogram (any other window)."""
    x = np.asarray(x, dtype=float)
    N = x.shape[-1]
    nfft = N if nfft is None else nfft
    _check_nfft(nfft, N)
    w = get_window(window, N)
    X = np.fft.rfft(x * w, n=nfft)
    P2 = np.abs(X) ** 2 / (fs * np.sum(w ** 2))
    return np.fft.rfftfreq(nfft, 1 / fs), _one_sided(P2, nfft)


def _averaged_periodogram(x, fs, L, step, window, nfft):
    x = np.asarray(x, dtype=float)
    N = x.shape[-1]
    if N < L:
        raise ValueError(f"signal length {N} < segment length {L}")
    nfft = L if nfft is None else nfft
    segs = sliding_window_view(x, L, axis=-1)[..., ::step, :]   # (..., K, L)
    f, P = periodogram(segs, fs, window=window, nfft=nfft)
    return f, P.mean(axis=-2)


def bartlett(x, fs, L, nfft=None):
    """Bartlett: average periodograms of K = N // L non-overlapping segments."""
    return _averaged_periodogram(x, fs, L, L, "rect", nfft)


def welch(x, fs, L, overlap=0.5, window="hann", nfft=None):
    """Welch: average modified periodograms of overlapping segments."""
    step = L - int(round(L * overlap))
    if step < 1:
        raise ValueError(f"overlap={overlap} leaves no hop")
    return _averaged_periodogram(x, fs, L, step, window, nfft)


def lag_window(name, M):
    """Symmetric lag window w[m], m = 0..M-1 (w[-m] = w[m])."""
    m = np.arange(M)
    if name in ("rect", "boxcar"):
        return np.ones(M)
    if name == "bartlett":
        return 1 - m / M                      # Fejer kernel >= 0 -> PSD >= 0
    if name == "hann":
        return 0.5 + 0.5 * np.cos(np.pi * m / M)
    raise ValueError(f"unknown lag window: {name}")


def blackman_tukey(x, fs, M, lag="bartlett", nfft=None):
    """Blackman-Tukey correlogram: FT of the windowed biased autocorrelation
    using lags |m| < M."""
    x = np.asarray(x, dtype=float)
    N = x.shape[-1]
    if not 1 <= M <= N:
        raise ValueError(f"M={M} must be in [1, {N}]")
    nfft = max(N, 2 * M - 1) if nfft is None else nfft
    _check_nfft(nfft, 2 * M - 1)
    rw = biased_autocorr(x, M - 1) * lag_window(lag, M)
    c = np.zeros(x.shape[:-1] + (nfft,))
    c[..., :M] = rw
    if M > 1:
        c[..., -(M - 1):] = rw[..., :0:-1]     # r[-m] = r[m]
    P2 = np.fft.rfft(c).real / fs
    return np.fft.rfftfreq(nfft, 1 / fs), _one_sided(P2, nfft)


# ---------------------------------------------------------- parametric

def ar_coeffs(x, order):
    """AR model by linear prediction (autocorrelation method, CH05-01).

    Solves the MMSE normal equations R a = -r for A(z) = 1 + sum a_k z^-k.
    Returns (a, sigma2) with a shape (..., order), sigma2 shape (...).
    """
    x = np.asarray(x, dtype=float)
    lead = x.shape[:-1]
    r = biased_autocorr(x, order).reshape(-1, order + 1)
    a = np.zeros((r.shape[0], order))
    sigma2 = np.zeros(r.shape[0])
    for i, ri in enumerate(r):
        if ri[0] <= 0:                          # silent frame -> zero model
            continue
        a[i] = solve_toeplitz(ri[:order], -ri[1:])
        sigma2[i] = ri[0] + ri[1:] @ a[i]
    return a.reshape(lead + (order,)), sigma2.reshape(lead)


def ar_psd(x, fs, order, nfft=None):
    """AR spectrum: P(f) = sigma2 / |A(e^jw)|^2."""
    x = np.asarray(x, dtype=float)
    nfft = x.shape[-1] if nfft is None else nfft
    _check_nfft(nfft, order + 1)
    a, sigma2 = ar_coeffs(x, order)
    A = np.fft.rfft(np.concatenate([np.ones(a.shape[:-1] + (1,)), a], axis=-1), n=nfft)
    P2 = sigma2[..., None] / np.abs(A) ** 2 / fs
    return np.fft.rfftfreq(nfft, 1 / fs), _one_sided(P2, nfft)


# ------------------------------------------------------------ dispatch

def estimate(method, x, fs, cfg):
    """Run one PSD method with parameters from the config module."""
    if method == "periodogram":
        return periodogram(x, fs, "rect", cfg.NFFT)
    if method == "bartlett":
        return bartlett(x, fs, cfg.SEG_LEN, cfg.NFFT)
    if method == "welch":
        return welch(x, fs, cfg.SEG_LEN, cfg.WELCH_OVERLAP, cfg.WELCH_WINDOW, cfg.NFFT)
    if method == "blackman_tukey":
        return blackman_tukey(x, fs, cfg.BT_MAX_LAG, cfg.BT_LAG_WINDOW, cfg.NFFT)
    if method == "ar":
        return ar_psd(x, fs, cfg.AR_ORDER, cfg.NFFT)
    raise ValueError(f"unknown PSD method: {method}")
