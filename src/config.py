"""Central configuration. Every PSD method must read its parameters from here
so all methods see the same data and the same frequency grid."""
from pathlib import Path

# ---- paths ----
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "thai-ser" / "data"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

# ---- dataset selection ----
MIC = "mic_clip"
ROOMS = ("A", "B")                      # zoom rows have no mic_clip
CLASSES = ("Neutral", "Happy", "Angry", "Sad")
LABEL_COL = "majority_emo"
GROUP_COL = "actor_id"                  # speaker-independent split

# ---- signal ----
SOURCE_SR = 44100
SR = 16000                              # resample 44.1k -> 16k (I/D = 160/441)
PRE_EMPHASIS = 0.97                     # y[n] = x[n] - a*x[n-1]

# ---- framing (shared by every PSD method) ----
FRAME_LEN = 1024                        # 64 ms @ 16 kHz
HOP = 512
NFFT = 1024                             # same frequency grid for every method

# ---- PSD method parameters ----
SEG_LEN = 256                           # Bartlett / Welch segment length
WELCH_OVERLAP = 0.5                     # -> 7 segments per frame
WELCH_WINDOW = "hann"
BT_MAX_LAG = 256                        # Blackman-Tukey max lag M
BT_LAG_WINDOW = "bartlett"              # triangular lag window keeps PSD >= 0
AR_ORDER = 18                           # common LPC rule of thumb: fs[kHz] + 2

PSD_METHODS = ("periodogram", "bartlett", "welch", "blackman_tukey", "ar")

# ---- pitch ----
PITCH_FMIN = 50.0
PITCH_FMAX = 500.0

# ---- evaluation ----
N_FOLDS = 5
SEED = 42
