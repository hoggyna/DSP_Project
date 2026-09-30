"""Read the Thai-SER parquet shards and select the rows used in this project."""
import io

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

from src import config as cfg

META_COLS = ["audio_id", "majority_emo", "assigned_emo", "agreement", "session_id",
             "room_type", "actor_id", "actor_gender", "actor_age", "turn_type",
             "script_intensity"]


def parquet_files():
    files = sorted(cfg.RAW_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {cfg.RAW_DIR}")
    return files


def is_selected(df):
    """Rows kept by the project: 4 classes, studio rooms A + B."""
    return df[cfg.LABEL_COL].isin(cfg.CLASSES) & df["room_type"].isin(cfg.ROOMS)


def load_metadata(selected_only=True):
    """Metadata of every row (no audio). Adds `shard` = parquet file name."""
    parts = []
    for f in parquet_files():
        d = pq.read_table(f, columns=META_COLS).to_pandas()
        d["shard"] = f.name
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    return df[is_selected(df)].reset_index(drop=True) if selected_only else df


def decode(audio_bytes):
    """FLAC bytes -> (float64 array (n,) or (n, ch), sample rate)."""
    return sf.read(io.BytesIO(audio_bytes), dtype="float64")


def iter_shard(path):
    """Yield (meta dict, source file name, audio bytes or None) for the
    selected rows of one parquet shard. Only the chosen mic is read."""
    t = pq.read_table(path, columns=META_COLS + [cfg.MIC])
    meta = t.select(META_COLS).to_pandas()
    keep = np.flatnonzero(is_selected(meta).to_numpy())
    mic = t.column(cfg.MIC)
    for i in keep:
        s = mic[int(i)].as_py()
        audio = s["bytes"] if s and s["bytes"] else None
        name = s["path"] if s else None
        yield meta.iloc[i].to_dict(), name, audio
