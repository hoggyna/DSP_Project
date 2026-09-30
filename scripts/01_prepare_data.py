"""Step 1 of the pipeline: raw parquet -> data/processed/wav/*.wav + metadata.csv

For every selected row (4 classes, rooms A + B, mic_clip):
    decode FLAC -> mono -> resample 44.1k -> 16k -> trim edge silence -> 16-bit WAV

Usage:
    .venv/bin/python scripts/01_prepare_data.py [--overwrite] [--workers N]
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src import config as cfg
from src import dataset, preprocess

WAV_DIR = cfg.PROCESSED_DIR / "wav"
META_CSV = cfg.PROCESSED_DIR / "metadata.csv"


def process_shard(path, overwrite):
    records, skipped = [], []
    for meta, name, audio in dataset.iter_shard(path):
        if audio is None:
            skipped.append(meta["audio_id"])
            continue
        out = WAV_DIR / (Path(name).stem + ".wav")
        x, sr = dataset.decode(audio)
        y = preprocess.load_for_storage(x, sr)
        peak = float(np.max(np.abs(y)))
        if overwrite or not out.exists():
            sf.write(out, np.clip(y, -1.0, 1.0), cfg.SR, subtype="PCM_16")
        records.append({
            "audio_id": meta["audio_id"],
            "file": str(out.relative_to(cfg.PROCESSED_DIR)),
            "emotion": meta[cfg.LABEL_COL],
            "actor_id": meta["actor_id"],
            "session_id": meta["session_id"],
            "room_type": meta["room_type"],
            "turn_type": meta["turn_type"],
            "script_intensity": meta["script_intensity"],
            "actor_gender": meta["actor_gender"],
            "agreement": meta["agreement"],
            "src_sr": sr,
            "src_channels": 1 if x.ndim == 1 else x.shape[1],
            "src_dur": len(x) / sr,
            "dur": len(y) / cfg.SR,
            "peak": peak,
            "shard": path.name,
        })
    return records, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    WAV_DIR.mkdir(parents=True, exist_ok=True)
    files = dataset.parquet_files()
    records, skipped = [], []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(process_shard, f, args.overwrite) for f in files]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="shards"):
            r, s = fut.result()
            records += r
            skipped += s

    df = pd.DataFrame(records).sort_values("audio_id").reset_index(drop=True)
    df.to_csv(META_CSV, index=False)

    print(f"\nwrote {len(df)} files -> {WAV_DIR}")
    print(f"metadata -> {META_CSV}")
    print(f"skipped (no {cfg.MIC} audio): {len(skipped)}")
    print(f"peak > 1.0 (clipped on save): {(df.peak > 1.0).sum()}")
    print(df.emotion.value_counts().to_string())
    print(df.room_type.value_counts().to_string())
    print(f"speakers: {df.actor_id.nunique()}")
    print(f"duration before trim: {df.src_dur.sum() / 3600:.2f} h, after: {df.dur.sum() / 3600:.2f} h")


if __name__ == "__main__":
    main()
