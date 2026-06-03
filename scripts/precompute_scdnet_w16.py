#!/usr/bin/env python3
"""Pre-compute SCD-Net w=16 features from w=1 native (Charades).

Reads SCD-Net features at 24 FPS (w=1, T~765), produces w=16 features
at ~1.5 FPS (T~47) via non-overlapping mean-pool over 16-frame windows.

Resume-safe: skips files already present in DST and loadable. Corrupted
files (e.g. from a previously killed run) are detected via np.load
and redone.

CPU-only, NFS-safe. Expected runtime: ~15-30 min on frontend.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

DEFAULT_SRC: str = (
    "/srv/storage/stars@storage3.sophia.grid5000.fr"
    "/mdiiorio/masters-thesis/Traineeship/MS-Temba"
    "/data/hf_features/Temporal_Action_Detection/charades_scdnet_full"
)
DEFAULT_DST: str = (
    "/srv/storage/stars@storage3.sophia.grid5000.fr"
    "/mdiiorio/masters-thesis/Traineeship/MS-Temba-v2"
    "/data/features_derived/charades_scdnet_w16"
)
WINDOW: int = 16
EXPECTED_D: int = 4096


def pool_w16(feat: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Non-overlapping mean-pool along the time axis.

    feat:   (T, D=4096) float32
    return: (T // window, D=4096) float32. Tail T % window frames dropped.
    """
    T, D = feat.shape
    n: int = T // window
    if n == 0:
        return feat.mean(axis=0, keepdims=True).astype(np.float32)
    trimmed = feat[: n * window]                                # (n*window, D)
    pooled = trimmed.reshape(n, window, D).mean(axis=1)         # (n, D)
    return pooled.astype(np.float32)


def is_already_done(out_path: Path) -> bool:
    """Resume-safe check: file exists AND loadable AND shape sane."""
    if not out_path.exists():
        return False
    try:
        arr = np.load(out_path)
        return arr.ndim == 2 and arr.shape[-1] == EXPECTED_D
    except Exception:
        # Corrupted (e.g. killed mid-write). Remove and redo.
        out_path.unlink(missing_ok=True)
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=str, default=DEFAULT_SRC)
    p.add_argument("--dst", type=str, default=DEFAULT_DST)
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    assert src.is_dir(), f"SRC dir not found: {src}"
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.npy"))
    print(f"[precompute] SRC: {src}")
    print(f"[precompute] DST: {dst}")
    print(f"[precompute] Found {len(files)} .npy files, window={args.window}")

    if args.dry_run:
        for f in files[:5]:
            print(f"  would process: {f.name}")
        print(f"  ... ({len(files)} total). No writes performed.")
        return

    n_done = n_skip = n_err = 0
    Ts_in: list[int] = []
    Ts_out: list[int] = []

    for f in tqdm(files):
        out_path = dst / f.name
        if is_already_done(out_path):
            n_skip += 1
            continue
        try:
            feat = np.load(f)
            if feat.ndim != 2:
                raise ValueError(f"Expected 2D, got {feat.shape}")
            if feat.shape[-1] != EXPECTED_D:
                if feat.shape[0] == EXPECTED_D:
                    feat = feat.T
                else:
                    raise ValueError(f"D={EXPECTED_D} not found in {feat.shape}")
            Ts_in.append(feat.shape[0])
            pooled = pool_w16(feat, window=args.window)
            Ts_out.append(pooled.shape[0])
            # Direct write. np.save adds .npy only if path doesn't end with it,
            # so passing out_path (already .npy) is correct.
            np.save(out_path, pooled)
            n_done += 1
        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")
            n_err += 1

    print(f"\n[precompute] Done: {n_done} written, {n_skip} skipped, {n_err} errors")
    if Ts_in:
        Ts_in_s = sorted(Ts_in); Ts_out_s = sorted(Ts_out)
        print(f"[precompute] T_in  min/median/max: "
              f"{Ts_in_s[0]} / {Ts_in_s[len(Ts_in_s)//2]} / {Ts_in_s[-1]}")
        print(f"[precompute] T_out min/median/max: "
              f"{Ts_out_s[0]} / {Ts_out_s[len(Ts_out_s)//2]} / {Ts_out_s[-1]}")


if __name__ == "__main__":
    main()
