"""Per-class and aggregate metrics logger for MS-Temba training.

Writes three artefacts in output_dir:
- metrics_summary.csv:           append per epoch, aggregate metrics
- metrics_per_class.csv:         append per epoch, one row per (epoch, class)
- metrics_per_class_final.json:  on finalize(), full per-class curves

Append-friendly: if files exist (resume scenario), new rows are appended
without rewriting the header. finalize() reads the CSV (not in-memory
state) so resume preserves all prior epochs.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import torch


class MetricsLogger:
    """Append-only metrics writer for training runs."""

    def __init__(
        self,
        output_dir: str | Path,
        class_names: Optional[list[str]] = None,
        n_blocks: int = 3,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names
        self.n_blocks = n_blocks
        self._summary_cols = self._build_summary_cols()

    def _build_summary_cols(self) -> list[str]:
        cols = [
            "epoch", "lr",
            "train_loss", "train_map",
            "val_loss", "val_map", "sample_val_map",
        ]
        for i in range(self.n_blocks):
            cols += [
                f"block_{i+1}_train_map",
                f"block_{i+1}_val_map",
                f"block_{i+1}_sample_val_map",
            ]
        cols += ["diversity_loss", "epoch_time_s"]
        return cols

    @property
    def summary_csv(self) -> Path:
        return self.output_dir / "metrics_summary.csv"

    @property
    def per_class_csv(self) -> Path:
        return self.output_dir / "metrics_per_class.csv"

    @property
    def per_class_json(self) -> Path:
        return self.output_dir / "metrics_per_class_final.json"

    @staticmethod
    def _append_row(path: Path, header: list[str], row: dict[str, Any]) -> None:
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in header})

    def log_epoch_summary(self, **metrics: Any) -> None:
        """Append a row to metrics_summary.csv. Unknown keys are ignored."""
        row = {k: metrics.get(k, "") for k in self._summary_cols}
        self._append_row(self.summary_csv, self._summary_cols, row)

    def log_epoch_per_class(
        self,
        epoch: int,
        ap_full: torch.Tensor,
        ap_sampled: Optional[torch.Tensor] = None,
        block_ap_full: Optional[list[torch.Tensor]] = None,
        block_ap_sampled: Optional[list[torch.Tensor]] = None,
    ) -> None:
        """Append per-class AP for the epoch.

        Shapes (all optional except ap_full):
            ap_full:           (n_classes,)
            ap_sampled:        (n_classes,)
            block_ap_full:     list of (n_classes,), length n_blocks
            block_ap_sampled:  list of (n_classes,), length n_blocks
        """
        n_classes = ap_full.numel()

        header = ["epoch", "class_id", "class_name", "ap_full"]
        if ap_sampled is not None:
            header.append("ap_sampled")
        if block_ap_full is not None:
            for i in range(len(block_ap_full)):
                header.append(f"ap_full_block_{i+1}")
        if block_ap_sampled is not None:
            for i in range(len(block_ap_sampled)):
                header.append(f"ap_sampled_block_{i+1}")

        ap_full_l = ap_full.detach().cpu().tolist()
        ap_sampled_l = ap_sampled.detach().cpu().tolist() if ap_sampled is not None else None
        block_full_l = (
            [t.detach().cpu().tolist() for t in block_ap_full]
            if block_ap_full is not None else None
        )
        block_sampled_l = (
            [t.detach().cpu().tolist() for t in block_ap_sampled]
            if block_ap_sampled is not None else None
        )

        write_header = not self.per_class_csv.exists()
        with open(self.per_class_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if write_header:
                writer.writeheader()
            for c in range(n_classes):
                row: dict[str, Any] = {
                    "epoch": epoch,
                    "class_id": c,
                    "class_name": (
                        self.class_names[c]
                        if self.class_names and c < len(self.class_names) else ""
                    ),
                    "ap_full": ap_full_l[c],
                }
                if ap_sampled_l is not None:
                    row["ap_sampled"] = ap_sampled_l[c]
                if block_full_l is not None:
                    for i, b in enumerate(block_full_l):
                        row[f"ap_full_block_{i+1}"] = b[c]
                if block_sampled_l is not None:
                    for i, b in enumerate(block_sampled_l):
                        row[f"ap_sampled_block_{i+1}"] = b[c]
                writer.writerow(row)

    def finalize(self) -> None:
        """Read per_class_csv, write per_class_json with summary curves.

        Resume-safe: reconstructs the full history from the CSV, including
        epochs from prior runs that wrote into the same file.
        """
        if not self.per_class_csv.exists():
            logging.info("[metrics] no per_class_csv to finalize, skipping")
            return

        # (class_id, class_name) -> metric -> list of (epoch, value)
        history: dict[tuple[int, str], dict[str, list[tuple[int, float]]]] = \
            defaultdict(lambda: defaultdict(list))

        with open(self.per_class_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    cid = int(row["class_id"])
                    ep = int(row["epoch"])
                except (ValueError, KeyError):
                    continue
                cname = row.get("class_name", "") or ""
                for k, v in row.items():
                    if k in ("epoch", "class_id", "class_name"):
                        continue
                    if v == "" or v is None:
                        continue
                    try:
                        history[(cid, cname)][k].append((ep, float(v)))
                    except ValueError:
                        continue

        out: dict[str, Any] = {"n_classes": len(history), "classes": {}}
        for (cid, cname), curves in sorted(history.items()):
            entry: dict[str, Any] = {"class_id": cid, "class_name": cname}
            for metric, points in curves.items():
                points_sorted = sorted(points, key=lambda p: p[0])
                best = max(points_sorted, key=lambda p: p[1])
                entry[metric] = {
                    "epochs": [p[0] for p in points_sorted],
                    "values": [p[1] for p in points_sorted],
                    "best_value": best[1],
                    "best_epoch": best[0],
                }
            out["classes"][str(cid)] = entry

        self.per_class_json.write_text(json.dumps(out, indent=2))
        logging.info(f"[metrics] wrote {self.per_class_json} "
                     f"({out['n_classes']} classes)")