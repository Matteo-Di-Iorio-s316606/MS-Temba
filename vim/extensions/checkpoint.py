"""Checkpoint, resume, early stopping, and OAR signal handling.

Self-contained extension; does not modify upstream modules. Plug into
MSTemba_main.py at three call sites: construction, end-of-epoch, exit.

Design:
- CheckpointState is a plain dataclass (no torch logic) — easy to extend.
- CheckpointManager does atomic save via tempfile + os.replace, so a SIGKILL
  mid-write leaves the previous checkpoint intact.
- EarlyStopper tracks patience on val mAP; serializable for resume.
- OARSignalHandler installs SIGUSR2 handler; only flips a flag (signal-safe).
"""

from __future__ import annotations

import logging
import os
import signal
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch

# -----------------------------------------------------------------------------
# Checkpoint state
# -----------------------------------------------------------------------------

@dataclass
class CheckpointState:
    """Everything that must persist between runs."""
    epoch: int
    best_val_map: float
    best_block_val_maps: list[float]
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: Optional[dict[str, Any]] = None
    ema_state: Optional[dict[str, Any]] = None
    early_stopper_state: Optional[dict[str, Any]] = None
    rng_state: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Atomic checkpoint manager
# -----------------------------------------------------------------------------

class CheckpointManager:
    """Save/load training state atomically.

    Path layout under `output_dir`:
        checkpoint_last.pth        # overwritten every epoch
        checkpoint_best.pth        # overwritten on val_map improvement
        block_<i>/checkpoint_best.pth  # per-block best (paper Fig. 5)
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def last_path(self) -> Path:
        return self.output_dir / "checkpoint_last.pth"

    @property
    def best_path(self) -> Path:
        return self.output_dir / "checkpoint_best.pth"

    def block_best_path(self, block_idx: int) -> Path:
        d = self.output_dir / f"block_{block_idx + 1}"
        d.mkdir(parents=True, exist_ok=True)
        return d / "checkpoint_best.pth"

    @staticmethod
    def _atomic_save(payload: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
        os.close(fd)
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)  # POSIX-atomic
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def save(self, state: CheckpointState, kind: str = "last") -> Path:
        """kind: 'last' | 'best' | 'block_<i>' (i in [0, 1, 2])."""
        payload = state.__dict__.copy()
        if kind == "last":
            path = self.last_path
        elif kind == "best":
            path = self.best_path
        elif kind.startswith("block_"):
            idx = int(kind.split("_", 1)[1])
            path = self.block_best_path(idx)
        else:
            raise ValueError(f"Unknown checkpoint kind: {kind}")
        self._atomic_save(payload, path)
        logging.info(f"[ckpt] saved {kind} -> {path}")
        return path

    def load(
        self,
        path: str | Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Any = None,
        ema: Any = None,
        early_stopper: Any = None,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> CheckpointState:
        """Load checkpoint into provided modules. Returns the state."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        payload = torch.load(path, map_location=map_location)

        model.load_state_dict(payload["model_state"], strict=strict)
        if optimizer is not None and payload.get("optimizer_state") is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None and payload.get("scheduler_state") is not None:
            scheduler.load_state_dict(payload["scheduler_state"])
        if ema is not None and payload.get("ema_state") is not None:
            # timm ModelEma stores the EMA shadow in `ema.ema`.
            ema.ema.load_state_dict(payload["ema_state"])
        if early_stopper is not None and payload.get("early_stopper_state") is not None:
            early_stopper.load_state_dict(payload["early_stopper_state"])

        rng = payload.get("rng_state", {})
        if "cpu" in rng:
            cpu_state = rng["cpu"]
            # Ensure ByteTensor on CPU regardless of map_location used above.
            if not isinstance(cpu_state, torch.ByteTensor):
                cpu_state = cpu_state.to(dtype=torch.uint8, device="cpu")
            torch.set_rng_state(cpu_state)
        if "cuda" in rng and torch.cuda.is_available():
            cuda_states = rng["cuda"]
            cuda_states = [
                s if isinstance(s, torch.ByteTensor) else s.to(dtype=torch.uint8, device="cpu")
                for s in cuda_states
            ]
            torch.cuda.set_rng_state_all(cuda_states)

        state = CheckpointState(**payload)
        logging.info(
            f"[ckpt] loaded {path} (epoch={state.epoch}, "
            f"best_val_map={state.best_val_map:.4f})"
        )
        return state


# -----------------------------------------------------------------------------
# Early stopping
# -----------------------------------------------------------------------------

class EarlyStopper:
    """Stop training if val metric does not improve for `patience` epochs.

    Improvement: current > best + min_delta.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.patience = patience
        self.min_delta = min_delta
        self.best: float = float("-inf")
        self.bad_epochs: int = 0

    def update(self, current: float) -> bool:
        """Returns True if this epoch was an improvement."""
        if current > self.best + self.min_delta:
            self.best = current
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    def state_dict(self) -> dict[str, Any]:
        return {"best": self.best, "bad_epochs": self.bad_epochs,
                "patience": self.patience, "min_delta": self.min_delta}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best = state["best"]
        self.bad_epochs = state["bad_epochs"]
        # patience and min_delta are config, not loaded from old runs


# -----------------------------------------------------------------------------
# OAR SIGUSR2 handler
# -----------------------------------------------------------------------------

class OARSignalHandler:
    """Install SIGUSR2 handler. OAR sends SIGUSR2 `--checkpoint N` seconds
    before walltime expiry.

    Sets a flag only. Loop is responsible for polling and saving.
    Never call torch.save from a signal handler (not async-signal-safe).
    """

    def __init__(self, on_signal: Optional[Callable[[], None]] = None) -> None:
        self._should_exit = False
        self._on_signal = on_signal
        signal.signal(signal.SIGUSR2, self._handler)

    def _handler(self, signum: int, frame: Any) -> None:
        self._should_exit = True
        logging.info("[oar] SIGUSR2 received, will save+exit at end of epoch")
        if self._on_signal is not None:
            try:
                self._on_signal()
            except Exception as e:
                logging.error(f"[oar] on_signal callback raised: {e}")

    @property
    def should_exit(self) -> bool:
        return self._should_exit


# -----------------------------------------------------------------------------
# Helper: snapshot full state into CheckpointState
# -----------------------------------------------------------------------------

def build_state(
    epoch: int,
    best_val_map: float,
    best_block_val_maps: list[float],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    ema: Any = None,
    early_stopper: Any = None,
    extra: Optional[dict[str, Any]] = None,
) -> CheckpointState:
    """Snapshot full training state into a CheckpointState."""
    # Force CPU storage to avoid map_location surprises on resume.
    rng_state: dict[str, Any] = {"cpu": torch.get_rng_state().cpu()}
    if torch.cuda.is_available():
        # get_rng_state_all() returns list[ByteTensor], one per device.
        rng_state["cuda"] = [s.cpu() for s in torch.cuda.get_rng_state_all()]

    return CheckpointState(
        epoch=epoch,
        best_val_map=best_val_map,
        best_block_val_maps=list(best_block_val_maps),
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict() if scheduler is not None else None,
        ema_state=(ema.ema.state_dict() if ema is not None else None),
        early_stopper_state=(early_stopper.state_dict() if early_stopper is not None else None),
        rng_state=rng_state,
        extra=extra or {},
    )