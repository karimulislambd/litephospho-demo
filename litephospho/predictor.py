"""Phosphorylation-site predictor: serves the trained LitePhospho model (ONNX).

The model takes a 31-residue window centred on a candidate S/T/Y site and outputs
a logit -> phosphorylation probability. Preprocessing here mirrors the training
pipeline exactly (WINDOW=31, X-padding, the same amino-acid vocabulary), so the
served predictions match the paper's reported scores bit-for-bit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort

# --- must match training exactly ---
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWYX")
VOCAB = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
PAD_IDX = VOCAB["X"]
WINDOW = 31
HALF = WINDOW // 2  # 15
SITES = ("S", "T", "Y")  # phospho-acceptor residues

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "litephospho_v4.onnx"


@dataclass
class SitePrediction:
    position: int          # 1-indexed position in the sequence
    residue: str           # S, T, or Y
    probability: float     # phosphorylation probability
    window: str            # the 31-residue context used
    is_site: bool          # probability >= threshold

    # Occlusion importance per window position (len 31), relative offsets -15..+15.
    importance: list[float] = field(default_factory=list)


@lru_cache(maxsize=1)
def _session() -> ort.InferenceSession:
    # Single-threaded, sequential execution. onnxruntime's default thread pool can
    # segfault under the constrained, multi-tenant CPU of free hosting; one thread
    # is plenty for this ~0.44M-param model and is far more stable.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"]
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _clean(seq: str) -> str:
    """Uppercase and strip anything that isn't a standard amino acid."""
    return "".join(c for c in seq.upper() if c in VOCAB and c != "X")


def _encode(windows: list[str]) -> np.ndarray:
    return np.array(
        [[VOCAB.get(a, PAD_IDX) for a in w] for w in windows], dtype=np.int64
    )


def _windows_for(seq: str) -> list[tuple[int, str, str]]:
    """Return (0-indexed pos, residue, 31-window) for every S/T/Y in the sequence."""
    padded = "X" * HALF + seq + "X" * HALF
    out = []
    for i, aa in enumerate(seq):
        if aa in SITES:
            out.append((i, aa, padded[i : i + WINDOW]))
    return out


def _predict_windows(windows: list[str]) -> np.ndarray:
    if not windows:
        return np.array([], dtype=np.float32)
    sess = _session()
    name = sess.get_inputs()[0].name
    logits = sess.run(None, {name: _encode(windows)})[0].ravel()
    return _sigmoid(logits)


def _occlusion(window: str, base_prob: float) -> list[float]:
    """Faithfulness-style importance: mask each position with X, measure the drop.

    Mirrors the paper's deletion test — a positive value means removing that
    residue lowers the phosphorylation probability (it supported the prediction).
    """
    variants = []
    for j in range(WINDOW):
        w = window[:j] + "X" + window[j + 1 :]
        variants.append(w)
    masked = _predict_windows(variants)
    return [float(base_prob - m) for m in masked]


def predict_sequence(
    seq: str, threshold: float = 0.5, explain: bool = False
) -> list[SitePrediction]:
    """Predict phosphorylation for every S/T/Y site in a protein sequence.

    `explain=False` by default: the occlusion explanation is expensive (31 masked
    inferences per site), so the UI computes it lazily for the selected site only
    via `explain()`. Pass `explain=True` to compute it for every site up front.
    """
    seq = _clean(seq)
    sites = _windows_for(seq)
    if not sites:
        return []

    probs = _predict_windows([w for _, _, w in sites])
    results: list[SitePrediction] = []
    for (pos0, res, window), p in zip(sites, probs, strict=False):
        pred = SitePrediction(
            position=pos0 + 1,
            residue=res,
            probability=float(p),
            window=window,
            is_site=bool(p >= threshold),
        )
        if explain:
            pred.importance = _occlusion(window, float(p))
        results.append(pred)
    return results


@lru_cache(maxsize=512)
def _explain_cached(window: str, base_prob: float) -> tuple[float, ...]:
    return tuple(_occlusion(window, base_prob))


def explain(pred: SitePrediction) -> list[float]:
    """Occlusion importance for a single predicted site — computed lazily + cached."""
    return list(_explain_cached(pred.window, round(pred.probability, 6)))
