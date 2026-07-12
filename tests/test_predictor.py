"""Tests for the LitePhospho predictor. These load the real ONNX model (no API key)."""
from __future__ import annotations

from litephospho.predictor import WINDOW, _clean, _windows_for, predict_sequence

# p53 N-terminal fragment (contains several S/T/Y sites).
P53 = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP"


def test_clean_strips_non_amino_acids():
    assert _clean("ac-de 12 fg\nHI") == "ACDEFGHI"


def test_windows_are_centered_on_sty():
    seq = "AAASAAA"
    windows = _windows_for(seq)
    assert len(windows) == 1  # one serine
    pos0, res, w = windows[0]
    assert res == "S"
    assert len(w) == WINDOW
    assert w[15] == "S"  # site sits at the window centre


def test_predict_returns_probabilities_per_site():
    preds = predict_sequence(P53, threshold=0.5, explain=False)
    assert preds, "expected S/T/Y sites"
    assert all(p.residue in ("S", "T", "Y") for p in preds)
    assert all(0.0 <= p.probability <= 1.0 for p in preds)
    # positions are 1-indexed and within the sequence
    assert all(1 <= p.position <= len(P53) for p in preds)


def test_explanation_shape_and_center_marked():
    preds = predict_sequence("AAAAAAAAAAAAAAASAAAAAAAAAAAAAAA", explain=True)
    assert preds
    imp = preds[0].importance
    assert len(imp) == WINDOW


def test_no_sites_returns_empty():
    assert predict_sequence("AAAGGGLLL", explain=True) == []
