"""LitePhospho — Explainable Phosphorylation-Site Predictor (Streamlit UI).

Paste a protein sequence; the app predicts phosphorylation at every S/T/Y site
using the trained LitePhospho model (served via ONNX Runtime) and explains each
prediction with an occlusion importance profile. Run:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from litephospho.predictor import SITES, explain, predict_sequence

st.set_page_config(
    page_title="LitePhospho — Phosphosite Predictor",
    page_icon=":material/biotech:",
    layout="wide",
)

# A real human protein fragment (p53 N-terminal region) as a ready example.
EXAMPLE = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP"


def _prob_color(p: float, threshold: float) -> str:
    if p >= threshold:
        span = max(1e-6, 1.0 - threshold)
        return f"rgba(34,197,94,{0.25 + 0.6 * min(1.0, (p - threshold) / span):.2f})"
    return "transparent"


def _render_sequence(seq: str, preds, threshold: float) -> str:
    prob_at = {p.position: p.probability for p in preds}
    out = []
    for i, aa in enumerate(seq, start=1):
        if aa in SITES and i in prob_at:
            p = prob_at[i]
            weight = "700" if p >= threshold else "400"
            out.append(
                f"<span title='pos {i}: {p:.2f}' style='background:{_prob_color(p, threshold)};"
                f"font-weight:{weight};border-radius:3px;padding:0 1px'>{aa}</span>"
            )
        else:
            out.append(f"<span style='color:#94a3b8'>{aa}</span>")
    return (
        "<div style='font-family:monospace;font-size:15px;line-height:2;"
        "word-break:break-all'>" + "".join(out) + "</div>"
    )


def _render_importance(window: str, imp: list[float]) -> str:
    """Render the 31-residue window coloured by occlusion importance."""
    if not imp:
        return ""
    hi = max(max(imp), 1e-6)
    cells = []
    for j, (aa, v) in enumerate(zip(window, imp, strict=False)):
        offset = j - 15
        alpha = max(0.0, v / hi) * 0.85
        bg = f"rgba(34,197,94,{alpha:.2f})" if v > 0 else "transparent"
        border = "2px solid #8b7cff" if offset == 0 else "1px solid #33334a"
        cells.append(
            f"<div style='display:inline-block;text-align:center;width:22px;"
            f"margin:1px;border-radius:4px;border:{border};background:{bg};"
            f"font-family:monospace;padding:3px 0' title='offset {offset:+d}, "
            f"importance {v:+.3f}'>{aa}</div>"
        )
    return "<div style='line-height:2.2'>" + "".join(cells) + "</div>"


def _sidebar():
    with st.sidebar:
        st.subheader(":material/science: About the model")
        st.markdown(
            "**LitePhospho** is a lightweight, explainable 1D-CNN that predicts "
            "S/T/Y phosphorylation sites in human proteins."
        )
        st.markdown(
            "- **~0.44M parameters** · **1.67 MB** · ONNX, CPU-only (~1 ms/site)\n"
            "- Evaluated on a **homology-controlled** (leakage-free) split\n"
            "- Held-out **ROC-AUC ≈ 0.80**, **PR-AUC ≈ 0.80**\n"
            "- Explanations via a faithful **occlusion (deletion) test**"
        )
        st.divider()
        st.caption("From the LitePhospho thesis (first author).")
        st.markdown("[Research repo & DOI](https://doi.org/10.5281/zenodo.21290440)")
        return st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.05)


def main():
    threshold = _sidebar()
    st.title("LitePhospho — Explainable Phosphosite Predictor")
    st.caption(
        "Paste a protein sequence to predict phosphorylation at each S/T/Y site — "
        "with a per-site explanation of which residues drove the prediction."
    )

    # Persist results across reruns so changing the threshold/selectbox does not reset.
    st.session_state.setdefault("seq", "")
    st.session_state.setdefault("preds", None)
    st.session_state.setdefault("pred_seq", "")

    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Load example (p53)", use_container_width=True):
            st.session_state.seq = EXAMPLE
    with col1:
        seq = st.text_area(
            "Protein sequence (one-letter amino acids)",
            value=st.session_state.seq,
            height=120,
            placeholder="e.g. MEEPQSDPSVEPPLSQETFSDLWKLL…",
        )

    if st.button("Predict phosphosites", type="primary"):
        if not seq.strip():
            st.warning("Paste a protein sequence first, or load the example.")
        else:
            with st.spinner("Running LitePhospho…"):
                st.session_state.preds = predict_sequence(seq, threshold=threshold)
                st.session_state.pred_seq = "".join(c for c in seq.upper() if c.isalpha())

    # --- Render results from session state (survives selectbox / threshold reruns) ---
    preds = st.session_state.preds
    if preds is None:
        return
    if not preds:
        st.info("No S/T/Y residues found in this sequence — nothing to score.")
        return

    pred_seq = st.session_state.pred_seq
    n_sites = sum(1 for p in preds if p.probability >= threshold)
    m1, m2, m3 = st.columns(3)
    m1.metric("Sequence length", len(pred_seq))
    m2.metric("S/T/Y candidate sites", len(preds))
    m3.metric(f"Predicted phosphosites (≥{threshold:g})", n_sites)

    st.markdown("#### Sequence (predicted sites highlighted)")
    st.markdown(_render_sequence(pred_seq, preds, threshold), unsafe_allow_html=True)

    st.markdown("#### Per-site predictions")
    df = pd.DataFrame(
        [
            {
                "Position": p.position,
                "Residue": p.residue,
                "Probability": round(p.probability, 3),
                "Prediction": "Phosphosite" if p.probability >= threshold else "—",
            }
            for p in preds
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    # --- Explanation ---
    st.markdown("#### Why? — occlusion explanation")
    st.caption(
        "For the selected site, each residue in the ±15 window is masked in turn; "
        "greener = removing it lowered the phosphorylation probability (it supported "
        "the prediction). The purple-bordered residue is the site itself."
    )
    labels = {f"pos {p.position} ({p.residue}) — {p.probability:.2f}": p for p in preds}
    choice = st.selectbox("Choose a site to explain", list(labels.keys()))
    chosen = labels[choice]
    with st.spinner("Explaining…"):
        imp = explain(chosen)
    st.markdown(_render_importance(chosen.window, imp), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
