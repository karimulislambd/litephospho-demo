# LitePhospho — Explainable Phosphosite Predictor (live demo)

[![CI](https://github.com/karimulislambd/litephospho-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/karimulislambd/litephospho-demo/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](#)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21290440-blue)](https://doi.org/10.5281/zenodo.21290440)

> Paste a protein sequence and get **phosphorylation-site predictions** for every S/T/Y
> residue — each one **explained** by which neighbouring residues drove the prediction.

This is the **interactive demo** for my first-author thesis model, **LitePhospho**. The
research code, data, and paper live in a separate, citable repository (see the DOI badge);
this app makes the trained model something anyone can *use in the browser*.

**Live demo:** _add your Streamlit URL here_

---

## What it does

1. **Predict** — for each S/T/Y in the sequence, the trained LitePhospho model outputs a
   phosphorylation probability.
2. **Explain** — an **occlusion (deletion) test** masks each residue in the ±15 window and
   measures the drop in probability, showing which residues supported the prediction —
   the same faithfulness method used in the paper.

## Why it's built this way

| Choice | Reason |
|---|---|
| **ONNX Runtime** (not PyTorch) | The model is exported to ONNX (1.67 MB) and served on CPU — lightweight, fast, cheap to host, and matches the paper's "green / deployable" claim. |
| **Preprocessing mirrors training** | Same `WINDOW=31`, X-padding, and amino-acid vocabulary, so served predictions match the paper's scores exactly (verified: correlation 1.0000 vs saved test scores). |
| **Occlusion, not gradients** | Explanations need no autodiff — pure ONNX inference — and reproduce the paper's deletion-test methodology. |

## The model

- Lightweight 1D-CNN, **~0.44M params**, **1.67 MB**, CPU-only (~1 ms/site)
- Trained on human S/T/Y sites (EPSD) with a **homology-controlled, leakage-free split**
- Held-out **ROC-AUC ≈ 0.80**, **PR-AUC ≈ 0.80**, well-calibrated (ECE ≈ 0.014)

> Research artifact & citation: **DOI 10.5281/zenodo.21290440**

## Quickstart

```bash
git clone https://github.com/karimulislambd/litephospho-demo.git
cd litephospho-demo

python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py     # http://localhost:8501
```

No API keys, no GPU — it runs entirely on CPU from the bundled ONNX model.

## Run the tests

```bash
pytest -q
ruff check .
```

## How it works

```
 protein sequence
        │  find every S/T/Y, take a 31-residue window (±15, X-padded)
        ▼
 encode to integer indices ──► ONNX model ──► logit ──► phospho probability
                                    │
                                    └─► occlusion: mask each residue, measure the drop
                                        = per-position importance (the explanation)
```

## Roadmap

- [ ] Fetch a sequence directly by UniProt accession
- [ ] Motif logo for high-confidence predicted sites
- [ ] Downloadable results (CSV)

---

Built by **Md Karimul Islam** — AI/ML Engineer · Computer Vision · LLM · XAI. Model from my
first-author undergraduate thesis (with co-authors & supervisor).
