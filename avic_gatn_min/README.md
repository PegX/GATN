# AViC@GATN (Minimal)

A **minimal, runnable** prototype of **AViC** (From Attacks to Circuits) built on top of the **GATN** idea (Graph Attention Transformer Network) for multi-label prediction.

This repo is intentionally small and self-contained:

- **Target model (MI target):** a minimal re-implementation of the GATN *topology transformer block* (two multi-head attention modules) that produces an adjacency matrix, followed by 2-layer GCN and a final bilinear scoring.
- **Circuits:** `(attn_id, head_id)` where `attn_id ∈ {0,1}` corresponds to the two attention modules inside the block.
- **Alg1 (Circuit discovery):** targeted head ablation + metric drop.
- **Alg2 (Circuit-aware attack):** PGD on class-node features (node embeddings), with an optional circuit-aware term that maximizes attention distortion on discovered circuits.
- **Alg3 (Trace/Attribution):** stub + exports attention snapshots.

> **Note:** The original upstream GATN repository (a791702141/GATN) is designed for COCO/VOC training and depends on dataset files, pretrained CNN backbone, and pickled topology/embeddings. This minimal prototype keeps the *core mechanism* needed for mechanistic vulnerability analysis, and replaces the dataset/backbone with a deterministic toy task so the pipeline can run out-of-the-box.

## Install

```bash
pip install torch pyyaml
```

## Run

```bash
python -m avic_gatn.scripts.run_end2end --config avic_gatn/configs/toy.yaml
```

It will write a JSON report under `outputs/`.

## Upstream reference

- Upstream code base (GATN): https://github.com/a791702141/GATN  
  In particular, the attention block structure and shapes follow `transformer.py` and the adjacency-generation idea in `models.py`.
