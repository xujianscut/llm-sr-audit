#!/bin/bash
# Fetch the public code/data this audit depends on, pinned to the commits we used.
set -e
cd "$(dirname "$0")/.."
git clone https://github.com/deep-symbolic-mathematics/LLM-SR.git && (cd LLM-SR && git checkout 41c212312df6c16d936c9cb395356a62774c47e3)
git clone https://github.com/Bon99yun/DoLQ.git && (cd DoLQ && git checkout 8fa2a9a65b4a9fae9ac072b61726b0594f1c9de3)
# LLM-SRBench is gated on Hugging Face: request access at https://huggingface.co/datasets/nnheui/llm-srbench,
# then run with HF_TOKEN set:
mkdir -p srbench
for f in lsr_bench_data.hdf5 data/lsr_synth_chem_react-00000-of-00001.parquet; do
  curl -L -H "Authorization: Bearer ${HF_TOKEN:?set HF_TOKEN}" -o "srbench/$(basename $f)" \
    "https://huggingface.co/datasets/nnheui/llm-srbench/resolve/main/$f"
done
python build_crk_llmsr.py
