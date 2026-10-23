#!/bin/bash
# Run LLM-SR (patched with patches/llmsr_env_vars.patch) against a local vLLM server.
#   1) vllm serve <path-to-Meta-Llama-3.1-8B-Instruct> --served-model-name llama31_8b --port 8011
#   2) VLLM_MODEL=llama31_8b python adapter.py 5000          # LLM-SR /completions -> vLLM chat API
#   3) bash scripts/run_llmsr.sh oscillator2 1 3000           # problem, run id, sample budget
# For CRK tasks copy crk_llmsr/data/* into LLM-SR/data/ and crk_llmsr/specs/* into LLM-SR/specs/ first.
set -e
p=$1; r=$2; budget=$3
cd "$(dirname "$0")/../LLM-SR"
LLMSR_MAX=$budget LLMSR_URL=http://127.0.0.1:5000/completions python main.py --problem_name $p \
  --spec_path ./specs/specification_${p}_numpy.txt --log_path ./logs/llama8b_${p}_r$r --run_id $r
python ../llmsr_eval.py ./logs/llama8b_${p}_r$r $p
