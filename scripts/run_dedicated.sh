#!/bin/bash
# Dedicated-GPU timing: one LLM-SR run per GPU, its own vLLM server and adapter, 4 pinned CPU cores per run.
cd /work1/jianxu/llmsr_audit
M=$(cat model_path.txt)
for g in 1 2 3 4 5 6 7 8 9; do
  CUDA_VISIBLE_DEVICES=$g nohup ~/miniconda3/envs/bedllm/bin/python -m vllm.entrypoints.openai.api_server --model $M \
    --served-model-name llama31_8b --port $((8020+g)) --max-model-len 8192 --gpu-memory-utilization 0.85 > logs/ded_vllm_g$g.log 2>&1 &
  VLLM_URL=http://127.0.0.1:$((8020+g))/v1/chat/completions VLLM_MODEL=llama31_8b nohup ~/miniconda3/envs/qgpy/bin/python adapter.py $((5020+g)) > logs/ded_adapter_g$g.log 2>&1 &
done
for g in 1 2 3 4 5 6 7 8 9; do
  until curl -s http://127.0.0.1:$((8020+g))/v1/models > /dev/null; do sleep 10; done; echo "server g$g up $(date +%H:%M)"
done
source ~/miniconda3/etc/profile.d/conda.sh && conda activate llmsr_audit
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
run_one() { g=$1; p=$2; r=$3; n=$4; L=./logs/ded_${p}_d$r; lo=$((4*(g-1))); hi=$((lo+3))
  mkdir -p $L; date +%s > $L.start
  LLMSR_MAX=$n LLMSR_URL=http://127.0.0.1:$((5020+g))/completions taskset -c $lo-$hi python main.py --problem_name $p \
    --spec_path ./specs/specification_${p}_numpy.txt --log_path $L --run_id $((100+r)) > $L.stdout 2>&1
  date +%s > $L.end; echo "DONE $p d$r g$g $(date +%H:%M)"; }
i=0
for d in oscillator1 oscillator2 bactgrow stressstrain; do for r in 1 2; do i=$((i+1)); run_one $i $d $r 3000 & done; done
( for n in CRK3 CRK4 CRK7 CRK12 CRK15 CRK17 CRK19 CRK22 CRK28 CRK30; do run_one 9 $n 1 1000; done ) &
wait
echo ALL_DONE $(date)
