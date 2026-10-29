# Copied Baselines and Uninformative Success Criteria in LLM-Guided Equation Discovery

Code, configurations and raw results for the paper

> **Copied Baselines and Uninformative Success Criteria: Re-examining the Gains of LLM-Guided Scientific Equation Discovery**
> Jian Xu (RIKEN iTHEMS). Under review.

The paper audits published LLM-based symbolic regression results on the LLM-SR benchmarks (LLM-SR, DecAEvolve, CoEvo) and on the DoLQ ODE benchmark. In brief:

* The PySR baseline on the four LLM-SR datasets was reported once and copied verbatim into later papers. Rerunning PySR under the configuration stated in the original paper improves all eight reported numbers when aggregated the same way (best of five runs).
* With both sides rerun under one protocol (Llama-3.1-8B, eight LLM-SR runs per dataset), LLM-SR wins at its natural budget on most datasets, but its runs take hours where PySR's take minutes, and whether the win survives a matched budget depends on how compute is counted. In amortised GPU-hours (about 0.3 per run) it does. At one hour of waiting time, PySR on 14 CPU cores against LLM-SR on a dedicated GPU, neither method is reliably ahead on the oscillators or the LLM-SRBench chemistry tasks, and PySR is ahead on stress-strain; E. coli growth favours LLM-SR under every account, by a large margin in the median with a bootstrap interval that still touches zero.
* DoLQ's trajectory-error success criterion is passed by dense sparse regression and by structures an LLM proposes without seeing data (constants then fitted to the data), mostly with wrong structure; a held-out initial condition exposes them. Structure recovery is judged by an explicit automatic rule (`dolq_terms.py`); DoLQ's own advantage in structure recovery is not in question.
* Run-to-run variation spans several orders of magnitude (the median of three LLM-SR runs and of eight differ by six orders on Oscillator 2), so best-of-k numbers cannot be compared with medians.

## Layout

| File | Purpose |
|---|---|
| `srbench_pysr.py` | PySR on the four LLM-SR datasets (default and "stated" configurations, iteration or wall-clock budgets) |
| `srbench_crk.py` | PySR on the LLM-SRBench chemistry (CRK) tasks |
| `dolq_baselines.py` | PySR and dense SINDy on the DoLQ benchmark, DoLQ's NMSE metrics |
| `dolq_closedbook.py` | Closed-book LLM test on DoLQ (descriptions only, no data, no search) |
| `dolq_heldout.py` | Re-scores every DoLQ candidate from the held-out initial condition |
| `dolq_terms.py` | Automatic term test: prunes constants whose removal changes the right-hand side by less than a relative RMS threshold (1e-3, sensitivity 1e-2), snaps constants, compares term skeletons with the ground truth |
| `adapter.py` | Translates LLM-SR's `/completions` requests to a vLLM OpenAI-compatible server |
| `patches/llmsr_env_vars.patch` | Two-line patch to LLM-SR: server URL and sample cap from environment variables |
| `llmsr_anytime.py`, `llmsr_eval.py` | Evaluation of LLM-SR runs (final and anytime), with a refit that must reproduce the training loss recorded during search |
| `llmsr_timing.py` | Wall-clock time per sample and LLM-SR's own generation/evaluation timers, for shared-GPU runs and dedicated-GPU timing runs |
| `llmsr_at_budget.py` | LLM-SR at one hour on the shared and on the dedicated-GPU clock (best-scoring program among the samples written within the hour) |
| `pysr_hour_queue.py` | Queue of one-hour PySR runs (extra seeds for the matched-compute comparison) |
| `scripts/run_dedicated.sh` | Dedicated-GPU timing runs: one LLM-SR run per GPU with its own vLLM server and four pinned CPU cores |
| `build_crk_llmsr.py` | Builds LLM-SR data directories and prompt specifications for the CRK tasks |
| `make_tables.py`, `make_figures.py` | Generate every table, every number quoted in the text (`numbers.tex`) and every figure of the paper from the result files |
| `results*/` | Raw results of every run (JSON) |
| `backup_remote_eval/` | LLM-SR evaluations computed on the original machines with a single BFGS refit (kept for the refit-sensitivity analysis) |

## Reproducing

```bash
pip install -r requirements.txt          # PySR installs Julia on first import
export HF_TOKEN=...                       # LLM-SRBench is gated: request access on Hugging Face first
bash scripts/setup_data.sh                # clones LLM-SR and DoLQ at the commits we used, downloads LLM-SRBench, builds CRK tasks

# tables and figures from the shipped results
TABLE_DIR=outputs/tables python make_tables.py
FIG_DIR=outputs/figs python make_figures.py
```

Re-running individual experiments:

```bash
# PySR, LLM-SR datasets (seed 0, 1000 iterations, no mini-batching; V3CFG=1 for 15 populations of size 33)
NOBATCH=1 python srbench_pysr.py oscillator2 0 1000
TBUDGET=100 NOBATCH=1 python srbench_pysr.py oscillator2 0 1000   # 100-second wall-clock budget
# PySR, CRK
python srbench_crk.py CRK7 0 1000
# DoLQ
python dolq_baselines.py sindy
python dolq_baselines.py pysr 0 1000 30
python dolq_heldout.py
# LLM-SR (see scripts/run_llmsr.sh for the server setup)
cd LLM-SR && git apply ../patches/llmsr_env_vars.patch && cd ..
bash scripts/run_llmsr.sh oscillator2 1 3000
```

## Notes

* **Refitting LLM-SR programs.** LLM-SR scores a program by fitting its parameters with a single BFGS run from a vector of ones. We found that this refit lands in different local optima on different numpy/scipy versions, which can change the out-of-distribution error of the same program by orders of magnitude. Our evaluation therefore requires the refit to reproduce the training loss recorded during search, retrying from seeded random starts when it does not (`llmsr_anytime.evaluate`).
* **Data.** LLM-SR and DoLQ data are fetched from their repositories; LLM-SRBench data are gated and are not redistributed here.
* **Compute.** PySR runs used a 14-core laptop CPU. LLM-SR runs used vLLM 0.9.2 on an NVIDIA A6000 and vLLM 0.32 on an NVIDIA GB200 (Triton attention backend), with 12 to 30 runs sharing one GPU. Separate timing runs (`scripts/run_dedicated.sh`, two per dataset and one per CRK task) gave each run a GPU to itself; sharing slowed runs down by a factor of 1.6 to 2.0, because a single LLM-SR run is latency-bound (77 to 93% of its time waits for the LLM).
* **Raw LLM-SR logs.** `llmsr_anytime.py`, `llmsr_timing.py` and `llmsr_at_budget.py` read LLM-SR's per-sample log directories, which are not included because of their size (about 600 MB); their outputs are in `results_llmsr_rerun/`.

## License

MIT for the code in this repository. LLM-SR, DoLQ and LLM-SRBench are subject to their own licenses.
