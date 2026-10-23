# Copied baselines and uninformative success criteria in LLM-guided equation discovery

Code, configurations and raw results for the paper

> **Copied baselines and uninformative success criteria: re-examining the gains of LLM-guided scientific equation discovery**
> Jian Xu (RIKEN iTHEMS). Under review.

The paper audits published LLM-based symbolic regression results on the LLM-SR benchmarks (LLM-SR, DecAEvolve, CoEvo) and on the DoLQ ODE benchmark. In brief:

* The PySR baseline on the four LLM-SR datasets was reported once and copied verbatim into later papers. Rerunning PySR under the configuration stated in the original paper improves all eight reported numbers when aggregated the same way (best of five runs).
* With both sides rerun under one protocol (Llama-3.1-8B for LLM-SR), LLM guidance and PySR are within about an order of magnitude of each other on the two oscillators, while LLM guidance gives genuine out-of-distribution gains on E. coli growth and on most LLM-SRBench chemistry tasks.
* DoLQ's trajectory-error success criterion is passed by dense sparse regression and by closed-book LLM guesses with wrong structure; a held-out initial condition exposes them.
* Run-to-run variation spans several orders of magnitude, so best-of-k numbers cannot be compared with medians.

## Layout

| File | Purpose |
|---|---|
| `srbench_pysr.py` | PySR on the four LLM-SR datasets (default and "stated" configurations, iteration or wall-clock budgets) |
| `srbench_crk.py` | PySR on the LLM-SRBench chemistry (CRK) tasks |
| `dolq_baselines.py` | PySR and dense SINDy on the DoLQ benchmark, DoLQ's NMSE metrics |
| `dolq_closedbook.py` | Closed-book LLM test on DoLQ (descriptions only, no data, no search) |
| `dolq_heldout.py` | Re-scores every DoLQ candidate from the held-out initial condition |
| `adapter.py` | Translates LLM-SR's `/completions` requests to a vLLM OpenAI-compatible server |
| `patches/llmsr_env_vars.patch` | Two-line patch to LLM-SR: server URL and sample cap from environment variables |
| `llmsr_anytime.py`, `llmsr_eval.py` | Evaluation of LLM-SR runs (final and anytime), with a refit that must reproduce the training loss recorded during search |
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
* **Compute.** PySR runs used a 14-core laptop CPU. LLM-SR runs used vLLM 0.9.2 on an NVIDIA A6000 and vLLM 0.32 on an NVIDIA GB200 (Triton attention backend), with many runs sharing one GPU; wall-clock times of LLM-SR runs therefore include queueing.

## License

MIT for the code in this repository. LLM-SR, DoLQ and LLM-SRBench are subject to their own licenses.
