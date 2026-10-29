"""LLM-SR at a fixed wall-clock budget: the best-scoring program among the samples written within the
budget, evaluated exactly as in llmsr_anytime.py. Two clocks are used for every run:
  shared     the run's own sample timestamps (one GPU served many concurrent runs);
  dedicated  the median number of samples that the dedicated-GPU timing runs of the same task wrote
             within the budget (one GPU per run; dedicated_timing.json from llmsr_timing.py).
Sampling does not depend on the hardware, so the dedicated clock is applied to the existing runs."""
import glob, json, os, sys
import numpy as np
from llmsr_anytime import HERE, evaluate

BUDGET_S = 3600


def samples(run_dir):
    S = []
    for f in glob.glob(f"{run_dir}/samples/samples_*.json"):
        d = json.load(open(f))
        sc = d.get("score"); sc = None if sc in (None, "None") else float(sc)
        S.append((int(d["sample_order"]), sc, d.get("function"), os.path.getmtime(f)))
    S.sort()
    return S


def best_at(S, ds, n, cache):
    cand = sorted([s for s in S[:n] if s[1] is not None and s[2]], key=lambda s: -s[1])
    for s in cand[:5]:
        if s[0] not in cache:
            try:
                cache[s[0]] = evaluate(s[2], ds, s[1])
            except Exception:
                continue
        r = cache[s[0]]
        return dict(n=n, ID=r[0], OOD=r[1], matched=bool(r[4]))
    return None


if __name__ == "__main__":
    outfile = sys.argv[1]
    T = [t for t in json.load(open(f"{HERE}/results_llmsr_rerun/dedicated_timing.json")) if t["dedicated"]]
    n_ded = {ds: int(np.median([t["n_at"][str(BUDGET_S)] for t in T if t["ds"] == ds])) for ds in {t["ds"] for t in T}}
    runs = [r["run"] for r in json.load(open(f"{HERE}/results_llmsr_rerun/anytime_tlt.json"))]
    runs += [r["run"] for r in json.load(open(f"{HERE}/results_llmsr_rerun/anytime_rikyu.json"))]
    out = []
    for run in runs:
        ds = os.path.basename(run)[len("llama8b_"):].rsplit("_r", 1)[0]
        S = samples(f"{HERE}/{run}"); t0 = S[0][3]; cache = {}
        n_sh = sum(s[3] - t0 <= BUDGET_S for s in S)
        out.append(dict(run=run, ds=ds, budget_s=BUDGET_S, shared=best_at(S, ds, n_sh, cache),
                        dedicated=best_at(S, ds, min(n_ded[ds], len(S)), cache) if ds in n_ded else None))
        print(run, out[-1]["shared"], out[-1]["dedicated"], flush=True)
    json.dump(out, open(outfile, "w"), indent=1)
