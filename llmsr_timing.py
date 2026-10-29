"""Timing of LLM-SR runs: wall-clock time at which sample n was written (file mtime, relative to
the first sample) and LLM-SR's own cumulative sample-generation and evaluation times (tensorboard).
Used for both the shared-GPU runs (llama8b_*) and the dedicated-GPU timing runs (ded_*)."""
import glob, json, os, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BUDGETS = [25, 50, 100, 200, 400, 800, 1000, 1600, 2400, 3000]
BUDGETS_S = [600, 1800, 3600, 7200, 10800]   # wall-clock budgets (s): number of samples written within each


def tb_total(run_dir, which):
    sub = f"{run_dir}/Total Sample_Evaluate Time_{which} time"
    ea = EventAccumulator(sub); ea.Reload()
    tag = ea.Tags()["scalars"][0]
    return float(ea.Scalars(tag)[-1].value)


def timing(run_dir):
    S = sorted((int(json.load(open(f))["sample_order"]), os.path.getmtime(f)) for f in glob.glob(f"{run_dir}/samples/samples_*.json"))
    t0 = S[0][1]
    name = os.path.basename(run_dir)
    ds = name.split("_", 1)[1].rsplit("_", 1)[0]
    return dict(run=name, ds=ds, dedicated=name.startswith("ded_"), n=len(S),
                wall=[dict(n=b, wall_s=S[b - 1][1] - t0) for b in BUDGETS if b <= len(S)] + [dict(n=len(S), wall_s=S[-1][1] - t0)],
                n_at={str(b): sum(t - t0 <= b for _, t in S) for b in BUDGETS_S},
                sample_s=tb_total(run_dir, "sample"), evaluate_s=tb_total(run_dir, "evaluate"))


if __name__ == "__main__":
    root, outfile = sys.argv[1], sys.argv[2]
    runs = [d for d in sorted(glob.glob(f"{root}/ded_*") + glob.glob(f"{root}/llama8b_*")) if os.path.isdir(f"{d}/samples")]
    res = []
    for d in runs:
        target = 1000 if "CRK" in d else 3000
        if len(glob.glob(f"{d}/samples/*.json")) < target:
            print("skip incomplete", d); continue
        res.append(timing(d))
    json.dump(res, open(outfile, "w"), indent=1)
    print("runs:", len(res))
