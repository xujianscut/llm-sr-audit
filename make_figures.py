"""Figures for the audit paper. Everything is read from result files; nothing is typed in.
Palette: reference categorical slots 1-3 (blue, orange, aqua) for series, neutral inks for
published reference values. Light mode only (print)."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_tables import (HERE, DS, DSNAME, CRK, REPORTED_PYSR, REPORTED_LLMSR_L8, REPORTED_DECA_L8,
                         REPORTED_CRK, load_glob, pysr_runs, finite_median, pysr_hour)

OUT = os.environ.get("FIG_DIR", os.path.join(HERE, "outputs", "figs")); os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "font.family": "serif", "lines.linewidth": 1.5})
NAMES = {"oscillator1": "Osc. 1", "oscillator2": "Osc. 2", "bactgrow": "E. coli", "stressstrain": "Stress"}


def clipv(v, lo=1e-11, hi=1e3):
    return np.clip(np.where(np.isfinite(v), v, hi), lo, hi)


def fig_partition():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=False)
    for k, (ax, split) in enumerate(zip(axes, ["ID", "OOD"])):
        i = 0 if split == "ID" else 1
        for j, ds in enumerate(DS):
            p = clipv(np.array([r[split] for r in pysr_runs(ds, "default")]))
            l = clipv(np.array([r[split] for r in load_glob(f"{HERE}/results_llmsr_rerun/{ds}_r*.json", 3)]))
            jit = np.linspace(-0.08, 0.08, len(p))
            ax.scatter(j - 0.15 + jit, p, s=16, color=BLUE, edgecolor="white", linewidth=0.6, zorder=3,
                       label="PySR rerun (5 seeds)" if j == 0 else None)
            ax.scatter(j + 0.15 + np.linspace(-0.05, 0.05, len(l)), l, s=18, marker="D", color=ORANGE,
                       edgecolor="white", linewidth=0.6, zorder=3, label=f"LLM-SR rerun ({len(l)} runs)" if j == 0 else None)
            ax.hlines(np.median(p), j - 0.27, j - 0.03, color=BLUE, lw=1.5, zorder=2)
            ax.hlines(np.median(l), j + 0.03, j + 0.27, color=ORANGE, lw=1.5, zorder=2)
            ax.scatter([j - 0.15], [REPORTED_PYSR[ds][i]], marker="x", s=36, color=INK, linewidth=1.4, zorder=4,
                       label="PySR as reported (copied)" if j == 0 else None)
            ax.scatter([j + 0.15], [REPORTED_LLMSR_L8[ds][i]], marker="s", s=22, facecolor="none", edgecolor=INK2, linewidth=1.2, zorder=4,
                       label="LLM-SR as reported" if j == 0 else None)
        ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels([NAMES[d] for d in DS])
        ax.set_title(f"{split} test NMSE", fontsize=8.5, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/partition.pdf", bbox_inches="tight"); plt.close(fig)


def fig_crk():
    """Per CRK task: LLM-SR (median of its runs) at one hour of dedicated-GPU time and at the end of its run,
    against one-hour PySR (median of seeds, bar = range over seeds); DecAEvolve as reported."""
    AT = json.load(open(f"{HERE}/results_llmsr_rerun/llmsr_at_budget.json"))
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    P, Plo, Phi, L1, LE, D = [], [], [], [], [], []
    for n in CRK:
        p = clipv(np.array([r["OOD"] for r in pysr_hour(n)]))
        P.append(np.median(p)); Plo.append(p.min()); Phi.append(p.max())
        L1.append(finite_median([a["dedicated"]["OOD"] for a in AT if a["ds"] == n]))
        LE.append(finite_median([r["OOD"] for r in load_glob(f"{HERE}/results_crk_llmsr/{n}_r*.json", 6)]))
        D.append(REPORTED_CRK[n][1])
    P, Plo, Phi, L1, LE, D = (clipv(np.array(v)) for v in (P, Plo, Phi, L1, LE, D))
    lim = (1e-6, 1e3)
    ax.plot(lim, lim, color=INK2, lw=0.8, ls="--", zorder=1)
    ax.fill_between(lim, lim, [lim[0]] * 2, color="#f3f2ee", zorder=0)
    ax.scatter(P, D, marker="x", s=26, color=INK2, linewidth=1.2, zorder=2, label="DecAEvolve (reported)")
    ax.vlines(P, np.minimum(L1, LE), np.maximum(L1, LE), color=ORANGE, lw=0.8, zorder=2)
    ax.errorbar(P, LE, xerr=[P - Plo, Phi - P], fmt="o", ms=6, mfc="white", mec=ORANGE, mew=1.1, color=ORANGE,
                elinewidth=0.8, capsize=0, zorder=3, label="LLM-SR, end of run")
    ax.scatter(P, L1, s=9, color=ORANGE, zorder=4, label="LLM-SR, 1 h on a dedicated GPU")
    for n, x, y in zip(CRK, P, L1):
        off = (3, -8) if n == "CRK19" else (3, 2)
        ax.annotate(n.replace("CRK", ""), (x, y), xytext=off, textcoords="offset points", fontsize=6, color=INK2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("PySR OOD NMSE (one-hour budget)"); ax.set_ylabel("LLM-guided OOD NMSE")
    ax.text(2e-6, 2e-1, "PySR better", fontsize=7, color=INK2); ax.text(2e-1, 2e-6, "LLM better", fontsize=7, color=INK2, ha="left")
    ax.legend(loc="lower left", fontsize=6.5, bbox_to_anchor=(0.0, 1.01), ncol=2, columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/crk.pdf", bbox_inches="tight"); plt.close(fig)


def fig_traj():
    T = json.load(open(f"{HERE}/results_llmsr_rerun/dolq_traj_P4.json"))
    t = np.array(T["t"])
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 3.6), sharex=True)
    for k, title in enumerate(["(a) training initial condition", "(b) held-out initial condition"]):
        tr = np.array(T["truth"][k])
        for dim in (0, 1):
            ax = axes[dim, k]
            ax.plot(t, tr[:, dim], color=INK, lw=2.2, label="ground truth", zorder=1)
            for name, col, ls, lab in [("sindy", BLUE, (0, (4, 2)), "dense SINDy"), ("closedbook", ORANGE, (0, (1.5, 1.5)), "closed-book guess")]:
                y = T[name][k]
                if y is not None:
                    y = np.array(y); ax.plot(t, y[:, dim], color=col, ls=ls, lw=1.6, label=lab, zorder=2)
            lo, hi = tr[:, dim].min(), tr[:, dim].max(); pad = 0.5 * (hi - lo) + 0.05
            ax.set_ylim(lo - pad, hi + pad); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
            if dim == 0: ax.set_title(title, fontsize=8.5, loc="left", color=INK)
            if k == 0: ax.set_ylabel(f"$x_{dim}(t)$")
            if dim == 1: ax.set_xlabel("time")
    axes[0, 1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/traj_p4.pdf", bbox_inches="tight"); plt.close(fig)





def fig_compute():
    """OOD NMSE versus wall-clock time: PySR at fixed wall-clock budgets plus 1000 iterations
    (median and interquartile range over seeds); LLM-SR anytime curves (median and IQR over runs) placed
    on two clocks, its own timestamps on the shared GPU and the dedicated-GPU timing runs."""
    import collections
    any_runs = json.load(open(f"{HERE}/results_llmsr_rerun/anytime_tlt.json"))
    T = [t for t in json.load(open(f"{HERE}/results_llmsr_rerun/dedicated_timing.json")) if t["dedicated"]]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.3), sharey=False)
    for ax, ds in zip(axes, DS):
        pts = collections.defaultdict(list)
        for f in glob.glob(f"{HERE}/results_llmsr/{ds}_s*_nobatch_t*.json"):
            r = json.load(open(f)); pts[r["tag"]].append((r["wall_s"], r["OOD"]))
        for r in pysr_runs(ds, "default"):
            pts["1000it"].append((r["wall_s"], r["OOD"]))
        xs, med, lo, hi = [], [], [], []
        for k, v in sorted(pts.items(), key=lambda kv: np.median([w for w, _ in kv[1]])):
            w = np.array([a for a, _ in v]); o = clipv(np.array([b for _, b in v]))
            xs.append(np.median(w)); med.append(np.median(o)); lo.append(np.percentile(o, 25)); hi.append(np.percentile(o, 75))
        ax.fill_between(xs, lo, hi, color=BLUE, alpha=0.15, lw=0); ax.plot(xs, med, "-o", color=BLUE, ms=3, label="PySR (14 CPU cores)")
        runs = [r for r in any_runs if r["ds"] == ds and len(r["curve"]) > 3]
        ded = [t for t in T if t["ds"] == ds]; assert len(ded) == 2, (ds, len(ded))
        tded = {w["n"]: np.median([x["wall_s"] for t in ded for x in t["wall"] if x["n"] == w["n"]]) for w in ded[0]["wall"]}
        ns = [n for n in (25, 50, 100, 200, 400, 800, 1000, 1600, 2400, 3000) if n in tded]
        XS, XD, M, L, H = [], [], [], [], []
        for n in ns:
            cs = [c for r in runs for c in r["curve"] if c["n"] == n]
            o = clipv(np.array([c["OOD"] for c in cs]))
            XS.append(np.median([c["wall_s"] for c in cs])); XD.append(tded[n])
            M.append(np.median(o)); L.append(np.percentile(o, 25)); H.append(np.percentile(o, 75))
        ax.fill_between(XD, L, H, color=ORANGE, alpha=0.15, lw=0)
        ax.plot(XD, M, "-D", color=ORANGE, ms=3, label=f"LLM-SR, dedicated GPU ({len(runs)} runs)")
        ax.plot(XS, M, ls=(0, (3, 2)), color=ORANGE, lw=1.1, alpha=0.75, label="LLM-SR, shared GPU")
        ax.axvline(3600, color=INK2, lw=0.6, ls=":", zorder=0)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(NAMES[ds], fontsize=8.5, loc="left", color=INK)
        ax.set_xlabel("wall-clock (s)"); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("OOD NMSE")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/compute.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_partition(); fig_crk(); fig_traj(); fig_compute()
    print(sorted(os.listdir(OUT)))
