"""Generate every results table of the audit paper from raw result files.

All numbers in tables/*.tex come from this script. Published numbers are
transcribed once below (REPORTED_*) with the source table cited, never typed into LaTeX.
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("TABLE_DIR", os.path.join(HERE, "outputs", "tables"))
os.makedirs(OUT, exist_ok=True)
DS = ["oscillator1", "oscillator2", "bactgrow", "stressstrain"]
DSNAME = {"oscillator1": "Oscillator 1", "oscillator2": "Oscillator 2", "bactgrow": "E.~coli growth", "stressstrain": "Stress--strain"}

# ---- published numbers (NMSE ID, OOD) ----------------------------------------------------
# LLM-SR (Shojaee et al., ICLR 2025, arXiv 2404.18400 v1-v3) Table 1, PySR row; identical in
# DecAEvolve Table 1 and CoEvo Table 1.
REPORTED_PYSR = {"oscillator1": (0.0009, 0.3106), "oscillator2": (0.0002, 0.0098),
                 "bactgrow": (0.0376, 1.0141), "stressstrain": (0.0331, 0.1304)}
# DecAEvolve (ICML 2026) Table 1, rows "LLM-SR (Llama-3.1-8B)" and "DecAEvolve (Llama-3.1-8B)".
REPORTED_LLMSR_L8 = {"oscillator1": (1.36e-5, 0.0009), "oscillator2": (4.61e-6, 0.0001),
                     "bactgrow": (0.0117, 0.0240), "stressstrain": (0.0376, 0.0761)}
REPORTED_DECA_L8 = {"oscillator1": (1.37e-6, 0.0002), "oscillator2": (3.64e-7, 2.11e-5),
                    "bactgrow": (0.0019, 0.0045), "stressstrain": (0.0144, 0.0322)}
# DecAEvolve (ICML 2026) Table 3 (LLM-SRBench chemistry, Llama-3.1-8B): (DecAEvolve ID, OOD, LLM-SR ID, OOD).
CRK = ["CRK3", "CRK4", "CRK7", "CRK12", "CRK15", "CRK17", "CRK19", "CRK22", "CRK28", "CRK30"]
REPORTED_CRK = {"CRK3": (1.6e-5, 9.0e-4, 1.1e-4, 9.9e-1), "CRK4": (6.9e-6, 9.2e-4, 3.5e-4, 5.9e-3),
                "CRK7": (6.5e-8, 5.5e-4, 2.0e-4, 1.1e-2), "CRK12": (3.3e-10, 2.2e-6, 3.2e-5, 8.1e-3),
                "CRK15": (1.4e-6, 3.7e-2, 1.7e-5, 1.9e-1), "CRK17": (1.5e-4, 1.8e-3, 2.6e-5, 2.7e-2),
                "CRK19": (2.2e-4, 1.5e-1, 1.1e-4, 9.7e-1), "CRK22": (1.9e-3, 1.7e2, 8.3e-7, 3.3e-2),
                "CRK28": (7.2e-6, 2.8e-2, 1.0e-3, 4.6e0), "CRK30": (5.0e-5, 4.2e-2, 1.1e-4, 1.4e0)}
# DoLQ (ICML 2026) Table 13: problems passing (NMSE test, term test) out of 8.
REPORTED_DOLQ = {"DoLQ": (7, 5), "LLM-SR": (4, 2), "ICSR": (2, 0), "LASR": (2, 2), "EDL": (0, 0)}


def fmt(v):
    if v is None or not np.isfinite(v):
        return "div."
    if v == 0:
        return "0"
    e = int(np.floor(np.log10(abs(v)))); m = v / 10 ** e
    if -2 <= e <= 0:
        return f"{v:.3g}" if e == 0 else f"{v:.2g}"
    return f"{m:.1f}e{e}"


def finite_median(xs):
    xs = np.array([x if np.isfinite(x) else np.inf for x in xs], dtype=float)
    return float(np.median(xs))


def load_glob(pattern, n_expected):
    fs = sorted(glob.glob(pattern))
    assert len(fs) >= n_expected, (pattern, len(fs))
    return [json.load(open(f)) for f in fs]


def write(name, s):
    open(os.path.join(OUT, name), "w").write(s)
    print("wrote", name)


# ---- Table: PySR on the four LLM-SR datasets --------------------------------------------
def pysr_runs(ds, tag):
    if tag == "default":   # PySR defaults, no mini-batching (stress-strain has <5000 rows, never batched)
        tag = "" if ds == "stressstrain" else "_nobatch"
    pat = f"{HERE}/results_llmsr/{ds}_s[0-9]_n1000{tag}.json"
    rs = [r for r in load_glob(pat, 5) if r.get("tag", "") == tag]
    assert len(rs) == 5, (ds, tag, len(rs))
    return rs


def table_pysr_reproduction():
    rows = []
    stats = {}
    for ds in DS:
        a = pysr_runs(ds, "default"); b = pysr_runs(ds, "_v3cfg_nobatch")
        rID, rOOD = REPORTED_PYSR[ds]
        mA = (finite_median([r["ID"] for r in a]), finite_median([r["OOD"] for r in a]))
        mB = (finite_median([r["ID"] for r in b]), finite_median([r["OOD"] for r in b]))
        bB = (min(r["ID"] for r in b), min(r["OOD"] for r in b))
        stats[ds] = dict(default=mA, v3=mB, v3best=bB, n_default=len(a), n_v3=len(b),
                         improve_med=[mB[0] < rID, mB[1] < rOOD], improve_best=[bB[0] < rID, bB[1] < rOOD])
        rows.append(f"{DSNAME[ds]} & {fmt(rID)} & {fmt(rOOD)} & {fmt(bB[0])} & {fmt(bB[1])} & {fmt(mB[0])} & {fmt(mB[1])} & {fmt(mA[0])} & {fmt(mA[1])} \\\\")
    n = min(min(v['n_default'], v['n_v3']) for v in stats.values())
    s = ("\\begin{tabular}{lcccccccc}\n\\toprule\n"
         " & \\multicolumn{2}{c}{Reported} & \\multicolumn{2}{c}{Stated, best} & \\multicolumn{2}{c}{Stated, median} & \\multicolumn{2}{c}{Default, median} \\\\\n"
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
         "Dataset & ID & OOD & ID & OOD & ID & OOD & ID & OOD \\\\\n\\midrule\n" + "\n".join(rows) +
         "\n\\bottomrule\n\\end{tabular}\n")
    write("pysr_reproduction.tex", s)
    return stats, n


# ---- Table: same-protocol LLM-SR rerun vs PySR --------------------------------------------
def table_gain_partition(stats):
    rows = []
    for ds in DS:
        rr = load_glob(f"{HERE}/results_llmsr_rerun/{ds}_r*.json", 3)
        l = (finite_median([r["ID"] for r in rr]), finite_median([r["OOD"] for r in rr]))
        lo_hi = (min(r["OOD"] for r in rr), max(r["OOD"] for r in rr))
        p = stats[ds]["default"]
        rows.append(f"{DSNAME[ds]} & {fmt(REPORTED_LLMSR_L8[ds][0])} & {fmt(REPORTED_LLMSR_L8[ds][1])} & "
                    f"{fmt(l[0])} & {fmt(l[1])} & {fmt(p[0])} & {fmt(p[1])} & {fmt(REPORTED_DECA_L8[ds][0])} & {fmt(REPORTED_DECA_L8[ds][1])} \\\\")
        stats[ds]["llmsr_rerun"] = l; stats[ds]["llmsr_ood_range"] = lo_hi; stats[ds]["n_llmsr"] = len(rr)
    s = ("\\begin{tabular}{lcccccccc}\n\\toprule\n"
         " & \\multicolumn{2}{c}{LLM-SR rep.} & \\multicolumn{2}{c}{LLM-SR rerun} & \\multicolumn{2}{c}{PySR rerun} & \\multicolumn{2}{c}{DecAEvolve rep.} \\\\\n"
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
         "Dataset & ID & OOD & ID & OOD & ID & OOD & ID & OOD \\\\\n\\midrule\n" + "\n".join(rows) +
         "\n\\bottomrule\n\\end{tabular}\n")
    write("gain_partition.tex", s)


# ---- Table: CRK -----------------------------------------------------------------------------
def table_crk():
    rows = []; wins = {1000: [0, 0], 5000: [0, 0]}
    have_llmsr = all(len(glob.glob(f"{HERE}/results_crk_llmsr/{n}_r*.json")) >= 3 for n in CRK)
    for n in CRK:
        dI, dO, lI, lO = REPORTED_CRK[n]
        cells = []
        for it in (1000, 5000):
            rs = load_glob(f"{HERE}/results_crk/{n}_s[0-9]_n{it}.json", 3)
            I, O = finite_median([r["ID"] for r in rs]), finite_median([r["OOD"] for r in rs])
            wins[it][0] += I < dI; wins[it][1] += O < dO
            if it == 1000: cells += [fmt(I), fmt(O)]
        rh = load_glob(f"{HERE}/results_crk/{n}_s[0-9]_t3600.json", 1)
        cells += [fmt(finite_median([r["ID"] for r in rh])), fmt(finite_median([r["OOD"] for r in rh]))]
        extra = ""
        if have_llmsr:
            rr = load_glob(f"{HERE}/results_crk_llmsr/{n}_r*.json", 3)
            extra = f" & {fmt(finite_median([r['ID'] for r in rr]))} & {fmt(finite_median([r['OOD'] for r in rr]))}"
        rows.append(f"{n} & " + " & ".join(cells) + extra + f" & {fmt(lI)} & {fmt(lO)} & {fmt(dI)} & {fmt(dO)} \\\\")
    head_extra = " & \\multicolumn{2}{c}{LLM-SR rerun}" if have_llmsr else ""
    cm_extra = "\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}" if have_llmsr else "\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}"
    ncol = 11 if have_llmsr else 9
    s = (f"\\begin{{tabular}}{{l{'c' * (ncol - 1)}}}\n\\toprule\n"
         " & \\multicolumn{2}{c}{PySR 1k} & \\multicolumn{2}{c}{PySR 1\\,h}" + head_extra +
         " & \\multicolumn{2}{c}{LLM-SR rep.} & \\multicolumn{2}{c}{DecAEvolve rep.} \\\\\n"
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}" + cm_extra + "\n"
         "Task" + " & ID & OOD" * ((ncol - 1) // 2) + " \\\\\n\\midrule\n" + "\n".join(rows) +
         "\n\\bottomrule\n\\end{tabular}\n")
    write("crk.tex", s)
    return wins, have_llmsr


# ---- Table: DoLQ --------------------------------------------------------------------------
# Term test: automatic rule in dolq_terms.py (prune constants whose removal changes the
# right-hand side by < 1e-3 relative RMS on the training points, then compare term skeletons).


def table_dolq():
    nm = {}
    for s in (0, 1, 2):
        rs = load_glob(f"{HERE}/results/pysr_s{s}_n1000_m30_P*.json", 8)
        assert len(rs) == 8
        nm[s] = sum(r["success_ID"] for r in rs)
    sindy = load_glob(f"{HERE}/results/sindy_P*.json", 8)
    n_sindy = sum(r["success_ID"] for r in sindy)
    cb = json.load(open(f"{HERE}/results_llmsr_rerun/closedbook_llama8b.json"))
    cb_pass = 0
    for p in range(1, 9):
        ok = [r for r in cb if r["pid"] == p and "intID" in r]
        if ok:
            best = min(ok, key=lambda r: r["train_cost"])
            cb_pass += max(best["intID"]) < 1e-3
    H = json.load(open(f"{HERE}/results_llmsr_rerun/dolq_heldout.json"))
    hp = [sum(H[str(p)][f"pysr_s{sd}"]["heldout_ic"] < 1e-3 for p in range(1, 9)) for sd in (0, 1, 2)]
    hs = sum(H[str(p)]["sindy"]["heldout_ic"] < 1e-3 for p in range(1, 9))
    hc = sum(H[str(p)].get("closedbook", {"heldout_ic": np.inf})["heldout_ic"] < 1e-3 for p in range(1, 9))
    heldout = dict(pysr=hp, sindy=hs, closedbook=hc)
    rows = [f"{k} (reported) & {v[0]}/8 & {v[1]}/8 & n/a \\\\" for k, v in REPORTED_DOLQ.items()]
    rows.append("\\midrule")
    rng = lambda a, b: f"{a}/8" if a == b else f"{a}--{b}/8"
    # automatic term test (dolq_terms.py): main threshold 1e-3, sensitivity 1e-2
    TT = json.load(open(f"{HERE}/results_llmsr_rerun/dolq_terms.json"))
    assert sorted(TT) == [str(p) for p in range(1, 9)]
    term = lambda k, dl: [p for p in range(1, 9) if k in TT[str(p)] and TT[str(p)][k][dl]["recovered"]]
    tp = [len(term(f"pysr_s{sd}", "0.001")) for sd in (0, 1, 2)]
    tp_loose = [len(term(f"pysr_s{sd}", "0.01")) for sd in (0, 1, 2)]
    # recovered and integrable from both initial conditions (unpruned model)
    stable = [len([p for p in term(f"pysr_s{sd}", "0.001") if np.isfinite(H[str(p)][f"pysr_s{sd}"]["train_ic"])
                   and np.isfinite(H[str(p)][f"pysr_s{sd}"]["heldout_ic"])]) for sd in (0, 1, 2)]
    p8_loose = sum(8 in term(f"pysr_s{sd}", "0.01") for sd in (0, 1, 2))
    ts, tc = len(term("sindy", "0.001")), len(term("closedbook", "0.001"))
    assert ts == len(term("sindy", "0.01")) and tc == len(term("closedbook", "0.01"))
    rows.append(f"PySR, each seed (ours) & {rng(min(nm.values()), max(nm.values()))} & {rng(min(tp), max(tp))} & {rng(min(hp), max(hp))} \\\\")
    rows.append(f"Dense SINDy (ours) & {n_sindy}/8 & {ts}/8 & {hs}/8 \\\\")
    rows.append(f"Closed-book LLM, no data (ours) & {cb_pass}/8 & {tc}/8 & {hc}/8 \\\\")
    s = ("\\begin{tabular}{lccc}\n\\toprule\nMethod & NMSE test & Term test & Held-out IC \\\\\n\\midrule\n" + "\n".join(rows) +
         "\n\\bottomrule\n\\end{tabular}\n")
    write("dolq.tex", s)
    return dict(pysr_nmse=nm, sindy=n_sindy, closedbook=cb_pass, heldout=heldout, term=dict(
        pysr=tp, pysr_loose=tp_loose, pysr_stable=stable, p8_loose=p8_loose, sindy=ts, closedbook=tc,
        pysr_sets=sorted({tuple(term(f"pysr_s{sd}", "0.001")) for sd in (0, 1, 2)})))


# ---- order-statistic analysis of best-of-k reporting (Section on theory) ------------------
def expected_min_normal(k, n=2_000_000, seed=0):
    z = np.random.default_rng(seed).standard_normal((n, k))
    return float(z.min(1).mean())


def theory_numbers():
    rng_ = np.random.default_rng(1)
    z5 = rng_.standard_normal((2_000_000, 5)); z3 = rng_.standard_normal((2_000_000, 3))
    e5 = float(z5.min(1).mean()); p_min5_lt_med3 = float((z5.min(1) < np.median(z3, 1)).mean())
    rows = []; sig_all = []
    for ds in DS:
        p = np.log10(clip_pos([r["OOD"] for r in pysr_runs(ds, "default")]))
        l = np.log10(clip_pos([r["OOD"] for r in load_glob(f"{HERE}/results_llmsr_rerun/{ds}_r*.json", 3)]))
        for meth, v in (("PySR", p), ("LLM-SR", l)):
            sd = float(np.std(v, ddof=1)); sig_all.append(sd)
            rows.append(f"{DSNAME[ds]} & {meth} & {len(v)} & {sd:.2f} & {-e5 * sd:.2f} & {np.median(v) - v.min():.2f} \\\\")
    tab = ("\\begin{tabular}{llcccc}\n\\toprule\nDataset & Method & runs & $\\hat\\sigma$ & predicted shift & observed shift \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("bestofk.tex", tab)
    from math import comb
    exact = 1 - comb(3, 2) / comb(8, 2)                       # Proposition 4(i), k=5, j=3
    assert abs(exact - 25 / 28) < 1e-12 and abs(p_min5_lt_med3 - exact) < 2e-3, (p_min5_lt_med3, exact)
    M5 = -1.16296                                             # tabulated E[min of 5 std normals]
    assert abs(e5 - M5) < 3e-3, e5
    e5 = M5
    return {"EMinFive": f"{e5:.3f}", "EMinFiveAbs": f"{-e5:.2f}", "PMinFiveBeatsMedThree": f"{100 * exact:.0f}",
            "SigmaMedian": f"{np.median(sig_all):.1f}", "SigmaMax": f"{max(sig_all):.1f}",
            "ShiftMedianDecades": f"{-e5 * np.median(sig_all):.1f}"}


def clip_pos(v):
    v = np.array([x if np.isfinite(x) else 1e3 for x in v], float)
    return np.clip(v, 1e-12, None)


def raw_tables():
    for split in ("ID", "OOD"):
        rows = []
        for ds in DS:
            for tag, lab in (("default", "default"), ("_v3cfg_nobatch", "stated")):
                rs = sorted(pysr_runs(ds, tag), key=lambda r: r["seed"])
                rows.append(f"{DSNAME[ds]} & PySR ({lab}) & " + " & ".join(fmt(r[split]) for r in rs) + " & & & \\\\")
            rr = sorted(load_glob(f"{HERE}/results_llmsr_rerun/{ds}_r*.json", 8), key=lambda r: int(r["log"].rsplit("_r", 1)[1]))
            rows.append(f"{DSNAME[ds]} & LLM-SR & " + " & ".join(fmt(r[split]) for r in rr) + " \\\\")
        write(f"raw_runs_{split.lower()}.tex", "\\begin{tabular}{ll" + "c" * 8 + "}\n\\toprule\nDataset & Method & " + " & ".join(str(i) for i in range(1, 9)) + " \\\\\n\\midrule\n"
              + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    H = json.load(open(f"{HERE}/results_llmsr_rerun/dolq_heldout.json"))
    rows = []
    for p in range(1, 9):
        h = H[str(p)]
        cells = [f"{fmt(h[k]['train_ic'])} / {fmt(h[k]['heldout_ic'])}" if k in h else "--" for k in ("pysr_s0", "pysr_s1", "pysr_s2", "sindy", "closedbook")]
        rows.append(f"P{p} & " + " & ".join(cells) + " \\\\")
    write("heldout_raw.tex", "\\begin{tabular}{lccccc}\n\\toprule\nSystem & PySR s0 & PySR s1 & PySR s2 & dense SINDy & closed-book \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    lines = []
    for p in range(1, 9):
        r = json.load(open(f"{HERE}/results/pysr_s0_n1000_m30_P{p}.json"))
        lines.append(f"P{p} {r['name']}")
        for j, (e, tr) in enumerate(zip(r["exprs"], r["truth"])):
            lines.append(f"  dx{j}/dt  PySR : {e}")
            lines.append(f"           truth: {tr}")
    open(os.path.join(OUT, "dolq_exprs.txt"), "w").write("\n".join(lines) + "\n")
    print("wrote raw tables")


# ---- bootstrap intervals for the same-protocol comparison -------------------------------
def bootstrap_partition(B=20000, seed=0):
    rng_ = np.random.default_rng(seed); rows = []; out = {}
    for ds in DS:
        res = {}
        for split in ("ID", "OOD"):
            p = np.log10(clip_pos([r[split] for r in pysr_runs(ds, "default")]))
            l = np.log10(clip_pos([r[split] for r in load_glob(f"{HERE}/results_llmsr_rerun/{ds}_r*.json", 3)]))
            bp = np.median(rng_.choice(p, (B, len(p))), 1); bl = np.median(rng_.choice(l, (B, len(l))), 1)
            d = bp - bl                      # log10 (PySR median / LLM-SR median); > 0 favours LLM-SR
            res[split] = (np.median(p) - np.median(l), np.percentile(d, 2.5), np.percentile(d, 97.5), len(p), len(l))
        out[ds] = res
        f = lambda t: f"{t[0]:+.1f} [{t[1]:+.1f}, {t[2]:+.1f}]"
        rows.append(f"{DSNAME[ds]} & {res['ID'][3]} & {res['ID'][4]} & {f(res['ID'])} & {f(res['OOD'])} \\\\")
    write("bootstrap.tex", "\\begin{tabular}{lcccc}\n\\toprule\nDataset & PySR runs & LLM-SR runs & ID & OOD \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    return out


# ---- macros for every number quoted in the running text ---------------------------------
# ---- matched compute (Section "compute") ------------------------------------------------------
# Three clocks for LLM-SR: its wall-clock time on one GPU shared by many concurrent runs (the runs
# in the paper), its wall-clock time on a dedicated GPU (timing runs, llmsr_timing.py), and GPU-hours
# amortised over the concurrent runs. Accuracy at a budget comes from llmsr_at_budget.py.
KEYS = [("OscOne", "oscillator1"), ("OscTwo", "oscillator2"), ("Ecoli", "bactgrow"), ("Stress", "stressstrain")]


def pysr_hour(ds):
    pat = f"{HERE}/results_crk/{ds}_s*_t3600.json" if ds.startswith("CRK") else f"{HERE}/results_llmsr/{ds}_s*_nobatch_t3600.json"
    return load_glob(pat, 3)


def boot_log_ratio(p, l, rng_, B=20000):
    """log10(median PySR / median LLM-SR) (> 0 favours LLM-SR), resampling runs within each method."""
    p, l = np.log10(clip_pos(p)), np.log10(clip_pos(l))
    d = np.median(rng_.choice(p, (B, len(p))), 1) - np.median(rng_.choice(l, (B, len(l))), 1)
    return float(np.median(p) - np.median(l)), d


def sgn(v):
    return f"{v:+.2f}" if abs(v) < 0.05 else f"{v:+.1f}"


def compute_matched():
    T = json.load(open(f"{HERE}/results_llmsr_rerun/dedicated_timing.json"))
    AT = json.load(open(f"{HERE}/results_llmsr_rerun/llmsr_at_budget.json"))
    conc = lambda run: 12 if int(run.rsplit("_r", 1)[1]) <= 3 else 20       # concurrent runs sharing the GPU
    tim = lambda ds, ded: [t for t in T if t["ds"] == ds and t["dedicated"] == ded]
    end_h = lambda ts: float(np.median([t["wall"][-1]["wall_s"] for t in ts])) / 3600
    rng_ = np.random.default_rng(0); m = {}; rows = []; acc = []; slow = []; gen = []
    for t in T:
        if t["dedicated"]:
            gen.append(t["sample_s"] / (t["sample_s"] + t["evaluate_s"]))
    for key, ds in KEYS:
        sh, de = tim(ds, False), tim(ds, True)
        assert len(sh) == 8 and len(de) == 2, (ds, len(sh), len(de))
        at = [a for a in AT if a["ds"] == ds]; assert len(at) == 8, (ds, len(at))
        wsh, wde = end_h(sh), end_h(de); slow.append(wsh / wde)
        gpu = float(np.median([t["wall"][-1]["wall_s"] / conc(t["run"]) for t in sh])) / 3600
        nsh = int(np.median([a["shared"]["n"] for a in at])); nde = at[0]["dedicated"]["n"]
        lsh, lde = [a["shared"]["OOD"] for a in at], [a["dedicated"]["OOD"] for a in at]
        lend = [r["OOD"] for r in load_glob(f"{HERE}/results_llmsr_rerun/{ds}_r*.json", 8)]
        ph = pysr_hour(ds); p = [r["OOD"] for r in ph]
        d, bd = boot_log_ratio(p, lde, rng_); lo, hi = np.percentile(bd, [2.5, 97.5])
        rows.append(f"{DSNAME[ds]} & {wsh:.1f} & {wde:.1f} & {gpu:.2f} & {nsh} & {nde} \\\\")
        acc.append(f"{DSNAME[ds]} & {fmt(finite_median(lsh))} & {fmt(finite_median(lde))} & {fmt(finite_median(lend))} & "
                   f"{fmt(finite_median(p))} ({len(p)}) & ${sgn(d)}$ $[{sgn(lo)}, {sgn(hi)}]$ \\\\")
        m.update({f"{key}LlmsrWallHours": f"{wsh:.1f}", f"{key}LlmsrDedWallHours": f"{wde:.1f}", f"{key}LlmsrGpuHours": f"{gpu:.2f}",
                  f"{key}LlmsrOneHourSamples": str(nsh), f"{key}LlmsrDedHourSamples": str(nde),
                  f"{key}LlmsrOneHourOOD": fmt(finite_median(lsh)), f"{key}LlmsrDedHourOOD": fmt(finite_median(lde)),
                  f"{key}LlmsrOneHourExact": str(sum(o < 1e-6 for o in lsh)), f"{key}LlmsrDedHourExact": str(sum(o < 1e-6 for o in lde)),
                  f"{key}PysrHourN": str(len(p)), f"{key}PysrHourExact": str(sum(o < 1e-6 for o in p)),
                  f"{key}PysrHourOOD": fmt(finite_median(p)), f"{key}PysrHourID": fmt(finite_median([r["ID"] for r in ph])),
                  f"{key}HourBoot": sgn(d), f"{key}HourBootLo": sgn(lo), f"{key}HourBootHi": sgn(hi)})
    # CRK: per task, LLM-SR (6 runs) at one hour on each clock and at the end, against one-hour PySR (3 seeds)
    csh = [t for t in T if t["ds"].startswith("CRK") and not t["dedicated"]]
    cde = [t for t in T if t["ds"].startswith("CRK") and t["dedicated"]]
    assert sorted({t["ds"] for t in cde}) == sorted(CRK), sorted({t["ds"] for t in cde})
    slow.append(end_h(csh) / end_h(cde))
    win = {"shared": [], "dedicated": [], "end": []}; boots = {k: [] for k in win}; decisive = [0, 0]
    for n in CRK:
        at = [a for a in AT if a["ds"] == n]; assert len(at) == 6, (n, len(at))
        p = [r["OOD"] for r in pysr_hour(n)]
        lend = [r["OOD"] for r in load_glob(f"{HERE}/results_crk_llmsr/{n}_r*.json", 6)]
        for k, l in (("shared", [a["shared"]["OOD"] for a in at]), ("dedicated", [a["dedicated"]["OOD"] for a in at]), ("end", lend)):
            d, bd = boot_log_ratio(p, l, rng_)
            win[k].append(d > 0); boots[k].append(bd > 0)
            if k == "dedicated":
                lo, hi = np.percentile(bd, [2.5, 97.5]); decisive[0] += hi < 0; decisive[1] += lo > 0
    for k, tag in (("shared", "AtOneHour"), ("dedicated", "AtDedHour"), ("end", "AtEnd")):
        cnt = np.sum(boots[k], 0)
        m[f"CrkLlmsrBeatsPysrHour{tag}"] = str(sum(win[k]))
        m[f"CrkLlmsrBeatsPysrHour{tag}Lo"], m[f"CrkLlmsrBeatsPysrHour{tag}Hi"] = (str(int(v)) for v in np.percentile(cnt, [2.5, 97.5]))
    m["CrkHourPysrDecisive"], m["CrkHourLlmsrDecisive"] = str(decisive[0]), str(decisive[1])
    m["CrkHourTasks"] = str(len(CRK)); m["CrkPysrHourN"] = str(min(len(pysr_hour(n)) for n in CRK))
    m["CrkLlmsrWallHours"], m["CrkLlmsrDedWallHours"] = f"{end_h(csh):.1f}", f"{end_h(cde):.1f}"
    m["CrkDedHourSamples"] = str(int(np.median([a["dedicated"]["n"] for a in AT if a["ds"].startswith("CRK")])))
    m["CrkDecaBeatsPysrHour"] = str(sum(REPORTED_CRK[n][1] < finite_median([r["OOD"] for r in pysr_hour(n)]) for n in CRK))
    m["SharedSlowdownLo"], m["SharedSlowdownHi"] = f"{min(slow):.1f}", f"{max(slow):.1f}"
    m["GenShareLo"], m["GenShareHi"] = f"{100 * min(gen):.0f}", f"{100 * max(gen):.0f}"
    rows.append(f"CRK (10 tasks) & {end_h(csh):.1f} & {end_h(cde):.1f} & -- & "
                f"{int(np.median([a['shared']['n'] for a in AT if a['ds'].startswith('CRK')]))} & {m['CrkDedHourSamples']} \\\\")
    write("compute.tex", "\\begin{tabular}{lccccc}\n\\toprule\n"
          " & \\multicolumn{2}{c}{Run time (h)} & GPU-h per run & \\multicolumn{2}{c}{Samples in 1\\,h} \\\\\n"
          "\\cmidrule(lr){2-3}\\cmidrule(lr){5-6}\n"
          "Dataset & shared & dedicated & (amortised) & shared & dedicated \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\n\\medskip\n\n"
          "\\begin{tabular}{lccccc}\n\\toprule\n"
          " & \\multicolumn{3}{c}{LLM-SR} & PySR & \\\\\n"
          "\\cmidrule(lr){2-4}\n"
          "Dataset & 1\\,h shared & 1\\,h dedicated & end & 1\\,h (seeds) & $\\Delta$ [95\\%] \\\\\n\\midrule\n"
          + "\n".join(acc) + "\n\\bottomrule\n\\end{tabular}\n")
    return m


def ratio(a, b):
    r = a / b
    return f"{r:,.0f}" if r >= 10 else f"{r:.1f}"


def write_macros():
    S = json.load(open(os.path.join(OUT, "summary.json")))
    st = S["stats"]; m = {}
    for key, ds in [("OscOne", "oscillator1"), ("OscTwo", "oscillator2"), ("Ecoli", "bactgrow"), ("Stress", "stressstrain")]:
        rep = REPORTED_PYSR[ds]; v3 = st[ds]["v3"]; dfl = st[ds]["default"]; ll = st[ds]["llmsr_rerun"]
        m[f"{key}IDGainStated"] = ratio(rep[0], v3[0]); m[f"{key}OODGainStated"] = ratio(rep[1], v3[1])
        m[f"{key}IDGainDefault"] = ratio(rep[0], dfl[0]); m[f"{key}OODGainDefault"] = ratio(rep[1], dfl[1])
        m[f"{key}OODPysrOverLlmsr"] = ratio(ll[1], dfl[1]) if dfl[1] < ll[1] else "--"
        m[f"{key}OODLlmsrOverPysr"] = ratio(dfl[1], ll[1]) if ll[1] < dfl[1] else "--"
        lo, hi = st[ds]["llmsr_ood_range"]
        m[f"{key}LlmsrOODSpanDecades"] = f"{np.log10(hi / lo):.0f}"
    m["NPysrSeeds"] = str(S["n_pysr_seeds"])
    m["NCellsImproveMedian"] = str(sum(sum(st[d]["improve_med"]) for d in DS))
    m["NCellsImproveBest"] = str(sum(sum(st[d]["improve_best"]) for d in DS))
    for key, ds in [("OscOne", "oscillator1"), ("OscTwo", "oscillator2"), ("Ecoli", "bactgrow"), ("Stress", "stressstrain")]:
        rep = REPORTED_PYSR[ds]; bb = st[ds]["v3best"]
        m[f"{key}IDGainStatedBest"] = ratio(rep[0], bb[0]); m[f"{key}OODGainStatedBest"] = ratio(rep[1], bb[1])
        ll = st[ds]["llmsr_rerun"]; dfl = st[ds]["default"]
        m[f"{key}IDPysrOverLlmsr"] = ratio(ll[0], dfl[0]) if dfl[0] < ll[0] else "--"
        m[f"{key}IDLlmsrOverPysr"] = ratio(dfl[0], ll[0]) if ll[0] < dfl[0] else "--"
    w = S["crk_pysr_wins_vs_decaevolve"]
    m["CrkPysrWinsIDOneK"], m["CrkPysrWinsOODOneK"] = str(w["1000"][0]), str(w["1000"][1])
    m["CrkPysrWinsIDFiveK"], m["CrkPysrWinsOODFiveK"] = str(w["5000"][0]), str(w["5000"][1])
    d = S["dolq"]; m["DolqSindyPass"] = str(d["sindy"]); m["DolqClosedbookPass"] = str(d["closedbook"])
    m["DolqPysrNmsePass"] = str(min(d["pysr_nmse"].values()))
    for key, ds in [("OscOne", "oscillator1"), ("OscTwo", "oscillator2")]:
        rs = pysr_runs(ds, "default"); o = [r["OOD"] for r in rs if np.isfinite(r["OOD"])]
        m[f"{key}PysrOODSpanDecades"] = f"{np.log10(max(o) / min(o)):.0f}"
    nb = nt = 0
    for n in CRK:
        rr = load_glob(f"{HERE}/results_crk_llmsr/{n}_r*.json", 3)
        pp = load_glob(f"{HERE}/results_crk/{n}_s[0-9]_n1000.json", 3)
        lo = finite_median([r["OOD"] for r in rr]); po = finite_median([r["OOD"] for r in pp])
        if lo < po / 1.5: nb += 1
        elif po < lo / 1.5: pass
        else: nt += 1
    m["CrkLlmsrBeatsPysrOOD"] = str(nb); m["CrkLlmsrTiesPysrOOD"] = str(nt)
    imp = [np.log10(REPORTED_PYSR[d][i] / ref[d][i]) for d in ("oscillator1", "oscillator2") for i in (0, 1)
           for ref in (REPORTED_LLMSR_L8, REPORTED_DECA_L8)]
    m["ImpliedOrdersLo"], m["ImpliedOrdersHi"] = f"{min(imp):.1f}", f"{max(imp):.1f}"
    ho = S["dolq"]["heldout"]
    tt = S["dolq"]["term"]
    assert len(tt["pysr_sets"]) == 1, tt["pysr_sets"]          # the same systems for every seed
    assert tt["pysr_stable"] == tt["pysr"], tt                   # text: every recovered structure integrates stably
    assert min(tt["pysr"]) == max(tt["pysr"]) and min(tt["pysr_stable"]) == max(tt["pysr_stable"])
    m["DolqTermPysr"] = str(tt["pysr"][0]); m["DolqTermPysrStable"] = str(tt["pysr_stable"][0])
    m["DolqTermPysrSystems"] = ", ".join(f"P{p}" for p in tt["pysr_sets"][0])
    m["DolqTermPysrLooseMax"] = str(max(tt["pysr_loose"])); m["DolqPEightLooseSeeds"] = str(tt["p8_loose"])
    m["DolqTermSindy"] = str(tt["sindy"]); m["DolqTermClosedbook"] = str(tt["closedbook"])
    m["DolqHeldoutPysr"] = str(min(ho["pysr"])); m["DolqHeldoutSindy"] = str(ho["sindy"]); m["DolqHeldoutClosedbook"] = str(ho["closedbook"])
    th = theory_numbers(); m.update(th)
    bs = bootstrap_partition()
    m["NLlmsrRuns"] = str(min(v["OOD"][4] for v in bs.values()))
    for key, ds in [("OscOne", "oscillator1"), ("OscTwo", "oscillator2"), ("Ecoli", "bactgrow"), ("Stress", "stressstrain")]:
        for sp in ("ID", "OOD"):
            v = bs[ds][sp]
            m[f"{key}{sp}Boot"], m[f"{key}{sp}BootLo"], m[f"{key}{sp}BootHi"] = f"{v[0]:+.1f}", f"{v[1]:+.1f}", f"{v[2]:+.1f}"
    m["NBootExcludeZero"] = str(sum((bs[d][sp][1] > 0) or (bs[d][sp][2] < 0) for d in DS for sp in ("ID", "OOD")))
    for key, ds in [("OscOne", "oscillator1"), ("OscTwo", "oscillator2"), ("Ecoli", "bactgrow"), ("Stress", "stressstrain")]:
        rr = sorted(load_glob(f"{HERE}/results_llmsr_rerun/{ds}_r*.json", 8), key=lambda r: int(r["log"].rsplit("_r", 1)[1]))
        pp = pysr_runs(ds, "default") + pysr_runs(ds, "_v3cfg_nobatch")
        m[f"{key}LlmsrExact"] = str(sum(r["OOD"] < 1e-6 for r in rr)); m[f"{key}NLlmsr"] = str(len(rr))
        m[f"{key}PysrExact"] = str(sum(r["OOD"] < 1e-6 for r in pp)); m[f"{key}NPysr"] = str(len(pp))
        m[f"{key}LlmsrMedFirstThree"] = fmt(finite_median([r["OOD"] for r in rr[:3]]))
        m[f"{key}LlmsrMedAll"] = fmt(finite_median([r["OOD"] for r in rr]))
        m[f"{key}FirstThreeVsAllDecades"] = f"{abs(np.log10(finite_median([r['OOD'] for r in rr[:3]]) / finite_median([r['OOD'] for r in rr]))):.0f}"
    m.update(compute_matched())
    m["EcoliPysrOneKOOD"] = fmt(finite_median([r["OOD"] for r in pysr_runs("bactgrow", "default")]))
    ek = [r["OOD"] for r in pysr_runs("bactgrow", "default")]
    m["EcoliPysrHeavyOneK"], m["EcoliPysrNOneK"] = str(sum(not (o < 100) for o in ek)), str(len(ek))
    m["EcoliPysrHeavyHour"] = str(sum(not (o < 100) for o in [r["OOD"] for r in pysr_hour("bactgrow")]))
    nc = [len(load_glob(f"{HERE}/results_crk_llmsr/{n}_r*.json", 6)) for n in CRK]
    m["NCrkLlmsrRuns"] = str(min(nc))
    rs = json.load(open(f"{HERE}/results_llmsr_rerun/refit_sensitivity.json"))
    m["RefitRuns"], m["RefitRestarted"], m["RefitUnmatched"] = str(rs["n"]), str(rs["restarted"]), str(rs["unmatched"])
    m["RefitRemoteDisagreePct"], m["RefitRemoteMaxFactor"] = f"{100 * rs['frac_gt_2']:.0f}", f"{rs['max']:,.0f}"
    tex = "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(m.items()))
    open(os.path.join(OUT, "numbers.tex"), "w").write(tex)
    print(tex)



if __name__ == "__main__":
    stats, n_seeds = table_pysr_reproduction()
    table_gain_partition(stats)
    wins, have_llmsr = table_crk()
    dq = table_dolq()
    summary = dict(n_pysr_seeds=n_seeds, crk_pysr_wins_vs_decaevolve=wins, crk_llmsr_rerun=have_llmsr, dolq=dq,
                   stats={k: {kk: vv for kk, vv in v.items()} for k, v in stats.items()})
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    raw_tables()
    write_macros()
