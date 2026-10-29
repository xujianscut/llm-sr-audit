"""Extra one-hour PySR seeds (review round). Sequential, same laptop as all other PySR runs.
A run that exceeds 1.5 h or crashes is logged and replaced by the next unused seed (at most once)."""
import os, subprocess, sys, time
PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "env", "bin", "python")
CRK = ["CRK3", "CRK4", "CRK7", "CRK12", "CRK15", "CRK17", "CRK19", "CRK22", "CRK28", "CRK30"]
jobs = [("crk", n, s) for n in CRK for s in ((2, 3) if n == "CRK17" else (1, 2))]
jobs += [("llmsr", d, s) for d in ("oscillator1", "oscillator2", "bactgrow") for s in (3, 4)]
jobs += [("llmsr", "stressstrain", s) for s in (0, 1, 2)]


def out(kind, name, s):
    return f"results_crk/{name}_s{s}_t3600.json" if kind == "crk" else f"results_llmsr/{name}_s{s}_nobatch_t3600.json"


def run(kind, name, s):
    if os.path.exists(out(kind, name, s)):
        return True
    script = "srbench_crk.py" if kind == "crk" else "srbench_pysr.py"
    env = dict(os.environ, TBUDGET="3600", NOBATCH="1")
    t0 = time.time()
    try:
        p = subprocess.run([PY, script, name, str(s), "1000"], env=env, timeout=5400, capture_output=True, text=True)
        ok = p.returncode == 0 and os.path.exists(out(kind, name, s))
        msg = p.stdout.strip().splitlines()[-1] if ok else f"FAIL rc={p.returncode} {p.stderr[-300:]!r}"
    except subprocess.TimeoutExpired:
        ok, msg = False, "TIMEOUT"
    print(f"{time.strftime('%m-%d %H:%M')} {name} s{s} {time.time()-t0:.0f}s {msg}", flush=True)
    return ok


for kind, name, s in jobs:
    if not run(kind, name, s):
        run(kind, name, s + 10)
print("QUEUE_DONE", flush=True)
