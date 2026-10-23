"""DoLQ 闭卷测试: 只给系统描述(DoLQ 实际喂给 LLM 的 description 字段)+变量数, 不给数据.
LLM 直接写 ODE 右端(常数 c0..), 再仅用 train 拟合常数(无结构搜索), 按 DoLQ 的积分 NMSE 打分.
"""
import json, re, sys, urllib.request, os
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

URL, MODEL = os.environ.get("VLLM_URL", "http://127.0.0.1:8011/v1/chat/completions"), os.environ["VLLM_MODEL"]
DATA = sys.argv[1]; K = int(sys.argv[2]); OUT = sys.argv[3]
PROMPT = """You are an expert scientist. A dynamical system is described as follows:

"{desc}"

The system has {d} state variables x0, ..., x{dm1}. Write the governing ordinary differential equations dx_i/dt = f_i(x0, ..., x{dm1}).
Use Python/numpy syntax (np.sin, np.exp, ...), state variables x0..x{dm1}, and unknown constants c0, c1, c2, ... (no numeric values).
Answer ONLY with JSON: {{"equations": ["<f_0>", ..., "<f_{dm1}>"]}}"""


def ask(desc, d, n):
    body = dict(model=MODEL, messages=[{"role": "user", "content": PROMPT.format(desc=desc, d=d, dm1=d - 1)}],
                n=n, temperature=0.8, max_tokens=1024)
    r = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return [c["message"]["content"] for c in json.loads(urllib.request.urlopen(r, timeout=900).read())["choices"]]


def parse(txt, d):
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    m = re.search(r"\{.*\}", txt, flags=re.S)
    eqs = json.loads(m.group(0))["equations"]
    assert len(eqs) == d
    return [e.split("=")[-1].strip() for e in eqs]


def nmse(a, b):
    return float(((a - b) ** 2).sum() / ((a ** 2).sum() + 1e-12))


def fit_eval(eqs, d, tr, te):
    nc = 1 + max([int(c) for e in eqs for c in re.findall(r"c(\d+)", e)] + [-1])
    def to_src(e):
        e = re.sub(r"c(\d+)", r"c[\1]", e)
        return "lambda X, c: " + re.sub(r"x(\d+)", r"X[\1]", e)
    fs = [eval(to_src(e), {"np": np}) for e in eqs]
    X = tr[[f"x{j}" for j in range(d)]].values.T; Y = tr[[f"x{j}_t" for j in range(d)]].values.T
    res = lambda c: np.concatenate([np.broadcast_to(f(X, c), Y[j].shape) - Y[j] for j, f in enumerate(fs)])
    best = None
    for s in range(8):
        c0 = np.random.default_rng(s).uniform(0.1, 3, max(nc, 1))
        try:
            r = least_squares(res, c0, max_nfev=4000)
            if np.isfinite(r.cost) and (best is None or r.cost < best.cost): best = r
        except Exception:
            pass
    if best is None: return None
    c = best.x
    Xte = te[[f"x{j}" for j in range(d)]].values; t = te["t"].values
    ncall = [0]
    def rhs(tt, x):
        ncall[0] += 1
        if ncall[0] > 200000:                    # 刚性/发散方程判失败, 防止积分空转
            raise RuntimeError("rhs budget")
        return np.array([np.broadcast_to(f(x, c), ()).item() for f in fs])
    try:
        sol = solve_ivp(rhs, (t[0], t[-1]), Xte[0], t_eval=t, method="LSODA", rtol=1e-10, atol=1e-12)
        integ = [nmse(Xte[:, j], sol.y[j]) for j in range(d)] if sol.success and sol.y.shape[1] == len(t) else [np.inf] * d
    except Exception:
        integ = [np.inf] * d
    return dict(consts=c.tolist(), train_cost=float(best.cost), intID=integ)


out = []
for pid in range(1, 9):
    spec = json.load(open(f"{DATA}/json/ID_0{pid}.json")); d = spec["dimension"]
    base = f"{DATA}/{d}D/ID_0{pid}/ID_0{pid}"
    tr, te = pd.read_csv(base + "_train.csv"), pd.read_csv(base + "_test_id.csv")
    for txt in ask(spec["description"], d, K):
        rec = dict(pid=pid, truth=spec["equation"]["python"], raw=txt[-600:])
        try:
            rec["eqs"] = parse(txt, d); rec.update(fit_eval(rec["eqs"], d, tr, te) or {})
        except Exception as e:
            rec["err"] = str(e)[:200]
        out.append(rec)
    ok = [r for r in out if r["pid"] == pid and "intID" in r]
    best = min(ok, key=lambda r: r["train_cost"]) if ok else None
    npass = sum(max(r["intID"]) < 1e-3 for r in ok)
    print(f"P{pid}: parsed {len(ok)}/{K}, pass NMSE-test {npass}/{K}; best-by-train-fit: {best['eqs'] if best else None} intID={max(best['intID']) if best else None:.2e}" if best else f"P{pid}: none parsed", flush=True)
    print(f"      truth: {spec['equation']['python']}", flush=True)
json.dump(out, open(OUT, "w"), indent=1)
