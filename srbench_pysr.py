"""重跑 LLM-SR (ICLR25) / DecAEvolve (ICML26) Table 1 里被抄录的 PySR 数字.
数据 = LLM-SR 官方仓库 data/{oscillator1,oscillator2,bactgrow,stressstrain}, 最后一列为目标.
NMSE = sum (y - yhat)^2 / sum (y - ybar)^2, 分 test_id / test_ood.
"""
import sys, json, os, time
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "LLM-SR", "data")


def nmse(y, p):
    return float(((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    ds, seed, niter = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    from pysr import PySRRegressor
    tr, ti, to = [pd.read_csv(f"{ROOT}/{ds}/{s}.csv") for s in ["train", "test_id", "test_ood"]]
    cols = tr.columns.tolist(); xc, yc = cols[:-1], cols[-1]
    assert len(tr) > 100 and set(cols) == set(ti.columns) == set(to.columns)
    t0 = time.time()
    tbudget = os.environ.get("TBUDGET")
    if tbudget:
        niter = 1000000
    m = PySRRegressor(niterations=niter, **({"timeout_in_seconds": float(tbudget)} if tbudget else {}), binary_operators=["+", "-", "*", "/", "^"],
                      unary_operators=["sin", "cos", "exp", "log", "sqrt"], maxsize=35,
                      constraints={"^": (-1, 1)}, random_state=seed, verbosity=0, progress=False,
                      temp_equation_file=True, model_selection="best",
                      batching=len(tr) > 5000 and not os.environ.get("NOBATCH"), batch_size=1000,
                      **({"populations": 15, "population_size": 33} if os.environ.get("V3CFG") else {}))
    m.fit(tr[xc].values, tr[yc].values, variable_names=xc)
    wall = time.time() - t0
    eqs = m.equations_
    best = m.sympy()
    res = dict(ds=ds, seed=seed, niter=niter, wall_s=wall, expr=str(best),
               train=nmse(tr[yc].values, m.predict(tr[xc].values)),
               ID=nmse(ti[yc].values, m.predict(ti[xc].values)),
               OOD=nmse(to[yc].values, m.predict(to[xc].values)))
    # 同时记录 Pareto 前沿上按训练损失最小(而非 score)选出的方程
    i_acc = int(np.argmin(eqs["loss"].values))
    res["acc_pick"] = dict(expr=str(eqs["sympy_format"].iloc[i_acc]), complexity=int(eqs["complexity"].iloc[i_acc]),
                           ID=nmse(ti[yc].values, m.predict(ti[xc].values, index=i_acc)),
                           OOD=nmse(to[yc].values, m.predict(to[xc].values, index=i_acc)))
    os.makedirs("results_llmsr", exist_ok=True)
    tag = ("_v3cfg" if os.environ.get("V3CFG") else "") + ("_nobatch" if os.environ.get("NOBATCH") else "") + (f"_t{tbudget}" if tbudget else "")
    res["tag"] = tag
    json.dump(res, open(f"results_llmsr/{ds}_s{seed}" + ("" if tbudget else f"_n{niter}") + f"{tag}.json", "w"), indent=1)
    print(f"{ds} s{seed} {wall:.0f}s  best: ID={res['ID']:.2e} OOD={res['OOD']:.2e}  | min-loss pick: ID={res['acc_pick']['ID']:.2e} OOD={res['acc_pick']['OOD']:.2e}  expr={res['expr'][:120]}", flush=True)


if __name__ == "__main__":
    main()
