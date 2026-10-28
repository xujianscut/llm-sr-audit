"""DecAEvolve Table 3 (LLM-SRBench LSR-Synth chem_react 10 题) 的非 LLM 对照: PySR.
NMSE = MSE / Var(y) (LLM-SRBench 协议); hdf5 列序 = symbols, 第 0 列为输出 ('O').
"""
import sys, json, os, time
import numpy as np, pandas as pd, h5py

H5 = os.path.join(os.path.dirname(__file__), "srbench", "lsr_bench_data.hdf5")
META = os.path.join(os.path.dirname(__file__), "srbench", "lsr_synth_chem_react-00000-of-00001.parquet")


def nmse(y, p):
    return float(np.mean((y - p) ** 2) / np.var(y))


def main():
    name, seed, niter = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    meta = pd.read_parquet(META).set_index("name").loc[name]
    props = str(meta["symbol_properties"]).strip("[]").replace("'", "").split()
    assert props[0] == "O" and all(p == "V" for p in props[1:]), props
    g = h5py.File(H5, "r")[f"lsr_synth/chem_react/{name}"]
    tr, te, ood = [np.array(g[k]) for k in ["train", "test", "ood_test"]]
    assert tr.shape == (4000, len(props))
    from pysr import PySRRegressor
    t0 = time.time()
    tbudget = os.environ.get("TBUDGET")
    if tbudget:
        niter = 1000000
    m = PySRRegressor(niterations=niter, **({"timeout_in_seconds": float(tbudget)} if tbudget else {}), binary_operators=["+", "-", "*", "/", "^"],
                      unary_operators=["sin", "cos", "exp", "log", "sqrt"], maxsize=35,
                      constraints={"^": (-1, 1)}, random_state=seed, verbosity=0, progress=False,
                      temp_equation_file=True, model_selection="best")
    m.fit(tr[:, 1:], tr[:, 0], variable_names=["t", "A"][: tr.shape[1] - 1])
    eqs = m.equations_; i_acc = int(np.argmin(eqs["loss"].values))
    res = dict(name=name, seed=seed, niter=niter, wall_s=time.time() - t0, expr=str(m.sympy()),
               truth=str(meta["expression"]),
               ID=nmse(te[:, 0], m.predict(te[:, 1:])), OOD=nmse(ood[:, 0], m.predict(ood[:, 1:])),
               acc_pick=dict(expr=str(eqs["sympy_format"].iloc[i_acc]),
                             ID=nmse(te[:, 0], m.predict(te[:, 1:], index=i_acc)),
                             OOD=nmse(ood[:, 0], m.predict(ood[:, 1:], index=i_acc))))
    os.makedirs("results_crk", exist_ok=True)
    tag = f"_t{tbudget}" if tbudget else f"_n{niter}"
    res["tag"] = tag
    json.dump(res, open(f"results_crk/{name}_s{seed}{tag}.json", "w"), indent=1)
    print(f"{name} s{seed} {res['wall_s']:.0f}s best ID={res['ID']:.2e} OOD={res['OOD']:.2e} | minloss ID={res['acc_pick']['ID']:.2e} OOD={res['acc_pick']['OOD']:.2e}", flush=True)


if __name__ == "__main__":
    main()
