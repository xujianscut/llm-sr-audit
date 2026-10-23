"""Build LLM-SR-format data directories and prompt specifications for the ten LLM-SRBench
chemistry (CRK) tasks used by DecAEvolve, from the gated LLM-SRBench release (not redistributed).
Expects srbench/lsr_bench_data.hdf5 and srbench/lsr_synth_chem_react-00000-of-00001.parquet
(see README). Writes crk_llmsr/data/<task>/{train,test_id,test_ood}.csv and crk_llmsr/specs/."""
import os
import numpy as np, pandas as pd, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = ["CRK3", "CRK4", "CRK7", "CRK12", "CRK15", "CRK17", "CRK19", "CRK22", "CRK28", "CRK30"]


def main():
    meta = pd.read_parquet(f"{HERE}/srbench/lsr_synth_chem_react-00000-of-00001.parquet").set_index("name")
    f = h5py.File(f"{HERE}/srbench/lsr_bench_data.hdf5", "r")
    spec0 = open(f"{HERE}/LLM-SR/specs/specification_oscillator1_numpy.txt").read()
    for n in TASKS:
        assert str(meta.loc[n, "symbol_properties"]).replace("'", "").strip("[]").split() == ["O", "V", "V"]
        g = f[f"lsr_synth/chem_react/{n}"]; d = f"{HERE}/crk_llmsr/data/{n}"; os.makedirs(d, exist_ok=True)
        for k, out in [("train", "train"), ("test", "test_id"), ("ood_test", "test_ood")]:
            a = np.array(g[k]); assert a.shape[1] == 3
            pd.DataFrame({"t": a[:, 1], "A": a[:, 2], "dA_dt": a[:, 0]}).to_csv(f"{d}/{out}.csv", index=False)
        s = spec0
        for a, b in [
            ("Find the mathematical function skeleton that represents acceleration in a damped nonlinear oscillator system with driving force, given data on position, and velocity.",
             "Find the mathematical function skeleton that represents the rate of change of concentration in chemistry reaction kinetics, given data on time and concentration at time t."),
            ("x, v = inputs[:,0], inputs[:,1]", "t, A = inputs[:,0], inputs[:,1]"),
            ("y_pred = equation(x, v, params)", "y_pred = equation(t, A, params)"),
            ('def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:\n    """ Mathematical function for acceleration in a damped nonlinear oscillator\n\n    Args:\n        x: A numpy array representing observations of current position.\n        v: A numpy array representing observations of velocity.',
             'def equation(t: np.ndarray, A: np.ndarray, params: np.ndarray) -> np.ndarray:\n    """ Mathematical function for rate of change of concentration in chemistry reaction kinetics\n\n    Args:\n        t: A numpy array representing time.\n        A: A numpy array representing concentration at time t.'),
            ("A numpy array representing acceleration as the result of applying the mathematical function to the inputs.",
             "A numpy array representing rate of change of concentration in chemistry reaction kinetics as the result of applying the mathematical function to the inputs."),
            ("dv = params[0] * x  +  params[1] * v  + params[2]\n    return dv", "dA_dt = params[0] * t + params[1] * A + params[2]\n    return dA_dt")]:
            assert a in s, a[:40]; s = s.replace(a, b)
        os.makedirs(f"{HERE}/crk_llmsr/specs", exist_ok=True)
        open(f"{HERE}/crk_llmsr/specs/specification_{n}_numpy.txt", "w").write(s)
    print("built", len(TASKS), "tasks")


if __name__ == "__main__":
    main()
