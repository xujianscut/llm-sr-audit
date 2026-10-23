"""Evaluate a finished LLM-SR run: the program with the best training score is refitted on the
training split until it reproduces the training loss recorded during search (see llmsr_anytime.evaluate),
then ID/OOD NMSE are reported. Usage: python llmsr_eval.py <run_log_dir> <problem_name>"""
import json, sys
from llmsr_anytime import anytime

if __name__ == "__main__":
    res = anytime((sys.argv[1], sys.argv[2]))
    last = res["curve"][-1]
    out = dict(log=sys.argv[1], ds=sys.argv[2], n_samples=last["n"], ID=last["ID"], OOD=last["OOD"],
               refit_matched=last["matched"], restarts=last["restarts"])
    json.dump(out, open(f"{sys.argv[1]}/eval.json", "w"), indent=1)
    print(json.dumps(out))
