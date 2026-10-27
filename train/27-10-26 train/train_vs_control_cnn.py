import argparse

import numpy as np

from nn import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=FEATURE_SETS, default="both")
    ap.add_argument("--only", nargs="*", choices=SYMPTOMS)
    ap.add_argument("--no-final", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="print validation precision/recall/F1 every epoch")
    a = ap.parse_args()
    D = load(a.features)
    cls = D["meta"]["symptom"].to_numpy()
    d = out_dir(f"vs_control_{a.features}")
    print(f"cnn | features={a.features} ({D['X'].shape[1]} columns, {D['seq'].shape[1]} sequence channels) | windows={len(cls):,} | people={len(D['people'])}")
    results = []
    for sym in a.only or SYMPTOMS:
        yb = (cls == sym).astype(int)
        D["symptom"] = sym

        def fit_predict(tr, va, sets):
            return fit_nn("cnn", D, tr, va, sets, yb, f"[{sym}] ", a.verbose)

        res = cv_symptom(D, sym, fit_predict, f"[{sym}] ")
        results.append(symptom_report(D, sym, res, d))
        if not a.no_final:
            rows = np.flatnonzero((cls == sym) | (cls == "control"))
            final_nn("cnn", D, rows, yb, int(np.median([i.get("best_epoch", 30) for i in res["infos"]])), d / sym)
    write_summary(results, d)


if __name__ == "__main__":
    main()
