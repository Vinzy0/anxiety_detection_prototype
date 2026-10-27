import argparse

import joblib
import numpy as np
from lightgbm import LGBMClassifier

from common import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=FEATURE_SETS, default="both")
    ap.add_argument("--only", nargs="*", choices=SYMPTOMS)
    ap.add_argument("--no-final", action="store_true")
    a = ap.parse_args()
    D = load(a.features)
    X, cls = D["X"], D["meta"]["symptom"].to_numpy()
    d = out_dir(f"vs_control_lgbm_{a.features}")
    print(f"LightGBM | features={a.features} ({X.shape[1]} columns) | windows={len(X):,} | people={len(D['people'])}")
    results = []
    for sym in a.only or SYMPTOMS:
        yb = (cls == sym).astype(int)

        def fit_predict(tr, va, sets):
            both = np.r_[tr, va]
            m = LGBMClassifier(**LGBM_PARAMS).fit(X[both], yb[both])
            return [m.predict_proba(X[s]) if len(s) else np.zeros((0, 2)) for s in sets], {}

        res = cv_symptom(D, sym, fit_predict, f"[{sym}] ")
        results.append(symptom_report(D, sym, res, d))
        if not a.no_final:
            rows = np.flatnonzero((cls == sym) | (cls == "control"))
            joblib.dump(dict(model=LGBMClassifier(**LGBM_PARAMS).fit(X[rows], yb[rows]), features=D["names"], symptom=sym,
                             feature_set=a.features, window=WINDOW, threshold=THRESHOLD), d / sym / "model.joblib")
    write_summary(results, d)


if __name__ == "__main__":
    main()
