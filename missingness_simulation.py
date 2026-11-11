# -*- coding: utf-8 -*-
"""
Missingness simulation.

Sets the 15 predictors in the development hold-out set to missing at the rates
observed in the external validation cohort, repeats 20 times, and reports the
mean AUC. Used to quantify how much of the loss of discrimination on transfer
is explained by differential availability of laboratory tests.

Input : filtered_custom_final.csv, external_missing_rates.json
Output: printed AUC before and after the simulation

external_missing_rates.json holds one key per predictor with the proportion
missing in the external validation cohort. It contains patient-derived
information about the single-centre cohort and is therefore not distributed
with this repository; the corresponding proportions are reported in S2 Table
of the manuscript. Place your own copy next to this script to run the analysis.

A placeholder file, external_missing_rates.template.json, shows the required
format. It contains illustrative values only and will not reproduce the
published result; the script refuses to run with it.
"""
import os, json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'filtered_custom_final.csv')
RATES = os.path.join(HERE, 'external_missing_rates.json')
MODELDIR = os.path.join(HERE, 'model')

if not os.path.exists(RATES):
    raise SystemExit(
        'external_missing_rates.json not found.\n'
        'This file holds the proportion of each predictor missing in the external\n'
        'validation cohort (S2 Table of the manuscript). Restore it from the\n'
        'repository to run the analysis.'
    )

features = json.load(open(os.path.join(MODELDIR, 'meta.json'), encoding='utf-8'))['features']
rates = json.load(open(RATES, encoding='utf-8'))

# Each entry carries the external rate, the development rate, and whether the
# predictor became less available on transfer. Refuse to run on a partial file.
bad = [f for f in features
       if not isinstance(rates.get(f), dict)
       or not isinstance(rates[f].get('external_missing'), (int, float))
       or not isinstance(rates[f].get('development_missing'), (int, float))]
if bad:
    raise SystemExit(
        'external_missing_rates.json is malformed: %d of %d predictors lack a numeric\n'
        'external_missing/development_missing value.\nMissing: %s'
        % (len(bad), len(features), ', '.join(bad))
    )

# Only predictors whose availability deteriorated are set to missing; the others
# were recorded more completely in the external cohort and are left unchanged.
SIMULATED = [f for f in features if rates[f]['external_missing'] > rates[f]['development_missing']]
print('Predictors set to missing (%d): %s' % (len(SIMULATED), ', '.join(SIMULATED)))
print('Predictors held unchanged (%d): %s'
      % (len(features) - len(SIMULATED),
         ', '.join(f for f in features if f not in SIMULATED)))

if not os.path.exists(DATA):
    raise SystemExit(
        'filtered_custom_final.csv not found.\n'
        'This file contains the development cohort derived from MIMIC-IV and cannot be\n'
        'redistributed. Regenerate it from MIMIC-IV (PhysioNet, credentialed access)\n'
        'using the inclusion and exclusion criteria in the Methods section.'
    )

df = pd.read_csv(DATA, low_memory=False)
target = 'aki'
leakage = ['resp_rate_first', 'spo2_first', 'weight', 'dbp_first', 'mbp_ni_first',
           'sbp_ni_first', 'temperature_first', 'mbp_first', 'heart_rate_first',
           'sbp_first', 'dbp_ni_first', 'height']
df = df.drop(columns=[c for c in leakage if c in df.columns])
mr = df.isnull().mean()
df = df.drop(columns=mr[mr > 0.8].index.tolist())

X = df.drop(columns=[target])
y = df[target]
cats = X.select_dtypes(include=['object', 'category']).columns.tolist()
for c in cats:
    X[c] = X[c].fillna('Unknown')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)


def auc(y_true, p):
    y_true = np.asarray(y_true).astype(bool)
    p = np.asarray(p, float)
    pos, neg = p[y_true], p[~y_true]
    return ((pos[:, None] > neg[None, :]).sum() +
            0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))


model = CatBoostClassifier(iterations=800, learning_rate=0.03, depth=5,
                           auto_class_weights='Balanced', random_seed=42,
                           verbose=False)
model.fit(X_train[features], y_train,
          cat_features=[c for c in cats if c in features],
          eval_set=(X_test[features], y_test), early_stopping_rounds=50)

Xte = X_test[features]
base = auc(y_test, model.predict_proba(Xte)[:, 1])
print('Hold-out AUC (observed): %.4f' % base)

rng = np.random.RandomState(42)
sim = []
for _ in range(20):
    Xm = Xte.copy()
    for f in SIMULATED:
        r = float(rates[f]['external_missing'])
        if r <= 0:
            continue
        Xm.loc[rng.rand(len(Xm)) < r, f] = np.nan
    sim.append(auc(y_test, model.predict_proba(Xm)[:, 1]))

print()
print('Hold-out AUC (observed):                 %.4f' % base)
print('Hold-out AUC (after simulated missingness): %.4f (SD %.4f)'
      % (float(np.mean(sim)), float(np.std(sim))))
print('Reported in the manuscript: 0.873 observed, 0.742 after simulated missingness')

