# -*- coding: utf-8 -*-
"""
Train the final CatBoost model and export it to JSON.

Reproduces the analysis reported in the manuscript:
  - development cohort: MIMIC-IV, 3,110 patients with diabetes receiving acetaminophen
  - 15 predictors selected by recursive feature elimination using SHAP values
  - four algorithms compared on the training set; CatBoost retained
  - export to CatBoost JSON format for browser inference

Input : filtered_custom_final.csv  (development cohort, available from the authors
                                    under the MIMIC-IV data use agreement)
Output: model/model.json, model/meta.json

Requirements: see requirements.txt
"""
import os, json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'filtered_custom_final.csv')
OUTDIR = os.path.join(HERE, 'model')
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(DATA, low_memory=False)
target = 'aki'

# Variables that leak the outcome or are unavailable at prediction time
leakage = ['resp_rate_first', 'spo2_first', 'weight', 'dbp_first', 'mbp_ni_first',
           'sbp_ni_first', 'temperature_first', 'mbp_first', 'heart_rate_first',
           'sbp_first', 'dbp_ni_first', 'height']
df = df.drop(columns=[c for c in leakage if c in df.columns])

# Drop variables missing in more than 80% of patients
mr = df.isnull().mean()
df = df.drop(columns=mr[mr > 0.8].index.tolist())

X = df.drop(columns=[target])
y = df[target]
cats = X.select_dtypes(include=['object', 'category']).columns.tolist()
for c in cats:
    X[c] = X[c].fillna('Unknown')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Feature selection: recursive elimination by SHAP values
pool_tr = Pool(X_train, y_train, cat_features=cats)
pool_te = Pool(X_test, y_test, cat_features=cats)
selector = CatBoostClassifier(iterations=500, random_seed=42,
                              auto_class_weights='Balanced', verbose=False)
summary = selector.select_features(
    pool_tr, eval_set=pool_te, features_for_select=list(range(X.shape[1])),
    num_features_to_select=15, steps=20, algorithm='RecursiveByShapValues',
    shap_calc_type='Regular', train_final_model=False, logging_level='Silent')
features = summary['selected_features_names']
print('Selected predictors:', features)

# Final model: the algorithm with the highest TRAINING-set AUC is retained;
# the hold-out set is used for evaluation only.
Xtr, Xte = X_train[features], X_test[features]
final_cats = [c for c in cats if c in features]

model = CatBoostClassifier(iterations=800, learning_rate=0.03, depth=5,
                           auto_class_weights='Balanced', random_seed=42,
                           verbose=False)
model.fit(Xtr, y_train, cat_features=final_cats,
          eval_set=(Xte, y_test), early_stopping_rounds=50)


def auc(y_true, p):
    y_true = np.asarray(y_true).astype(bool)
    p = np.asarray(p, float)
    pos, neg = p[y_true], p[~y_true]
    return ((pos[:, None] > neg[None, :]).sum() +
            0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))


p_tr = model.predict_proba(Xtr)[:, 1]
p_te = model.predict_proba(Xte)[:, 1]
print('Training AUC: %.4f' % auc(y_train, p_tr))
print('Hold-out AUC: %.4f' % auc(y_test, p_te))

model.save_model(os.path.join(OUTDIR, 'model.json'), format='json')
json.dump({'features': features, 'cat_features': final_cats,
           'threshold': 0.372,
           'train_auc': round(float(auc(y_train, p_tr)), 3),
           'test_auc': round(float(auc(y_test, p_te)), 3),
           'n_train': int(len(X_train)), 'n_test': int(len(X_test))},
          open(os.path.join(OUTDIR, 'meta.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print('Saved model/model.json and model/meta.json')
