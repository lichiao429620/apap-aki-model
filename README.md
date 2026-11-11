# Acetaminophen-associated AKI risk model

Code, model weights and web calculator accompanying the manuscript:

> Machine learning model for predicting acetaminophen-associated acute kidney injury in hospitalized
> patients with diabetes: a development and external validation study

## Contents

| Path | Description |
|---|---|
| `train_model.py` | Trains the final CatBoost model and exports it to JSON |
| `missingness_simulation.py` | Reproduces the analysis of why performance falls on transfer |
| `model/model.json` | Trained CatBoost model in CatBoost JSON format |
| `model/meta.json` | Predictor list, decision threshold, AUC values |
| `external_missing_rates.json` | Proportion of each predictor missing in each cohort (S2 Table) |
| `web_calculator/` | Self-contained browser calculator (open `index.html`) |
| `requirements.txt` | Python dependencies |

## Missingness simulation

`missingness_simulation.py` quantifies how much of the loss of discrimination on transfer is
explained by differences in which laboratory tests were ordered. For each predictor,
`external_missing_rates.json` records the proportion missing in the development and external
cohorts, as reported in S2 Table of the manuscript.

Only predictors that were **less** completely recorded in the external cohort are set to
missing, at the external rate; the three predictors recorded more completely externally
(fibrinogen, albumin and creatine kinase) are left unchanged. Setting all fifteen predictors
to the external rates instead would model a loss of availability that did not occur, and
gives a lower AUC. The simulation is repeated 20 times from a fixed seed.

`external_missing_rates.template.json` shows the file layout with placeholder values; the
script stops with an explanatory message if the real file is absent or still contains
placeholders.

## Model summary

- Algorithm: CatBoost (oblivious trees, 15 predictors)
- Development cohort: MIMIC-IV, 2,177 training and 933 hold-out patients
- Hold-out AUC: 0.873
- Decision threshold: 0.372 (Youden index in the external cohort)
- External validation: single-centre cohort of 570 admissions, AUC 0.716 before recalibration

## Predictors

- `smoker`
- `pao2fio2ratio_first`
- `pco2_first`
- `aado2_calc_first`
- `baseexcess_first`
- `calcium_free_first`
- `immature_granulocytes_first`
- `albumin_first`
- `bun_first`
- `ptt_first`
- `inr_first`
- `fibrinogen_first`
- `ck_cpk_first`
- `bicarbonate_first`
- `hemoglobin_a1c_first`

Missing values are handled natively by CatBoost and must **not** be imputed.

## Running the web calculator

Open `web_calculator/index.html` in any modern browser. No server, build step or
installation is required, and no patient data leaves the browser. The calculator is
also hosted at https://warm-concha-a598ee.netlify.app

## Running inference without retraining

Model weights are provided in `model/model.json`, so predictions can be reproduced without
access to any patient data. Reload them with:

```python
from catboost import CatBoostClassifier
import json

model = CatBoostClassifier()
model.load_model('model/model.json', format='json')

features = json.load(open('model/meta.json'))['features']
# pass a DataFrame with columns in the order given by `features`
proba = model.predict_proba(df[features])[:, 1]
```

The browser calculator in `web_calculator/app.js` implements the same inference in plain
JavaScript. Note that CatBoost stores split thresholds at single precision, so comparisons
are made with `Math.fround()`; comparing at double precision changes the result by up to
0.005 for values that fall exactly on a threshold.

## Retraining from source data

```bash
pip install -r requirements.txt
python train_model.py
```

`train_model.py` expects `filtered_custom_final.csv` next to the script. That file
contains the development cohort derived from MIMIC-IV and cannot be redistributed directly;
it can be regenerated from MIMIC-IV (PhysioNet, credentialed access) using the inclusion
and exclusion criteria described in the Methods section of the manuscript.

## Data availability

The MIMIC-IV database is available from PhysioNet subject to completion of a data use
agreement and required training (https://physionet.org/content/mimiciv/). The single-centre
dataset is not publicly available because it contains patient-identifiable information;
de-identified analysis tables are available from the corresponding author on reasonable request.

## License

MIT. See `LICENSE`.

## Disclaimer

Research use only. Not a medical device. Not validated prospectively. Clinical decisions must
not be based on this model alone.
