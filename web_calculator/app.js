/* CatBoost oblivious-tree inference and user interface */
(function () {
  'use strict';

  const FEATS = META.features;
  const floatFeatures = MODEL.features_info.float_features;

  const idxToPos = {};
  floatFeatures.forEach(function (ff, pos) { idxToPos[ff.feature_index] = pos; });

  const nameToPos = {};
  floatFeatures.forEach(function (ff, pos) {
    const nm = FEATS[ff.feature_index];
    if (nm !== undefined) nameToPos[nm] = pos;
  });

  const SCALE = (MODEL.scale_and_bias && MODEL.scale_and_bias[0] !== undefined)
    ? Number(MODEL.scale_and_bias[0]) : 1;
  let BIAS = 0;
  if (MODEL.scale_and_bias && MODEL.scale_and_bias[1] !== undefined) {
    const b = MODEL.scale_and_bias[1];
    BIAS = Array.isArray(b) ? Number(b[0] || 0) : Number(b || 0);
  }

  function sigmoid(z) {
    if (z > 40) return 1;
    if (z < -40) return 0;
    return 1 / (1 + Math.exp(-z));
  }

  function predict(x) {
    const trees = MODEL.oblivious_trees;
    let sum = 0;
    for (let t = 0; t < trees.length; t++) {
      const tree = trees[t];
      const splits = tree.splits;
      let idx = 0;
      for (let j = 0; j < splits.length; j++) {
        const sp = splits[j];
        const pos = idxToPos[sp.float_feature_index];
        const v = x[pos];
        let bit = 0;
        if (v !== null && v !== undefined && !Number.isNaN(v)) {
          // CatBoost borders are float32; compare at float32 precision so that a value
        // sitting exactly on a border is classified as not greater, as in CatBoost.
        bit = Math.fround(v) > Math.fround(sp.border) ? 1 : 0;
        }
        idx |= (bit << j);
      }
      sum += tree.leaf_values[idx];
    }
    return sigmoid(sum * SCALE + BIAS);
  }

  const form = document.getElementById('risk-form');
  const resultBox = document.getElementById('result');
  const probEl = document.getElementById('prob');
  const barFill = document.getElementById('bar-fill');
  const barMark = document.getElementById('bar-mark');
  const bandEl = document.getElementById('band');
  const adviceEl = document.getElementById('advice');

  function readInputs() {
    const x = new Array(floatFeatures.length).fill(NaN);
    FEATS.forEach(function (name) {
      const el = document.getElementById(name);
      if (!el) return;
      const raw = el.value.trim();
      if (raw === '') return;
      const v = Number(raw);
      if (Number.isNaN(v)) return;
      const pos = nameToPos[name];
      if (pos !== undefined) x[pos] = v;
    });
    return x;
  }

  function render(p) {
    const pct = p * 100;
    probEl.textContent = pct.toFixed(1) + '%';
    barFill.style.width = Math.min(100, pct) + '%';
    barMark.style.left = (META.threshold * 100) + '%';

    let cls, text, advice;
    if (p < META.threshold) {
      cls = 'low'; text = 'Low risk';
      advice = 'Below the decision threshold. Routine creatinine monitoring is likely sufficient.';
      barFill.style.background = 'var(--low)';
    } else if (p < 0.6) {
      cls = 'mid'; text = 'Above threshold \u00b7 intermediate to high risk';
      advice = 'Above the decision threshold. Consider more frequent creatinine measurement, avoiding additional nephrotoxic drugs, and reviewing volume status and oxygenation.';
      barFill.style.background = 'var(--mid)';
    } else {
      cls = 'high'; text = 'High risk';
      advice = 'Well above the decision threshold. Close monitoring of renal function is advisable, together with a review of whether acetaminophen and other nephrotoxic exposures remain necessary and correction of modifiable risk factors.';
      barFill.style.background = 'var(--high)';
    }
    bandEl.className = 'band ' + cls;
    bandEl.textContent = text;
    adviceEl.textContent = advice;
    resultBox.classList.remove('hidden');
    resultBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    const x = readInputs();
    const filled = x.filter(function (v) { return !Number.isNaN(v); }).length;
    if (filled === 0) {
      alert('Please enter at least one laboratory value.');
      return;
    }
    render(predict(x));
  });

  const DEMO = {
    aado2_calc_first: 320, pao2fio2ratio_first: 240, pco2_first: 42,
    baseexcess_first: -3, bicarbonate_first: 22, ptt_first: 32, inr_first: 1.25,
    fibrinogen_first: 300, ck_cpk_first: 180, bun_first: 22, albumin_first: 3.3,
    calcium_free_first: 1.12, immature_granulocytes_first: 0.9,
    hemoglobin_a1c_first: 8.1, smoker: 0
  };
  document.getElementById('fill-demo').addEventListener('click', function () {
    Object.keys(DEMO).forEach(function (k) {
      const el = document.getElementById(k);
      if (el) el.value = DEMO[k];
    });
  });

  window.__predict = predict;
  window.__FEATS = FEATS;
})();
