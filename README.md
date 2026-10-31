# Dry-Eye Auto-Detection System

An end-to-end pipeline that decides whether a user has dry-eye disease (DED)
by combining:

* **Behavioural features** mined from a short face video (blink rate, palpebral
  fissure height, complete vs. incomplete blinks, eye-closing/opening
  durations, …).
* **Ocular surface signal** — eye-redness probability produced by a fine-tuned
  **ResNet-50** CNN on the most-open-eye frame of the same video.
* The patient-reported **OSDI** score (parsed from the video filename).

All of the above are concatenated into a single feature vector that is then
classified by an **XGBoost** model whose ground truth is the
`OSDI + NIBUT` clinical gold standard.

```
  Front-end / mobile app
        │  POST /upload_video  (.webm)
        ▼
  receive_videos.py  (Flask, port 5001)
        │  bucket files into /data/videos/YYYY-MM-DD/HH-MM/
        ▼
  dryeye_evaluate_20250212.py   (cron-style, every 5 minutes)
        │
        ├── Stage 1   DE_system/blinking_detect.py
        │             └─ dlib 68-pt landmarks → blinking_series.xlsx
        │
        ├── Stage 2   DE_Vfeature_extract/blinking_analyze_new.py
        │             └─ local-threshold blink detection → blinking_results.xlsx
        │
        ├── Stage 2.5 DE_RedEye/red_eye_detect.py
        │             └─ ResNet-50 on max-open-eye frame
        │                → eye_redness_prob + eye_redness_label appended
        │
        └── Stage 3   DE_XGBoost/XGBoost_predict.py
                      └─ XGBoost binary classifier → predict_results.xlsx
                         + POST :5002/return_result back to the front-end
```

---

## 1. Repository layout

```
<repo-root>/
├── README.md
├── receive_videos.py                # Flask upload endpoint (:5001)
├── dryeye_evaluate_20250212.py      # 5-minute scheduler that runs the whole pipeline
│
├── DE_system/                       # Stage 1 — frame-level eye-opening time series
│   ├── README.md                    #   how to fetch the dlib landmark weights
│   ├── blinking_detect.py
│   └── middle_stage_fig/            #   runtime-generated visualisations
│
├── DE_Vfeature_extract/             # Stage 2 — high-level blink features
│   ├── blinking_analyze_new.py
│   └── blinking_analyze_start.xlsx  #   column template for the output sheet
│
├── DE_RedEye/                       # Stage 2.5 — eye-redness CNN
│   ├── README.md                    #   ResNet-50 weight format + training recipe
│   └── red_eye_detect.py
│
├── DE_XGBoost/                      # Stage 3 — XGBoost classifier
│   ├── DE_xgboost.py                # training (5-fold CV + Bootstrap 95% CI)
│   ├── DE_xgboost_find_Random.py    # random_state / feature-subset sweep
│   ├── XGBoost_predict.py           # inference helper used by the scheduler
│   ├── AI_dryeye_diagnosis.xlsx     # training data
│   ├── xgb_model.json               # trained model (deployed)
│   ├── feature_names.pkl            # ordered feature list expected by the model
│   ├── result.jpg                   # ROC curve
│   ├── importance_results.jpg       # top-5 feature importance
│   ├── XGB_with_95CI_result_diagnose.xlsx   # metrics with 95% CI
│   └── DE_AUC.txt                   # plain-text metric dump
│
└── videos/                          # placeholder folder for the runtime video tree
```

> All scripts use the absolute prefix `/data/...`. When deploying, place the
> entire repository under `/data/` so paths resolve to `/data/DE_system/…`,
> `/data/videos/…`, `/data/DE_RedEye/…`, etc. The repository root may be
> renamed freely; only the sub-folder names matter.

---

## 2. Pipeline stages

### Stage 1 — Blinking time-series extraction (`DE_system/blinking_detect.py`)

* dlib’s frontal face detector + the 68-landmark predictor are applied to
  each frame (frames are first downscaled by `scaling = 0.3`).
* Per-frame **palpebral fissure height** is the average of
  `|y37 - y41| + |y38 - y40|` (right eye) and `|y43 - y47| + |y44 - y48|`
  (left eye).
* A second signal — browline-to-eye distance — is also computed for later
  QC and recorded in the column `periorbital_motion`.
* Visualisations of the masked eye region with the open/close trace are
  saved to `DE_system/middle_stage_fig/<video>.png`.
* All videos in a 5-minute folder are processed in parallel
  (`multiprocessing.Pool(30)`); the result is written to
  `<folder>/blinking_series.xlsx` with three columns:
  `video_name`, the per-frame eye-opening list (`blinking`), and the
  per-frame `periorbital_motion` list.

### Stage 2 — High-level blink features (`DE_Vfeature_extract/blinking_analyze_new.py`)

Reads `blinking_series.xlsx`, then for every video:

1. A 4-second sliding window computes a **local threshold** (mean −
   k · std).
2. Frames whose normalised opening drops below the local threshold are
   labelled *closed*; consecutive closed frames are merged into a single
   blink.
3. Adjacent blinks separated by less than 0.2 s are merged together.
4. A blink lasting longer than 0.2 s counts as a **complete blink**;
   shorter ones are **incomplete**.

`pid` and `OSDI` are parsed out of the video filename
(convention `pid_<id>_OSDI_<value>_*.webm`). The output
`<folder>/blinking_results.xlsx` contains, in addition to `video_name`,
`pid` and `OSDI`, the following per-video features:

| Excel column | Meaning |
| --- | --- |
| `max_palpebral_fissure_height` / `min_palpebral_fissure_height` | palpebral fissure max / min |
| `complete_blink_count` / `incomplete_blink_count` / `total_blink_count` | complete / incomplete / total blink counts |
| `complete_blink_timestamps` / `incomplete_blink_timestamps` / `blink_annotation` | blink timestamps and merged annotations |
| `max_blink_interval` | longest inter-blink interval |
| `mean_blink_duration` / `min_blink_duration` | mean / min blink duration |
| `max_eye_close_duration` / `min_eye_close_duration` | max / min eye-closing duration |
| `max_eye_open_duration` / `min_eye_open_duration` | max / min eye-opening duration |
| `video_quality` | flagged `low_quality` when the std of the trace is too high |

A per-video PNG plot of the open-rate curve, the threshold and the
detected blinks is also written next to the spreadsheet.

### Stage 2.5 — Eye-redness CNN (`DE_RedEye/red_eye_detect.py`)

A binary **ResNet-50** classifier (red eye vs. normal eye, 224×224, ImageNet
normalisation) is run on the single most representative frame of each video:

1. The blink series from Stage 1 is parsed back from
   `blinking_series.xlsx` and the **frame with the maximum eye opening**
   is located via `np.argmax`.
2. That frame is re-read from the source video, dlib runs **once** on it,
   and a padded bounding box around landmarks **36–47** crops both eyes
   together. The crop is also saved to
   `<folder>/<video>_max_open_eye.png` for auditing.
3. The crop is fed into ResNet-50; the softmax probability of the
   *red-eye* class is written back into `blinking_results.xlsx` as two
   new columns — `eye_redness_prob` in `[0, 1]` and the binary
   `eye_redness_label` (`1` when probability ≥ 0.5).

If `DE_RedEye/resnet50_redeye.pth` is not present the function logs a
warning and exits gracefully — the rest of the pipeline still runs, and
XGBoost simply ignores the absent column.

> The CNN is intentionally kept lightweight and is **not** distributed
> with the source. See `DE_RedEye/README.md` for the expected weight
> format and how to train your own.

### Stage 3 — XGBoost dry-eye classifier (`DE_XGBoost/`)

#### Training (offline)

`DE_xgboost.py` reads `AI_dryeye_diagnosis.xlsx`, with the binary label
provided by the OSDI + NIBUT clinical gold standard
(column `gold_standard_OSDI_NIBUT`). The currently shipped model is
trained on the behavioural-only feature set:

```python
independent_vars = [
    'OSDI', 'max_blink_interval',
    'max_palpebral_fissure_height', 'min_palpebral_fissure_height',
    'complete_blink_count', 'incomplete_blink_count', 'total_blink_count',
    'min_blink_duration', 'mean_blink_duration',
    'max_eye_close_duration', 'max_eye_open_duration',
]
```

To re-train **with the eye-redness feature** simply un-comment the
alternative line in `DE_xgboost.py`:

```python
independent_vars = [..., 'max_eye_open_duration', 'eye_redness_prob']
```

The training routine performs `StratifiedKFold(n_splits=5)` cross-validation
and computes 95 % bootstrap CIs (1 000 resamples) for
Accuracy / Sensitivity / Specificity / F1 / AUC. It then refits on the
full dataset to dump:

* `xgb_model.json` — the deployed model
* `feature_names.pkl` — feature order expected at inference time
* `result.jpg` — ROC
* `importance_results.jpg` — top-5 feature importance
* `XGB_with_95CI_result_diagnose.xlsx` — metric table with CIs

`DE_xgboost_find_Random.py` sweeps `random_state ∈ [6, 100)` against three
candidate feature subsets (OSDI-only, 6-feature, 12-feature) — useful for
picking the best seed during development.

#### Inference (online)

`XGBoost_predict.predict_new_data(folder_path)`:

1. Loads `<folder>/blinking_results.xlsx`.
2. Loads `xgb_model.json` and `feature_names.pkl`.
3. Coerces non-numeric values (especially `OSDI` / `pid`) into proper
   types via simple regex extraction.
4. Calls `model.predict` and `model.predict_proba`.
5. Writes `<folder>/predict_results.xlsx` with two extra columns —
   `predicted_dryeye_label` and `dryeye_prob`.
6. Returns `(pid, preds, preds_proba)` as `pandas.Series` objects.

#### Performance (current shipped model, see `DE_AUC.txt`)

| Metric | Value | 95 % CI |
| --- | --- | --- |
| AUC | **0.924** | — |
| Accuracy | 0.895 | 0.879 – 0.911 |
| Sensitivity | 0.550 | — |
| Specificity | 0.956 | — |
| F1 | 0.610 | — |

Top-5 most important features (in order): `OSDI`,
`min_palpebral_fissure_height`, `max_eye_open_duration`,
`complete_blink_count` and `total_blink_count`.

---

## 3. Upload / scheduler contract

### 3.1 Video upload (`receive_videos.py`)

```
POST http://<host>:5001/upload_video
Content-Type: multipart/form-data
form fields: video=<file.webm>
```

The server timestamps the request and saves the file to:

```
/data/videos/2025-05-07/14-25/<filename>.webm
```

Healthcheck: `GET /health` → `{"status": "ok"}`.

The filename **must** match `pid_<id>_OSDI_<value>_*.webm`; otherwise the
downstream stages cannot recover `pid` or `OSDI` from it.

### 3.2 Scheduler (`dryeye_evaluate_20250212.py`)

```python
while True:
    if datetime.now().minute % 5 == 0:
        folder = /data/videos/<date>/<HH>-<MM>
        if folder has videos:
            blinking_extract(folder)        # Stage 1
            video_feature_extract(folder)   # Stage 2
            red_eye_predict(folder)         # Stage 2.5
            pid, preds, proba = predict_new_data(folder)   # Stage 3
            POST http://localhost:5002/return_result
    sleep(60)
```

The JSON sent back to the front-end:

```json
{
  "pid":          ["123", "456"],
  "dryeye_label": [0, 1],
  "dryeye_probs": [0.12, 0.83]
}
```

---

## 4. File-naming and output conventions

| Item | Convention |
| --- | --- |
| Video bucket | `/data/videos/YYYY-MM-DD/HH-MM/` (5-minute window) |
| Video filename | `pid_<id>_OSDI_<value>_*.webm` |
| Stage 1 output | `<bucket>/blinking_series.xlsx` |
| Stage 2 output | `<bucket>/blinking_results.xlsx` (per-video blink features) |
| Stage 2.5 outputs | `<bucket>/<video>_max_open_eye.png` plus the `eye_redness_prob` and `eye_redness_label` columns appended in-place to `blinking_results.xlsx` |
| Stage 3 output | `<bucket>/predict_results.xlsx` (`predicted_dryeye_label` + `dryeye_prob`) |

---

## 5. Deployment

### 5.1 Python dependencies

```bash
# Python 3.8 recommended
pip install flask requests
pip install opencv-python dlib
pip install pandas numpy openpyxl matplotlib
pip install scikit-learn xgboost
pip install torch torchvision pillow      # for the eye-redness CNN
```

> `dlib` requires CMake / Boost / a C++ toolchain on most platforms.
> `torch` / `torchvision` versions should match your CUDA build if you
> want GPU inference; CPU works fine for a single 224×224 forward pass.

### 5.2 External model weights

Two binary blobs are intentionally **not** committed to the repository.
Each model folder ships its own `README.md` with download / training
instructions and the exact path the weights are expected at:

* **dlib 68-point face-landmark predictor** — see
  [`DE_system/README.md`](DE_system/README.md). Final path:
  `DE_system/shape_predictor_68_face_landmarks.dat`.
* **ResNet-50 eye-redness classifier** — see
  [`DE_RedEye/README.md`](DE_RedEye/README.md). Final path:
  `DE_RedEye/resnet50_redeye.pth`. If absent, Stage 2.5 is skipped
  gracefully and the rest of the pipeline still runs.

### 5.3 Launch

```bash
# Terminal 1 — upload service
python receive_videos.py
# → :5001/upload_video

# Terminal 2 — scheduler / pipeline driver
python dryeye_evaluate_20250212.py
```

The front-end / down-stream consumer must implement
`POST :5002/return_result` to receive the predictions.

---

## 6. Re-training the XGBoost classifier

When you have new labelled data, overwrite
`DE_XGBoost/AI_dryeye_diagnosis.xlsx` (it must contain the OSDI + NIBUT
gold-standard label column `gold_standard_OSDI_NIBUT` and every column
listed in `independent_vars`) and run:

```bash
cd DE_XGBoost
python DE_xgboost.py
# regenerates xgb_model.json / feature_names.pkl / result.jpg /
#             XGB_with_95CI_result_diagnose.xlsx
```

To search for a better seed / feature subset:

```bash
python DE_xgboost_find_Random.py
```

To include the eye-redness feature in the model, edit `DE_xgboost.py`
and switch to the `independent_vars` line that ends with
`'eye_redness_prob'` (both lines are present, only one is active at a
time). Make sure the training spreadsheet actually contains a populated
`eye_redness_prob` column — either by running `red_eye_predict` over the
training videos in advance or by computing it from your own labelled
crops.
