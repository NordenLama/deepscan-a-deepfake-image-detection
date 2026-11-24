# DeepScan — Deepfake Image Detection

This project classifies full images as **real** or **AI-generated/manipulated** with
hand-engineered HOG shape, LBP texture, and FFT frequency features, plus supervised
scikit-learn models.
It is an academic prototype; its output is not proof of authenticity.

## Dataset

Supply labeled full images in this layout:

```text
backend/data/
├── real/   # authentic face images
└── fake/   # manipulated or synthetic face images
```

Supported image formats are JPEG, PNG, BMP, WebP, and TIFF. Training and inference
use the whole image, resized with aspect ratio preserved; no face detection is
required. For scene detection, include diverse real photographs and AI-generated
scenes in both classes (not only face portraits), with varied subjects, sources,
resolutions, and lighting. Keep the classes representative of the intended use and
use images with appropriate rights.

## Install, train, and run

The saved `.joblib` models are stored with Git LFS because the ensemble exceeds
GitHub's regular file-size limit. Install Git LFS before cloning so the model files
are downloaded rather than their small pointer files:

```powershell
git lfs install
git clone https://github.com/NordenLama/deepscan-a-deepfake-image-detection.git
Set-Location deepscan-a-deepfake-image-detection
```

From the project directory, create/activate a virtual environment and install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
```

Train models with stratified cross-validation:

```powershell
python backend\train_models.py --data-dir backend\data
```

Training evaluates and saves SVM, Random Forest, ANN (`MLPClassifier`), Decision
Tree, Naive Bayes, and a soft-voting ensemble to `backend\saved_models\`. Random
oversampling is applied to training folds only; validation folds remain untouched.
All labeled examples are retained, including rare but valid cases. Metrics include
accuracy, precision, recall, F1-score, and ROC-AUC.
Naive Bayes is retained for coursework comparison but is not exposed as an inference
choice because it labeled the supplied AI examples as real after retraining.
After changing the dataset or preprocessing, retrain the models to refresh the saved
artifacts. If the API is already running, stop it with Ctrl+C and restart it after
training; model artifacts are loaded into memory at API startup. The website defaults
to SVM because it had the strongest cross-validation
accuracy, F1-score, and fake-class recall on the current dataset. The
reported cross-validation is a random image-level split within this dataset; it does
not measure performance on new sources or generators. Most images are 300×300 face
portraits, with only a few user-supplied AI examples beyond that distribution. The
scores do not establish reliable detection of unfamiliar synthetic images or
scene-level AI content. Add diverse real and fake examples from the intended
sources/generators, and evaluate against a separate source-held-out test set. At
inference, the API reports an uncertain result when the
classifier's top class probability is below 65%; a confident result can still be
wrong, especially for images outside the training distribution. Treat outputs as
screening signals, not proof of authenticity.

Start the API:

```powershell
python -m uvicorn backend.main:app --reload
```

If running Uvicorn from inside `backend`, use `python -m uvicorn main:app --reload`.
The frontend posts an image and the selected model to `http://127.0.0.1:8000/predict`.
Open `frontend\index.html` in a browser after the API is running. `GET /health`
reports which model artifacts were loaded. The API starts without trained models so it
can report a useful error; `/predict` returns HTTP 503 until the requested model exists.

## Syllabus scope

The implementation uses supervised feature engineering (HOG/LBP/FFT), training-fold
imbalance handling, an ANN trained by backpropagation, classical classifiers,
cross-validation, standard classification metrics, and bagging/soft voting. It also
reports supplementary Module 6 association rules between above-median LBP
histogram bins and class labels; these rules are descriptive and do not replace the
supervised classifiers. It does not use Hidden Markov Models or clustering.
