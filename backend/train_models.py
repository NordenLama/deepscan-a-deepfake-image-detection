"""Train, cross-validate, and save the project's supervised image classifiers."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

if __package__:
    from .feature_extractor import (
        HOG_FEATURE_SIZE,
        LBP_BINS,
        decode_image,
        features_from_image,
    )
else:
    from feature_extractor import (
        HOG_FEATURE_SIZE,
        LBP_BINS,
        decode_image,
        features_from_image,
    )

SEED = 42
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_DIRECTORIES = {"real": 0, "fake": 1}
MODEL_FILENAMES = {
    "svm": "svm.joblib",
    "random_forest": "random_forest.joblib",
    "ann": "ann.joblib",
    "decision_tree": "decision_tree.joblib",
    "naive_bayes": "naive_bayes.joblib",
    "ensemble": "ensemble.joblib",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("train_models")


def build_models() -> dict[str, Any]:
    """Create fresh estimators for each cross-validation fold and final fit."""
    return {
        "svm": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                estimator=SVC(
                    C=5.0,
                    kernel="rbf",
                    class_weight="balanced",
                    random_state=SEED,
                ),
                method="sigmoid",
                cv=2,
                ensemble=False,
            ),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=24,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
        "ann": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=300,
                early_stopping=False,
                random_state=SEED,
            ),
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=24,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
        ),
        "naive_bayes": make_pipeline(StandardScaler(), GaussianNB(var_smoothing=1e-8)),
    }


def build_ensemble(models: dict[str, Any] | None = None) -> VotingClassifier:
    base_models = models if models is not None else build_models()
    return VotingClassifier(
        estimators=[(name, base_models[name]) for name in MODEL_FILENAMES if name != "ensemble"],
        voting="soft",
        n_jobs=-1,
    )


def load_dataset(data_directory: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read labeled images and build HOG/LBP/FFT features with progress tracking."""
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []

    for class_name, label in CLASS_DIRECTORIES.items():
        class_directory = data_directory / class_name
        if not class_directory.is_dir():
            raise FileNotFoundError(
                f"Expected a '{class_name}' image folder at {class_directory}."
            )

        image_paths = sorted(
            path
            for path in class_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not image_paths:
            raise ValueError(f"No supported images found in {class_directory}.")

        logger.info("Extracting features for class '%s' (%d images)...", class_name, len(image_paths))
        for image_path in tqdm(image_paths, desc=f"Loading {class_name}"):
            try:
                image_bytes = image_path.read_bytes()
                image = decode_image(image_bytes)
                feature_rows.append(features_from_image(image))
                labels.append(label)
            except Exception as error:
                logger.warning("Skipping unreadable image %s: %s", image_path, error)

    if not feature_rows:
        raise ValueError(f"No usable images were found under {data_directory}.")

    features = np.vstack(feature_rows)
    target = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(target, minlength=2)
    if np.any(counts == 0):
        raise ValueError(
            "Both real/ and fake/ folders must contain readable images."
        )
    logger.info(
        "Successfully loaded %d images (real=%d, fake=%d), each with %d features.",
        len(target),
        counts[0],
        counts[1],
        features.shape[1],
    )
    return features, target


def balance_training_data(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly oversample minority classes using training data only."""
    classes, counts = np.unique(labels, return_counts=True)
    target_count = int(counts.max())
    rng = np.random.default_rng(seed)
    indices = [
        np.concatenate(
            (
                np.flatnonzero(labels == class_label),
                rng.choice(
                    np.flatnonzero(labels == class_label),
                    size=target_count - int(count),
                    replace=True,
                ),
            )
        )
        for class_label, count in zip(classes, counts)
    ]
    balanced_indices = np.concatenate(indices)
    rng.shuffle(balanced_indices)
    return features[balanced_indices], labels[balanced_indices]


def mine_association_rules(features: np.ndarray, labels: np.ndarray) -> list[dict[str, float | str]]:
    """Mine one-item LBP-bin to class rules, ranked by confidence and lift."""
    lbp_histograms = features[:, HOG_FEATURE_SIZE:HOG_FEATURE_SIZE + LBP_BINS]
    thresholds = np.median(lbp_histograms, axis=0)
    rules: list[dict[str, float | str]] = []
    sample_count = len(labels)

    for bin_index, threshold in enumerate(thresholds):
        present = lbp_histograms[:, bin_index] > threshold
        antecedent_count = int(np.count_nonzero(present))
        if antecedent_count == 0:
            continue
        for class_label, class_name in ((0, "real"), (1, "fake")):
            class_count = int(np.count_nonzero(labels == class_label))
            matching_count = int(np.count_nonzero(present & (labels == class_label)))
            support = matching_count / sample_count
            confidence = matching_count / antecedent_count
            lift = confidence / (class_count / sample_count)
            if support >= 0.02 and confidence >= 0.5 and lift > 1:
                rules.append(
                    {
                        "antecedent": f"lbp_bin_{bin_index}_above_median",
                        "consequent": f"class_{class_name}",
                        "support": round(support, 4),
                        "confidence": round(confidence, 4),
                        "lift": round(lift, 4),
                    }
                )

    return sorted(
        rules,
        key=lambda rule: (float(rule["lift"]), float(rule["confidence"])),
        reverse=True,
    )[:20]


def _positive_probability(estimator: Any, features: np.ndarray) -> np.ndarray:
    probabilities = estimator.predict_proba(features)
    classes = list(estimator.classes_)
    if 1 not in classes:
        raise ValueError("A trained classifier is missing the fake class (label 1).")
    return probabilities[:, classes.index(1)]


def _calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    fake_probabilities: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "precision": round(float(precision_score(labels, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(labels, fake_probabilities)), 4),
    }


def evaluate_models(
    features: np.ndarray,
    labels: np.ndarray,
    folds: int,
) -> dict[str, dict[str, float]]:
    """Compute out-of-fold metrics; outlier filtering and balancing stay inside each fold."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    model_names = list(MODEL_FILENAMES)
    out_of_fold_predictions = {
        name: np.empty(len(labels), dtype=np.int64) for name in model_names
    }
    out_of_fold_probabilities = {
        name: np.empty(len(labels), dtype=np.float64) for name in model_names
    }

    for fold_number, (train_indices, validation_indices) in enumerate(
        splitter.split(features, labels),
        start=1,
    ):
        train_features = features[train_indices]
        train_labels = labels[train_indices]
        train_features, train_labels = balance_training_data(
            train_features,
            train_labels,
            SEED + fold_number,
        )
        fold_models = build_models()
        fold_models["ensemble"] = build_ensemble(fold_models)

        for name, estimator in fold_models.items():
            fitted = clone(estimator)
            fitted.fit(train_features, train_labels)
            validation_features = features[validation_indices]
            out_of_fold_predictions[name][validation_indices] = fitted.predict(
                validation_features
            )
            out_of_fold_probabilities[name][validation_indices] = _positive_probability(
                fitted,
                validation_features,
            )
        logger.info("Completed cross-validation fold %d/%d.", fold_number, folds)

    return {
        name: _calculate_metrics(
            labels,
            out_of_fold_predictions[name],
            out_of_fold_probabilities[name],
        )
        for name in model_names
    }


def train_and_save(
    data_directory: Path,
    model_directory: Path,
    folds: int = 5,
) -> dict[str, dict[str, float]]:
    features, labels = load_dataset(data_directory)
    class_counts = np.bincount(labels, minlength=2)
    if np.min(class_counts) < 4:
        raise ValueError(
            "At least four readable images per class are needed for stratified "
            "cross-validation and SVM probability calibration."
        )
    folds = min(folds, int(np.min(class_counts)))
    if folds < 2:
        raise ValueError("Cross-validation requires at least two folds.")

    metrics = evaluate_models(features, labels, folds)
    association_rules = mine_association_rules(features, labels)
    for name, scores in metrics.items():
        logger.info(
            "%s | accuracy=%.3f precision=%.3f recall=%.3f f1=%.3f ROC-AUC=%.3f",
            name,
            scores["accuracy"],
            scores["precision"],
            scores["recall"],
            scores["f1"],
            scores["roc_auc"],
        )

    balanced_features, balanced_labels = balance_training_data(
        features,
        labels,
        SEED,
    )
    final_models = build_models()
    final_models["ensemble"] = build_ensemble(final_models)
    model_directory.mkdir(parents=True, exist_ok=True)

    for name, estimator in final_models.items():
        estimator.fit(balanced_features, balanced_labels)
        joblib.dump(
            {
                "estimator": estimator,
                "metrics": metrics[name],
                "feature_extractor": "whole-image HOG+LBP+FFT",
                "class_mapping": {"real": 0, "fake": 1},
            },
            model_directory / MODEL_FILENAMES[name],
        )
        logger.info("Saved %s to %s", name, model_directory / MODEL_FILENAMES[name])

    metrics_path = model_directory / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    rules_path = model_directory / "association_rules.json"
    rules_path.write_text(json.dumps(association_rules, indent=2), encoding="utf-8")
    logger.info(
        "Saved cross-validation metrics to %s and %d association rules to %s. "
        "All labeled images were retained for the final fit.",
        metrics_path,
        len(association_rules),
        rules_path,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data_directory = Path(__file__).resolve().parent / "data"
    default_model_directory = Path(__file__).resolve().parent / "saved_models"
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_directory,
        help="Dataset with real/ and fake/ subdirectories.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_directory,
        help="Directory in which trained joblib artifacts are written.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Maximum stratified CV folds.")
    arguments = parser.parse_args()

    if arguments.folds < 2:
        parser.error("--folds must be at least 2.")
    train_and_save(arguments.data_dir, arguments.model_dir, arguments.folds)


if __name__ == "__main__":
    main()