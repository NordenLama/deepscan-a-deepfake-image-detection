import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend.main import MODEL_FILES, analyze_image
from backend.train_models import load_dataset
from backend.feature_extractor import prepare_image


class FixedEstimator:
    classes_ = np.array([0, 1])

    def predict(self, features):
        return np.array([1])

    def predict_proba(self, features):
        return np.array([[0.1, 0.9]])


class LowConfidenceEstimator(FixedEstimator):
    def predict(self, features):
        return np.array([0])

    def predict_proba(self, features):
        return np.array([[0.51, 0.49]])


class ConfidentRealEstimator(FixedEstimator):
    def predict(self, features):
        return np.array([0])

    def predict_proba(self, features):
        return np.array([[0.8, 0.2]])


class WholeImageTests(unittest.TestCase):
    def test_known_unreliable_naive_bayes_model_is_not_offered_for_inference(self):
        self.assertNotIn("naive_bayes", MODEL_FILES)

    def test_inference_accepts_image_without_a_face(self):
        image = np.zeros((96, 160, 3), dtype=np.uint8)
        success, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(success)

        result = analyze_image(encoded.tobytes(), estimator=FixedEstimator())
        self.assertTrue(result["isFake"])
        self.assertIs(result["isFake"], True)
        self.assertEqual(result["fake_pct"], 90.0)

    def test_inference_abstains_when_model_probability_is_ambiguous(self):
        image = np.zeros((96, 160, 3), dtype=np.uint8)
        success, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(success)

        result = analyze_image(encoded.tobytes(), estimator=LowConfidenceEstimator())
        self.assertTrue(result["isUncertain"])
        self.assertIsNone(result["isFake"])
        self.assertEqual(result["verdict"], "Uncertain")

    def test_inference_reports_confident_real_verdict(self):
        image = np.zeros((96, 160, 3), dtype=np.uint8)
        success, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(success)

        result = analyze_image(encoded.tobytes(), estimator=ConfidentRealEstimator())
        self.assertFalse(result["isUncertain"])
        self.assertFalse(result["isFake"])
        self.assertIs(result["isFake"], False)
        self.assertEqual(result["verdict"], "Real (Authentic)")

    def test_training_uses_full_images_without_face_detection(self):
        image = np.zeros((96, 160, 3), dtype=np.uint8)
        success, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(success)

        with tempfile.TemporaryDirectory() as temporary_directory:
            data_directory = Path(temporary_directory)
            for class_name in ("real", "fake"):
                class_directory = data_directory / class_name
                class_directory.mkdir()
                (class_directory / "sample.jpg").write_bytes(encoded.tobytes())

            features, labels = load_dataset(data_directory)

        self.assertEqual(features.shape[0], 2)
        np.testing.assert_array_equal(labels, np.array([0, 1]))

    def test_prepare_image_preserves_the_whole_image_aspect_ratio(self):
        image = np.zeros((96, 160, 3), dtype=np.uint8)
        image[:, :40] = 255

        prepared = prepare_image(image)

        self.assertEqual(prepared.shape, (256, 256, 3))
        self.assertGreater(prepared[:, :20].mean(), prepared[:, -20:].mean())


if __name__ == "__main__":
    unittest.main()
