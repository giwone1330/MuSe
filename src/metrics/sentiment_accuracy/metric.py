import re

import datasets
import evaluate


class SentimentAccuracy(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="Compute sentiment classification accuracy from generated text.",
            inputs_description="Predictions and references as strings.",
            citation="",
            homepage="",
            codebase_urls=[],
            reference_urls=[],
            features=datasets.features.features.Features(
                {
                    "predictions": datasets.features.features.Value("string", id="sequence"),
                    "references": datasets.features.features.Value("string", id="sequence"),
                }
            ),
        )

    def _extract_label(self, text):
        lowered = str(text).lower()

        # Prefer content generated after the sentiment cue to avoid matching words
        # that appear inside the review text itself.
        if "sentiment:" in lowered:
            lowered = lowered.rsplit("sentiment:", 1)[1]

        match = re.search(r"\b(positive|negative)\b", lowered)
        if not match:
            return None
        return match.group(1)

    def _compute(self, predictions, references):
        total = len(predictions)
        valid = 0
        correct = 0
        invalid = 0

        for pred, ref in zip(predictions, references):
            pred_label = self._extract_label(pred)
            ref_label = self._extract_label(ref)

            if pred_label is None or ref_label is None:
                invalid += 1
                continue

            valid += 1
            if pred_label == ref_label:
                correct += 1

        return {
            "accuracy": correct / valid if valid > 0 else 0.0,
            "correct": correct,
            "valid": valid,
            "invalid": invalid,
            "total": total,
            "coverage": valid / total if total > 0 else 0.0,
        }
