import re
import datasets
import evaluate


class ClassificationAccuracy(evaluate.Metric):
    """
    Generic classification accuracy metric that extracts labels from generated text
    using regex patterns.
    """

    def _info(self):
        return evaluate.MetricInfo(
            description="Compute classification accuracy from generated text predictions.",
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
        """
        Extract a label from text by finding common classification patterns.
        """
        lowered = str(text).lower().strip()

        # Common patterns for different tasks
        patterns = [
            # Sentiment (positive/negative)
            r"\b(positive|negative|yes|no|true|false)\b",
            # Topic (world, sports, business, sci/tech)
            r"\b(world|sports|business|sci/tech|scitec|tech)\b",
            # NLI (entailment, neutral, contradiction)
            r"\b(entailment|neutral|contradiction)\b",
            # Acceptability
            r"\b(acceptable|unacceptable|accept|unaccept)\b",
            # Solution choice
            r"\b(solution\s*[12]|sol[12]|first|second)\b",
            # Generic label extraction
            r":\s*([\w\s/]+)(?:\s|$|\.)",
        ]

        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                label = match.group(1).strip().lower()
                # Normalize common variations
                if label in ["scitec", "sci/tech", "sci-tech"]:
                    return "sci/tech"
                if label in ["sol1", "solution 1", "first"]:
                    return "solution1"
                if label in ["sol2", "solution 2", "second"]:
                    return "solution2"
                if label in ["unaccept", "unacceptable"]:
                    return "unacceptable"
                return label

        return None

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
