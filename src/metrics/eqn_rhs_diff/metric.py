import evaluate
import decimal
import datasets
import numpy as np

class EqnRHSdiff(evaluate.Metric):
    def __init__(self, **kwargs):
        # This is a robust way to handle arguments. We manually pop the arguments
        # needed for this specific metric ('tokenizer', 'equals_token') and pass
        # all remaining arguments (like data_dir, config_name, etc.) to the parent.
        # This prevents the AttributeError by ensuring the parent class receives
        # all the arguments it expects from the evaluate.load() function.
        if 'tokenizer' not in kwargs:
            raise ValueError("A 'tokenizer' instance must be provided when loading this metric.")
        
        self.tokenizer = kwargs.pop('tokenizer')
        self.equals_token = kwargs.pop('equals_token', '=') # Pop with a default value

        super().__init__(**kwargs)


    def _info(self):
        return evaluate.MetricInfo(
            description="Parse and evaluate the difference of the RHS of the equation",
            inputs_description="Predictions and references",
            citation="",
            homepage="",
            codebase_urls=[],
            reference_urls=[],
            features=datasets.features.features.Features({
                "predictions": datasets.features.features.Value("string", id="sequence"),
                "references": datasets.features.features.Value("string", id="sequence"),
            }),
        )
    
    def parse_rhs(self, token_str):
        # parse everything between = and [eos]
        if self.equals_token in token_str:
            equals_index = token_str.index(self.equals_token) # first occurence
        else:
            equals_index = -1
        
        if self.tokenizer.eos_token in token_str:
            eos_index = token_str.index(self.tokenizer.eos_token) # first occurence
        else:
            eos_index = len(token_str)
        
        rhs_str = token_str[equals_index+1:eos_index]
        if isinstance(rhs_str, list):
            rhs_str = "".join(rhs_str).strip()
        return rhs_str


    def _compute(self, predictions, references):
        diffs = []
        errors = 0
        correct =0
        for p, r in zip(predictions, references):
            p_tokens = p.split(" ")
            r_tokens = r.split(" ")
            p_rhs = self.parse_rhs(p_tokens)
            r_rhs = self.parse_rhs(r_tokens)

            try:
                p_numeric = decimal.Decimal(p_rhs)
                r_numeric = decimal.Decimal(r_rhs)
                abs_diff = abs(p_numeric-r_numeric)
                diffs.append(abs_diff)
                if p_numeric == r_numeric:
                    correct += 1
            except:
                errors += 1

        total_predictions = len(predictions)
        numeric_count = len(diffs)

        return {
            "correct/totalpred": correct / total_predictions if total_predictions > 0 else 0.0,
            "correct/numeric": correct / numeric_count if numeric_count > 0 else 0.0,
            "non_numeric/totalpred": errors / total_predictions if total_predictions > 0 else 0.0,
            "avg_abs_diff": float(np.mean(diffs)) if numeric_count > 0 else 0.0,
        }
    