import evaluate
import decimal

class EqnRHSdiff(evaluate.Metric):
    def __init__(self, tokenizer, equals_token, **kwargs):
        self.equals_token = equals_token
        self.tokenizer = tokenizer


    def _info(self):
        return evaluate.MetricInfo(
            description="Parse and evaluate the difference of the RHS of the equation",
            inputs_description="Predictions and references",
            features=None,
            citation="",
            homepage="",
            codebase_urls=[],
            reference_urls=[],
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
        return rhs_str


    def _compute(self, predictions, references):
        diffs = []
        errors = 0
        correct =0
        for p, r in zip(predictions, references):
            p_rhs = self.parse_rhs(p)
            r_rhs = self.parse_rhs(r)

            try:
                p_numeric = decimal.Decimal(p_rhs)
                r_numeric = decimal.Decimal(r_rhs)
                abs_diff = abs(p_numeric-r_numeric)
                diffs.append(abs_diff)
                if p_numeric == r_numeric:
                    correct += 1
            except:
                errors += 1
        return {
            "evaluated_accuracy": correct / (len(diffs)+ errors),
            "evaluated_accuracy_for_numeric": correct / len(diffs),
            "non_numeric": errors,
            "avg_abs_diff": sum(diffs) / len(diffs),
            }
    


class MyMetricTemplate(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="Parse and evaluate the difference of the RHS of the equation",
            inputs_description="Predictions and references",
            features=None,
            citation="",
            homepage="",
            codebase_urls=[],
            reference_urls=[],
        )

    def _compute(self, predictions, references):
        errors = [abs(p - r) for p, r in zip(predictions, references)]
        return {"mae": sum(errors) / len(errors)}
    
if __name__ == "__main__":
    metric = EqnRHSdiff()
