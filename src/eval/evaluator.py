import wandb
import torch
import hydra
import evaluate
import math
from torch.utils.data import DataLoader


from src.utils.utils import exist_and_not_none

class CustomEvaluator:
    def __init__(self, cfg, model, tokenizer, datasets, splits=None):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.datasets = datasets #{"name(split)":dataset}
        self.splits = splits

        if cfg.state.device == "cuda":
            self.model.to(cfg.state.device) # the data will follow model's device

        self.metrics = None
        if exist_and_not_none(self.cfg, "evaluator", "non_inst", "metrics"):
            metric_dict={}
            for metric_name, metric in self.cfg.evaluator.non_inst.metrics.items():
                metric_obj = hydra.utils.instantiate(metric, tokenizer=self.tokenizer, _convert_="partial")
                metric_dict[metric_name]=metric_obj
            # combine
            self.metrics = evaluate.combine(metric_dict)



    def evaluate_dataset(self, dataset, dataset_name):

        # use best model or last model
        # if use_best_model:
        #     self.get_best_model()
        # else: # use last model
        #     self.get_last_model()

        self.model.eval()
        all_predictions = []
        all_references = []

        default_max_length = 1024
        if exist_and_not_none(self.cfg, "dataset", "non_inst", "pre_processor", "tokenize_args", "max_length"):
            default_max_length = self.cfg.dataset.non_inst.pre_processor.tokenize_args.max_length
        elif exist_and_not_none(self.cfg, "trainer", "args", "max_length"):
            default_max_length = self.cfg.trainer.args.max_length

        configured_max_length = self.cfg.evaluator.non_inst.get("max_length", default_max_length)
        max_new_tokens = self.cfg.evaluator.non_inst.generate_args.get("max_new_tokens", 0)
        model_max_positions = getattr(self.model.config, "max_position_embeddings", None)
        if model_max_positions is not None:
            prompt_max_length = min(configured_max_length, model_max_positions - max_new_tokens)
        else:
            prompt_max_length = configured_max_length

        
        # map pre_processing to dataset
        # processed_dataset = dataset.map(lambda x: self.preprocess_dataset(x), batched=False)
        def preprocess_function(examples):
            prompts = examples["prompt"]
            model_inputs = self.tokenizer(
                prompts,
                max_length=prompt_max_length,
                truncation=True,
            )
            
            return model_inputs

        processed_dataset = dataset.map(preprocess_function, batched=True)
        processed_dataset.set_format("torch")

        for i, data in enumerate(processed_dataset):
            with torch.no_grad():
                input_ids = data["input_ids"].unsqueeze(0).to(self.model.device)
                generated_tokens = self.model.generate(
                    input_ids,
                    **self.cfg.evaluator.non_inst.generate_args,
                )
                output_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

                all_predictions.append(output_text)
                all_references.append(data["completion"])

                if self.cfg.env.verbose:
                    print(f"output: {output_text}, answer: {data['completion']}")
        
        metric_results = {}
        if self.metrics is not None:
            metric_results = self.metrics.compute(predictions=all_predictions, references=all_references)

            print(metric_results)

            if exist_and_not_none(self.cfg.env, "tracker", "wandb"):
                # put prefix dataset name to metric keys
                logged_metric_results = {f"{dataset_name}_{k}": v for k, v in metric_results.items()}
                wandb.log(logged_metric_results)

        return metric_results

    def evaluate(self):
        final_metric = {}

        if self.splits is None:
            datasets_to_eval = self.datasets
        else:
            requested_splits = [self.splits] if isinstance(self.splits, str) else list(self.splits)
            datasets_to_eval = {name: self.datasets[name] for name in requested_splits if name in self.datasets}

            missing_splits = [name for name in requested_splits if name not in self.datasets]
            if len(missing_splits) > 0:
                print(f"Warning: requested evaluator splits not found: {missing_splits}")

        for dataset_name, dataset in datasets_to_eval.items():
            print(f"Evaluating on dataset: {dataset_name}")
            metric_results = self.evaluate_dataset(dataset, dataset_name)
            final_metric[dataset_name] = metric_results

        print("Evaluation complete.")


class CustomEvaluatorBatched:
    def __init__(self, cfg, model, tokenizer, datasets, splits=None):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.datasets = datasets  # {"name(split)": dataset}
        self.splits = splits

        self.metrics = None
        if exist_and_not_none(self.cfg, "evaluator", "non_inst", "metrics"):
            metric_dict = {}
            for metric_name, metric in self.cfg.evaluator.non_inst.metrics.items():
                metric_obj = hydra.utils.instantiate(metric, tokenizer=self.tokenizer, _convert_="partial")
                metric_dict[metric_name] = metric_obj
            # combine
            self.metrics = evaluate.combine(metric_dict)

    def collate_fn(self, batch):
        """Custom collate function to handle batching with padding."""
        # Extract prompts and completions
        prompts = [item['prompt'] for item in batch]
        completions = [item['completion'] for item in batch]
        
        # Tokenize prompts with padding
        tokenized = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'completions': completions
        }

    def evaluate_dataset(self, dataset, dataset_name):
        self.model.eval()
        all_predictions = []
        all_references = []
        
        # Get batch size from config
        batch_size = self.cfg.evaluator.non_inst.get('batch_size', 8)
        
        # Create DataLoader with custom collate function
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        
        for batch_idx, batch in enumerate(dataloader):
            with torch.no_grad():
                # Move batch to device
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                
                # Batched generation
                generated_tokens = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    **self.cfg.evaluator.non_inst.generate_args
                )
                
                # Decode all outputs in the batch
                output_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                
                all_predictions.extend(output_texts)
                all_references.extend(batch['completions'])
                
                if self.cfg.env.verbose:
                    for output, answer in zip(output_texts, batch['completions']):
                        print(f"output: {output}, answer: {answer}")
        
        metric_results = {}
        if self.metrics is not None:
            metric_results = self.metrics.compute(predictions=all_predictions, references=all_references)

            print(metric_results)

            if exist_and_not_none(self.cfg.env, "tracker", "wandb"):
                # put prefix dataset name to metric keys
                logged_metric_results = {f"{dataset_name}_{k}": v for k, v in metric_results.items()}
                wandb.log(logged_metric_results)

        return metric_results

    def evaluate(self):
        final_metric = {}

        if self.splits is None:
            datasets_to_eval = self.datasets
        else:
            requested_splits = [self.splits] if isinstance(self.splits, str) else list(self.splits)
            datasets_to_eval = {name: self.datasets[name] for name in requested_splits if name in self.datasets}

            missing_splits = [name for name in requested_splits if name not in self.datasets]
            if len(missing_splits) > 0:
                print(f"Warning: requested evaluator splits not found: {missing_splits}")

        for dataset_name, dataset in datasets_to_eval.items():
            print(f"Evaluating on dataset: {dataset_name}")
            metric_results = self.evaluate_dataset(dataset, dataset_name)
            final_metric[dataset_name] = metric_results

        print("Evaluation complete.")


class CustomEvaluatorNextToken:
    """
    Evaluator that uses next token prediction instead of autoregressive generation.
    Much faster as it only requires a single forward pass through the model.
    """
    def __init__(self, cfg, model, tokenizer, datasets, splits=None):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.datasets = datasets  # {"name(split)": dataset}
        self.splits = splits

        self.metrics = None
        if exist_and_not_none(self.cfg, "evaluator", "non_inst", "metrics"):
            metric_dict = {}
            for metric_name, metric in self.cfg.evaluator.non_inst.metrics.items():
                metric_obj = hydra.utils.instantiate(metric, tokenizer=self.tokenizer, _convert_="partial")
                metric_dict[metric_name] = metric_obj
            # combine
            self.metrics = evaluate.combine(metric_dict)

    def collate_fn(self, batch):
        """Custom collate function to handle batching with padding."""
        # Extract prompts and completions
        prompts = [item['prompt'] for item in batch]
        completions = [item['completion'] for item in batch]
        
        # Combine prompt and completion for full sequence
        full_sequences = [f"{prompt}{completion}" for prompt, completion in zip(prompts, completions)]
        
        # Tokenize full sequences with padding
        tokenized_full = self.tokenizer(
            full_sequences,
            padding=True,
            return_tensors="pt"
        )
        
        # Tokenize prompts to know where completion starts
        tokenized_prompts = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': tokenized_full['input_ids'],
            'attention_mask': tokenized_full['attention_mask'],
            'prompt_lengths': [len(ids) for ids in tokenized_prompts['input_ids']],
            'completions': completions,
            'prompts': prompts
        }

    def evaluate_dataset(self, dataset, dataset_name):
        self.model.eval()
        all_predictions = []
        all_references = []
        
        # Get batch size from config
        batch_size = self.cfg.evaluator.non_inst.get('batch_size', 8)
        
        # Create DataLoader with custom collate function
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        
        for batch_idx, batch in enumerate(dataloader):
            with torch.no_grad():
                # Move batch to device
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                
                # Forward pass to get logits
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                
                # Get predictions for each sample in the batch
                for i, prompt_length in enumerate(batch['prompt_lengths']):
                    # Get the completion part of the sequence
                    # Logits are shifted by 1 (predict next token)
                    # So logits[prompt_length-1] predicts token at position prompt_length
                    completion_logits = logits[i, prompt_length-1:-1]  # Exclude last logit (no next token to predict)
                    
                    # Get predicted token IDs (greedy decoding)
                    predicted_ids = torch.argmax(completion_logits, dim=-1)
                    
                    # Decode the predicted tokens
                    predicted_text = self.tokenizer.decode(predicted_ids, skip_special_tokens=True)
                    
                    all_predictions.append(predicted_text)
                    all_references.append(batch['completions'][i])
                    
                    if self.cfg.env.verbose:
                        print(f"prompt: {batch['prompts'][i]}")
                        print(f"output: {predicted_text}, answer: {batch['completions'][i]}")
        
        metric_results = {}
        if self.metrics is not None:
            metric_results = self.metrics.compute(predictions=all_predictions, references=all_references)

            print(metric_results)

            if exist_and_not_none(self.cfg.env, "tracker", "wandb"):
                # put prefix dataset name to metric keys
                logged_metric_results = {f"{dataset_name}_{k}": v for k, v in metric_results.items()}
                wandb.log(logged_metric_results)

        return metric_results

    def evaluate(self):
        final_metric = {}

        if self.splits is None:
            datasets_to_eval = self.datasets
        else:
            requested_splits = [self.splits] if isinstance(self.splits, str) else list(self.splits)
            datasets_to_eval = {name: self.datasets[name] for name in requested_splits if name in self.datasets}

            missing_splits = [name for name in requested_splits if name not in self.datasets]
            if len(missing_splits) > 0:
                print(f"Warning: requested evaluator splits not found: {missing_splits}")

        for dataset_name, dataset in datasets_to_eval.items():
            print(f"Evaluating on dataset: {dataset_name}")
            metric_results = self.evaluate_dataset(dataset, dataset_name)
            final_metric[dataset_name] = metric_results

        print("Evaluation complete.")


class CustomEvaluatorPerplexity:
    """
    Evaluator for language modeling quality using forward-pass loss and perplexity.
    Interface is compatible with CustomEvaluator.
    """

    def __init__(self, cfg, model, tokenizer, datasets, splits=None):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.splits = splits

        if cfg.state.device == "cuda":
            self.model.to(cfg.state.device)

    def evaluate_dataset(self, dataset, dataset_name):
        self.model.eval()

        default_max_length = 1024
        if exist_and_not_none(self.cfg, "dataset", "non_inst", "pre_processor", "tokenize_args", "max_length"):
            default_max_length = self.cfg.dataset.non_inst.pre_processor.tokenize_args.max_length
        elif exist_and_not_none(self.cfg, "trainer", "args", "max_length"):
            default_max_length = self.cfg.trainer.args.max_length

        configured_max_length = self.cfg.evaluator.non_inst.get("max_length", default_max_length)
        model_max_positions = getattr(self.model.config, "max_position_embeddings", None)
        if model_max_positions is not None:
            max_length = min(configured_max_length, model_max_positions)
        else:
            max_length = configured_max_length

        total_nll = 0.0
        total_tokens = 0
        sample_count = 0

        for data in dataset:
            with torch.no_grad():
                if "input_ids" in data:
                    input_ids = data["input_ids"]
                    if not torch.is_tensor(input_ids):
                        input_ids = torch.tensor(input_ids, dtype=torch.long)
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)
                    input_ids = input_ids[:, :max_length].to(self.model.device)
                else:
                    if "prompt" in data and "completion" in data:
                        text = f"{data['prompt']}{data['completion']}"
                    elif "text" in data:
                        text = data["text"]
                    else:
                        continue

                    enc = self.tokenizer(
                        text,
                        max_length=max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    input_ids = enc["input_ids"].to(self.model.device)

                if input_ids.size(1) < 2:
                    continue

                outputs = self.model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss

                token_count = input_ids.size(1) - 1
                total_nll += float(loss.item()) * token_count
                total_tokens += token_count
                sample_count += 1

        if total_tokens == 0:
            metric_results = {
                "loss": float("nan"),
                "perplexity": float("nan"),
                "tokens": 0,
                "samples": sample_count,
            }
        else:
            avg_nll = total_nll / total_tokens
            metric_results = {
                "loss": avg_nll,
                "perplexity": math.exp(avg_nll),
                "tokens": total_tokens,
                "samples": sample_count,
            }

        print(metric_results)

        if exist_and_not_none(self.cfg.env, "tracker", "wandb"):
            logged_metric_results = {f"{dataset_name}_{k}": v for k, v in metric_results.items()}
            wandb.log(logged_metric_results)

        return metric_results

    def evaluate(self):
        final_metric = {}

        if self.splits is None:
            datasets_to_eval = self.datasets
        else:
            requested_splits = [self.splits] if isinstance(self.splits, str) else list(self.splits)
            datasets_to_eval = {name: self.datasets[name] for name in requested_splits if name in self.datasets}

            missing_splits = [name for name in requested_splits if name not in self.datasets]
            if len(missing_splits) > 0:
                print(f"Warning: requested evaluator splits not found: {missing_splits}")

        for dataset_name, dataset in datasets_to_eval.items():
            print(f"Evaluating on dataset: {dataset_name}")
            metric_results = self.evaluate_dataset(dataset, dataset_name)
            final_metric[dataset_name] = metric_results

        print("Evaluation complete.")



