import math
import os
import random
from datasets import Dataset, DatasetDict, Features, Sequence, Value


def _seeded_dataset_split(dataset, test_size, seed=42):
    """
    Split a Dataset into train/validation using seeded index sampling.

    Sampling is done without replacement so the validation size is exact and
    deterministic for a given seed.
    """
    dataset_len = len(dataset)
    if dataset_len == 0:
        empty = dataset.select([])
        return {"train": dataset, "test": empty}

    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("Float test_size must be between 0 and 1.")
        val_size = math.ceil(dataset_len * test_size)
    else:
        val_size = int(test_size)

    if dataset_len < 2:
        empty = dataset.select([])
        return {"train": dataset, "test": empty}

    val_size = max(1, min(val_size, dataset_len - 1))

    rng = random.Random(seed)
    val_indices = sorted(rng.sample(range(dataset_len), val_size))
    val_index_set = set(val_indices)
    train_indices = [idx for idx in range(dataset_len) if idx not in val_index_set]

    return {
        "train": dataset.select(train_indices),
        "test": dataset.select(val_indices),
    }


class formatter():
    def __init__(self, method_name, masker_args=None):
        self.method_name = method_name
        self.masker_args = masker_args
        # masker_args include formatting stuff like
        # zeropad, reverse, indexhint

    def format(self, *args, **kwargs):
        if not hasattr(self, self.method_name):
            raise ValueError(f"No such method: {self.method_name}")
        method = getattr(self, self.method_name)
        if not callable(method):
            raise ValueError(f"Attribute {self.method_name} is not callable.")

        return method(*args, **kwargs)
        
    def format_simple_text(self, row_dict):
        # row_dict is a row of the raw dataset
        return row_dict["text"]
    
    def format_ga(self, row_dict):
        # row_dict is a row of the raw dataset
        # return f"Q: {row_dict['question']}\nA: {row_dict['answer']}"
        return f"Q: {row_dict['question']}\nA: {row_dict['answer']}"
    
    def format_equations(self, row_dict,):
        # Separate operands and operators
        operands = {int(k.split("_")[-1]): v for k, v in row_dict.items() if k.startswith("operand")}
        operators = {int(k.split("_")[-1]): v for k, v in row_dict.items() if k.startswith("operator")}

        # sanity check : num(operands) = num(operators) + 1
        if len(operands) != len(operators) + 1:
            raise ValueError(
                f"Invalid expression: got {len(operands)} operands and {len(operators)} operators"
            )

        # Sort by index
        sorted_operands = [v for _, v in sorted(operands.items())]
        sorted_operators = [v for _, v in sorted(operators.items())]

        # Build expression as a string
        expression_parts = []
        for i, operand in enumerate(sorted_operands):
            expression_parts.append(str(operand))
            if i < len(sorted_operators):
                expression_parts.append(sorted_operators[i])
        
        # add ={answer}
        expression_parts.append("=")
        expression_parts.append(row_dict["answer"])

        expression = "".join(expression_parts)
        return expression


    def format_equations_for_prompt_completion(self, row_dict,):
        out_dict = {}

        # Separate operands and operators
        operands = {int(k.split("_")[-1]): v for k, v in row_dict.items() if k.startswith("operand")}
        operators = {int(k.split("_")[-1]): v for k, v in row_dict.items() if k.startswith("operator")}

        # sanity check : num(operands) = num(operators) + 1
        if len(operands) != len(operators) + 1:
            raise ValueError(
                f"Invalid expression: got {len(operands)} operands and {len(operators)} operators"
            )

        # Sort by index
        sorted_operands = [v for _, v in sorted(operands.items())]
        sorted_operators = [v for _, v in sorted(operators.items())]

        # Build expression as a string
        expression_parts = []
        for i, operand in enumerate(sorted_operands):
            expression_parts.append(str(operand))
            if i < len(sorted_operators):
                expression_parts.append(sorted_operators[i])
        
        expression_parts.append("=")
        expression = "".join(expression_parts)

        out_dict["prompt"] = expression
        out_dict["completion"] = str(row_dict["answer"])
        
        return out_dict
    
    
    def format_imdb_for_prompt_completion(self, row_dict):
        out_dict = {}
        label = "positive" if row_dict["label"] == 1 else "negative"
        out_dict["prompt"] = f"Review: {row_dict['text']}\nSentiment:"   # no trailing space
        out_dict["completion"] = f" {label}"                              # space moves here
        return out_dict

    def format_ag_news_for_prompt_completion(self, row_dict):
        out_dict = {}
        # AG News labels: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
        label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        label = label_map.get(row_dict["label"], "Unknown")
        out_dict["prompt"] = f"Article: {row_dict['text']}\nCategory:"
        out_dict["completion"] = f" {label}"
        return out_dict

    def format_snli_for_prompt_completion(self, row_dict):
        out_dict = {}
        # SNLI labels: 0=entailment, 1=neutral, 2=contradiction, -1=no label
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        label = label_map.get(row_dict["label"], "unknown")
        out_dict["prompt"] = f"Premise: {row_dict['premise']}\nHypothesis: {row_dict['hypothesis']}\nLabel:"
        out_dict["completion"] = f" {label}"
        return out_dict

    def format_boolq_for_prompt_completion(self, row_dict):
        out_dict = {}
        label = "yes" if row_dict["answer"] else "no"
        out_dict["prompt"] = f"Passage: {row_dict['passage']}\nQuestion: {row_dict['question']}\nAnswer:"
        out_dict["completion"] = f" {label}"
        return out_dict

    def format_piqa_for_prompt_completion(self, row_dict):
        out_dict = {}
        # PIQA label: 0 = sol1 is better, 1 = sol2 is better
        solution = row_dict["sol1"] if row_dict["label"] == 0 else row_dict["sol2"]
        out_dict["prompt"] = f"Goal: {row_dict['goal']}\nSolution:"
        out_dict["completion"] = f" {solution}"
        return out_dict

    def format_cola_for_prompt_completion(self, row_dict):
        out_dict = {}
        label = "acceptable" if row_dict["label"] == 1 else "unacceptable"
        out_dict["prompt"] = f"Sentence: {row_dict['sentence']}\nLabel:"
        out_dict["completion"] = f" {label}"
        return out_dict

    def format_conll2003_for_prompt_completion(self, row_dict):
        out_dict = {}
        # CoNLL-2003 NER: join tokens then predict NER tags
        tokens = row_dict["tokens"]
        ner_tags = row_dict["ner_tags"]
        # Create a simple mapping from tag IDs to labels (simplified for demo)
        ner_label_map = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"}
        ner_label = " ".join([ner_label_map.get(tag, "O") for tag in ner_tags])
        tokens_str = " ".join(tokens)
        out_dict["prompt"] = f"Tokens: {tokens_str}\nNER:"
        out_dict["completion"] = f" {ner_label}"
        return out_dict

    def format_wikitext_for_prompt_completion(self, row_dict):
        out_dict = {}
        text = row_dict["text"].strip()
        if len(text) < 20:  # Skip very short lines
            out_dict["prompt"] = text
            out_dict["completion"] = ""
        else:
            # Split text roughly in half for prompt/completion
            mid = len(text) // 2
            out_dict["prompt"] = text[:mid]
            out_dict["completion"] = text[mid:]
        return out_dict

class masker():
    def __init__(self, method_name, masker_args=None):
        self.method_name = method_name
        self.masker_args = masker_args
        # masker_args include formatting stuff like
        # zeropad, reverse, indexhint

    def mask(self, *args, **kwargs):
        if not hasattr(self, self.method_name):
            raise ValueError(f"No such method: {self.method_name}")
        method = getattr(self, self.method_name)
        if not callable(method):
            raise ValueError(f"Attribute {self.method_name} is not callable.")

        return method(*args, **kwargs)
        
    def mask_simple_text(self, row_dict):
        # row_dict is a row of the raw dataset
        return row_dict["text"]
    
    def mask_equations(self, row, token_split, pad_token):
        # index of a substring having element
        input_ids = row['input_ids']
        indices = [i for i, w in enumerate(token_split) if self.masker_args['substring'] in w]
        if len(indices)==1:
            idx = indices[0]
            # mask all the tokens before it
            # masked_tokens = [
            #     -100 if i <= idx else t
            #     for i, t in enumerate(input_ids)
            # ]
            masked_tokens = [pad_token] * (idx + 1) + input_ids[idx + 1:]
        else:
            raise ValueError(f"Multiple / No tokens with substring found : {len(indices)}")

        return masked_tokens

class causal_lm_pre_processor():
    def __init__(self, tokenizer, map_args, formatter, masker=None, tokenize_args=None, manipulate_args=None, ):
        self.tokenizer=tokenizer
        self.tokenize_args = tokenize_args
        self.map_args = map_args
        self.manipulate_args = manipulate_args
        self.formatter = formatter
        self.masker = masker

        if str(self.map_args["num_proc"]).lower() == "max":
            self.map_args["num_proc"] = os.cpu_count()

        if self.tokenize_args == None:
            self.tokenize = False
    
    def tokenize(self, data_entry):
        args = [data_entry[column] for column in self.columns]
        return self.tokenizer(*args, **self.tokenize_args)
    
    def tokenize_batch(self, data_entry_batch):
        # DL = "dictionary of list"
        # LD = "list of dictionary"
        # # dl to ld
        # LD = [dict(zip(DL,t)) for t in zip(*DL.values())]
        # # ld to dl
        # DL = {k: [dic[k] for dic in LD] for k in LD[0]}

        if self.tokenize:
            x_texts = [] # list of input strings
            token_splits = []
            # Build expressions + target answer strings
            for values in zip(*data_entry_batch.values()):
                row = dict(zip(data_entry_batch.keys(), values))
                x_text = self.formatter.format(row) # x_text = one prompt
                x_texts.append(x_text)
                token_splits.append(self.tokenizer.tokenize(x_text, **self.tokenize_args))

            x_enc = self.tokenizer(x_texts, **self.tokenize_args)
            # x_enc = {"test" : ['sentence1', 'sentence2...]}
            
            if self.masker != None:
                masked_idx = []
                for i, values in enumerate(zip(*x_enc.values())):
                    row = dict(zip(x_enc.keys(), values))
                    masked = self.masker.mask(row, token_splits[i], self.tokenizer.pad_token_id)
                    masked_idx.append(masked)
            
                x_enc['labels'] = masked_idx

            return x_enc
        else: # no tokenization
            x_texts = [] # list of formatted docs
            token_splits = []
            # Build expressions + target answer strings
            for values in zip(*data_entry_batch.values()):
                row = dict(zip(data_entry_batch.keys(), values))
                x_text = self.formatter.format(row) # x_text = one prompt
                x_texts.append(x_text)

            x_formatted = {k: [dic[k] for dic in x_texts] for k in x_texts[0]}

            return x_formatted
    
        
    # def manipulate_dataset(self,tokenized_dataset):
    #     # dataset column operations
    #     processed_dataset = tokenized_dataset.remove_columns(
    #         [col for col in tokenized_dataset.column_names if col not in self.manipulate_args]
    #     )

    #     # Step 2: rename each column
    #     for old, new in self.manipulate_args.items():
    #         if old != new:
    #             processed_dataset = processed_dataset.rename_column(old, new)

    #     return processed_dataset

    def manipulate_dataset(self, tokenized_dataset):
        """
        Select and rename columns in a Dataset or DatasetDict.
        Uses self.manipulate_args = {old_col: new_col}.
        """

        def process_split(ds):
            # Step 1: keep only selected columns
            ds = ds.remove_columns(
                [col for col in ds.column_names if col not in self.manipulate_args]
            )
            # Step 2: rename columns
            for old, new in self.manipulate_args.items():
                if old != new:
                    ds = ds.rename_column(old, new)
            return ds

        # Handle DatasetDict (multiple splits)
        if isinstance(tokenized_dataset, DatasetDict):
            return DatasetDict({split: process_split(ds) for split, ds in tokenized_dataset.items()})

        # Handle single Dataset
        elif isinstance(tokenized_dataset, Dataset):
            return process_split(tokenized_dataset)

        else:
            raise TypeError(f"Unsupported type: {type(tokenized_dataset)}")


    def process(self,raw_dataset):

        if self.map_args["batched"] == True:
            tokenized_dataset = raw_dataset.map(self.tokenize_batch, **self.map_args)
        else:
            tokenized_dataset = raw_dataset.map(self.tokenize, **self.map_args)
        if self.manipulate_args is not None:
            processed_dataset = self.manipulate_dataset(tokenized_dataset)
        else:
            processed_dataset = tokenized_dataset
        return processed_dataset


class packed_causal_lm_pre_processor():
    """
    Packed training preprocessor: tokenizes, concatenates, and chunks into fixed-size blocks.
    Optionally splits train into train/val via train2val ratio.
    
    Args:
        train2val: Validation ratio (e.g., 0.1 = 10% val, 90% train). 
                   If None, uses original validation split from dataset.
    """
    def __init__(self, tokenizer, formatter, block_size=512, train2val=None, map_args=None):
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.block_size = block_size
        self.train2val = train2val  # validation ratio (e.g., 0.1 = 10% for val, 90% for train)
        self.map_args = map_args or {}

        if str(self.map_args.get("num_proc", "")).lower() == "max":
            self.map_args["num_proc"] = os.cpu_count()

    def _format_and_tokenize_batch(self, batch):
        """Format texts and tokenize without padding/truncation."""
        texts = []
        for values in zip(*batch.values()):
            row = dict(zip(batch.keys(), values))
            texts.append(self.formatter.format(row))

        enc = self.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=False
        )
        return {"input_ids": enc["input_ids"]}

    def _group_texts(self, batch):
        """Concatenate and pack tokens into fixed-size blocks."""
        if len(batch["input_ids"]) == 0:
            return {"input_ids": [], "labels": []}

        # Concatenate all token sequences in this batch
        concatenated_ids = sum(batch["input_ids"], [])

        # Compute how many complete blocks we can make
        total_len = len(concatenated_ids)
        total_len = (total_len // self.block_size) * self.block_size

        if total_len == 0:
            return {"input_ids": [], "labels": []}

        # Slice into fixed-size blocks
        blocks = [
            concatenated_ids[i : i + self.block_size]
            for i in range(0, total_len, self.block_size)
        ]
        
        return {
            "input_ids": blocks,
            "labels": blocks  # For causal LM, labels = input_ids
        }

    def process(self, raw_dataset):
        """
        Process dataset:
        1. Format and tokenize texts
        2. Concatenate and pack into blocks
        3. Optionally create train/val split from train
        4. Flatten blocks into individual rows
        """
        # Get column names (handle both Dataset and DatasetDict)
        if isinstance(raw_dataset, DatasetDict):
            column_names = raw_dataset["train"].column_names
        else:
            column_names = raw_dataset.column_names

        # Step 1: Format and tokenize
        map_kwargs = dict(self.map_args)
        map_kwargs.pop("batched", None)
        map_kwargs.pop("batch_size", None)

        tokenized_dataset = raw_dataset.map(
            self._format_and_tokenize_batch,
            batched=True,
            remove_columns=column_names,
            **map_kwargs
        )

        # Step 2: Pack into blocks
        packed_dataset = tokenized_dataset.map(
            self._group_texts,
            batched=True,
            batch_size=self.map_args.get("batch_size", 1000),
            **map_kwargs
        )

        # Step 3: Flatten blocks into individual rows
        def flatten_blocks(batch):
            """Convert list of blocks into individual row examples."""
            flat_input_ids = []
            flat_labels = []
            for input_ids_block, labels_block in zip(batch["input_ids"], batch["labels"]):
                if isinstance(input_ids_block, list) and len(input_ids_block) == self.block_size:
                    flat_input_ids.append(input_ids_block)
                    flat_labels.append(labels_block)
            
            return {"input_ids": flat_input_ids, "labels": flat_labels}

        flat_dataset = packed_dataset.map(
            flatten_blocks,
            batched=True,
            batch_size=self.map_args.get("batch_size", 1000),
            **map_kwargs
        )

        # Step 4: Apply train/val split if train2val is specified
        if isinstance(flat_dataset, DatasetDict):
            result = {}

            # Handle train split first
            if "train" in flat_dataset:
                if self.train2val is not None and len(flat_dataset["train"]) > 0:
                    # Split train into train/val using train2val as validation ratio
                    split = _seeded_dataset_split(flat_dataset["train"], self.train2val, seed=42)
                    result["train"] = split["train"]
                    result["validation"] = split["test"]
                else:
                    result["train"] = flat_dataset["train"]

            # Copy other splits.
            # If train2val is set, original validation is intentionally overridden.
            for split_name, split_data in flat_dataset.items():
                if split_name == "train":
                    continue
                if self.train2val is not None and split_name == "validation":
                    continue
                result[split_name] = split_data

            return DatasetDict(result)
        else:
            # Single dataset: only split if train2val is specified
            if self.train2val is not None and len(flat_dataset) > 0:
                split = _seeded_dataset_split(flat_dataset, self.train2val, seed=42)
                return DatasetDict({
                    "train": split["train"],
                    "validation": split["test"]
                })
            else:
                # Return as-is
                return flat_dataset


class efficient_packed_causal_lm_pre_processor():
    """
    Efficient packed causal LM preprocessor.

    This version hardcodes the input field to `text`, tokenizes in batches,
    and then packs tokens into fixed-size blocks across each full split.
    """
    def __init__(self, tokenizer, block_size=512, train2val=None, map_args=None):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.train2val = train2val
        self.map_args = map_args or {}

        if str(self.map_args.get("num_proc", "")).lower() == "max":
            self.map_args["num_proc"] = os.cpu_count()

    def _tokenize_batch(self, batch):
        """Tokenize raw text without truncation so packing can happen afterwards."""
        enc = self.tokenizer(
            batch["text"],
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        return {"input_ids": enc["input_ids"]}

    def _iter_dataset_batches(self, dataset, batch_size):
        """Yield dictionary-of-lists batches from a Dataset."""
        if len(dataset) == 0:
            return

        if hasattr(dataset, "iter"):
            yield from dataset.iter(batch_size=batch_size)
            return

        for start_idx in range(0, len(dataset), batch_size):
            yield dataset[start_idx : start_idx + batch_size]

    def _pack_tokenized_split(self, tokenized_split):
        """
        Pack an entire tokenized split while carrying leftover tokens across
        tokenizer batches so no full blocks are lost at batch boundaries.
        """
        batch_size = self.map_args.get("batch_size", 1000)
        features = Features({"input_ids": Sequence(Value("int64"))})

        def generate_blocks():
            buffer = []
            buffer_start = 0

            for batch in self._iter_dataset_batches(tokenized_split, batch_size):
                for input_ids in batch["input_ids"]:
                    if not input_ids:
                        continue

                    buffer.extend(input_ids)

                    while len(buffer) - buffer_start >= self.block_size:
                        block_end = buffer_start + self.block_size
                        yield {"input_ids": buffer[buffer_start:block_end]}
                        buffer_start = block_end

                    # Periodically compact the buffer after consuming full blocks.
                    if buffer_start >= self.block_size * 32:
                        buffer = buffer[buffer_start:]
                        buffer_start = 0

        return Dataset.from_generator(generate_blocks, features=features)

    def process(self, raw_dataset):
        """
        Process dataset:
        1. Read the hardcoded `text` column.
        2. Tokenize raw text in batches.
        3. Pack into fixed-size blocks across the full split.
        4. Optionally split train into train/validation.
        """
        if isinstance(raw_dataset, DatasetDict):
            text_dataset = DatasetDict(
                {split: ds.select_columns(["text"]) for split, ds in raw_dataset.items()}
            )
        else:
            text_dataset = raw_dataset.select_columns(["text"])

        map_kwargs = dict(self.map_args)
        map_kwargs.pop("batched", None)
        map_kwargs.pop("batch_size", None)

        tokenized_dataset = text_dataset.map(
            self._tokenize_batch,
            batched=True,
            remove_columns=["text"],
            batch_size=self.map_args.get("batch_size", 1000),
            **map_kwargs,
        )

        if isinstance(tokenized_dataset, DatasetDict):
            packed_dataset = DatasetDict(
                {
                    split_name: self._pack_tokenized_split(split_data)
                    for split_name, split_data in tokenized_dataset.items()
                }
            )
        else:
            packed_dataset = self._pack_tokenized_split(tokenized_dataset)

        if isinstance(packed_dataset, DatasetDict):
            result = {}

            if "train" in packed_dataset:
                if self.train2val is not None and len(packed_dataset["train"]) > 0:
                    split = _seeded_dataset_split(packed_dataset["train"], self.train2val, seed=42)
                    result["train"] = split["train"]
                    result["validation"] = split["test"]
                else:
                    result["train"] = packed_dataset["train"]

            for split_name, split_data in packed_dataset.items():
                if split_name == "train":
                    continue
                if self.train2val is not None and split_name == "validation":
                    continue
                result[split_name] = split_data

            return DatasetDict(result)

        if self.train2val is not None and len(packed_dataset) > 0:
            split = _seeded_dataset_split(packed_dataset, self.train2val, seed=42)
            return DatasetDict({"train": split["train"], "validation": split["test"]})

        return packed_dataset



    
