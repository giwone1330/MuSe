import random
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm
import os
import sys
import numpy as np


# create equation datasets
class EquationSyntheticDatasetBuilder:
    def __init__(self, generate_args, save_args, staging_dir):
        # splitsizes: dict[str, int],
        # digitranges: dict[str, dict[str, list[int]]], # {split, {operand_1 : [mindigit, maxdigit], operand_2 : [mindigit, maxdigit]}} minmax for each operands.
        # balance_digits: dict[str, bool],
        # shuffle: dict[str, bool],
        # operators: dict[str, dict[str, list[str]]], # {operator_1 : ["+", "-"], operator_2 : ["+"]} any of the two for the first operator, then + for the second operator
        # radix: int,
        # parenthesis: None = None, #TODO
        # seed: int | None = None,

        # generate args
        self.generate_args = generate_args
        self.splitsizes = generate_args["splitsizes"]
        self.digitranges = generate_args["digitranges"]
        # self.requiredranges = generate_args["requiredranges"]
        self.balance_digits = generate_args["balance_digits"]
        self.shuffle = generate_args["shuffle"]
        self.operators = generate_args["operators"]
        self.radix = generate_args["radix"]
        self.parenthesis = generate_args["parenthesis"]
        self.batch_size = generate_args["batch_size"]

        # save args
        self.save_args = save_args
        self.operation_map = {
            "+": "a",
            "-": "s",
            "/": "fd",
            "//": "id",
            "*": "m",
            "%": "r"
        }

        # staging_dir
        self.staging_dir = staging_dir

        

        # add sanity checks here
    def get_name(self, split="train"):

        operands = self.digitranges[split]
        operators = self.operators[split]
        operands = {int(k.split("_")[-1]): v for k, v in operands.items() if k.startswith("operand")}
        operators = {int(k.split("_")[-1]): v for k, v in operators.items() if k.startswith("operator")}  
        
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
        for i, operand_lengths in enumerate(sorted_operands):
            expression_parts.append(f"{operand_lengths[0]}-{operand_lengths[1]}")
            if i < len(sorted_operators):
                ops = []
                for unmapped_op in sorted_operators[i]:
                    mapped_op = self.operation_map[unmapped_op]
                    ops.append(mapped_op)
                sorted_ops = sorted(ops)
                expression_parts.extend(sorted_ops)
        name = "".join(expression_parts)
        return name

    def solve(self, data: dict):
        # Separate operands and operators
        operands = {int(k.split("_")[-1]): v for k, v in data.items() if k.startswith("operand")}
        operators = {int(k.split("_")[-1]): v for k, v in data.items() if k.startswith("operator")}

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
        expression = "".join(expression_parts)

        # Evaluate the expression safely
        result = eval(expression)
        return expression, result

    def _make_examples(self, split, n):
        """Generate n synthetic examples (rows) using NumPy vectorization."""
        out_dicts = []

        # Pre-allocate results as dict of lists
        batch = {col: [] for col in self.digitranges[split].keys()}
        for operator_column in self.operators[split].keys():
            batch[operator_column] = []

        # --- Vectorized operand sampling ---
        for operand_column, (min_digit, max_digit) in self.digitranges[split].items():
            (min_digit, max_digit) = (int(min_digit), int(max_digit))
            rng = np.random.default_rng()
            # operands = rng.integers(low, high + 1, size=n)
            if self.balance_digits[split]:
                # sample digit length uniformly
                digits = rng.integers(min_digit, max_digit + 1, size=n)

                # now sample numbers for each digit group
                # operands = np.empty(n, dtype=np.int64)
                operands = []
                for d in range(min_digit, max_digit + 1):
                    mask = digits == d
                    count = mask.sum()
                    if count > 0:
                        low, high = 10**(d - 1), 10**d - 1
                        # operands[mask] = rng.integers(low, high + 1, size=count)
                        operands.extend([str(random.randint(low, high)) for _ in range(count)])

            else:
                low, high = 10**(min_digit - 1), 10**max_digit - 1
                # operands = rng.integers(low, high + 1, size=n)
                operands = [str(random.randint(low, high)) for _ in range(n)]

            batch[operand_column] = operands
            # batch[operand_column] = operands.astype(str)

        # --- Vectorized operator sampling ---
        for operator_column, operators in self.operators[split].items():
            batch[operator_column] = np.random.choice(operators, size=n).astype(str)

        # Convert back to list of dicts
        out_tuples = [
            tuple(sorted((col, str(batch[col][i])) for col in batch))
            for i in range(n)
        ]
        return out_tuples

    
    def _make_datalist(self, split, data_dict_set):
        splitsize = self.splitsizes[split]
        progress_bar = tqdm(range(splitsize), desc=f"Sample : {split}")
        loop_counter = 0
        data_split = set()

        # process requiredranges
        #TODO if split in self.requiredranges:

        while len(data_split) < splitsize:
            # try splitsize*10 to sample data that meets the conditions
            if loop_counter > splitsize*2:
                print("Exiting to prevent an infinite loop when creating dataset!")
                break
            before_batch = len(data_split)

            # sample data
            data_sample = set(self._make_examples(split, self.batch_size))

            for k, v in data_dict_set.items():
                data_sample = data_sample - v

            data_split.update(data_sample)

            loop_counter +=1
            progress_bar.update(min(splitsize,len(data_split))-before_batch)
        
        # if self.shuffle[split]:
        #     random.shuffle(data_split)

        data_dict_set[split] = data_split

        return data_dict_set
    

    def build(self) -> DatasetDict:
        """Build and return a DatasetDict with train/test/validation splits."""
        data_dict_set = {}
        priority = ["test", "eval", "train"]
        splits = self.splitsizes.keys()
        for pri in priority:
            if pri in splits: # create the test first (smallest first)
                split = pri
                splits = [x for x in splits if x != split] # remove running split from the queue
                data_dict_set = self._make_datalist(split, data_dict_set)

        # resuing for remainer of the splits
        for split in splits:
            data_dict_set = self._make_datalist(self, split, data_dict_set)

        # from data_dict_set make acutal datasets
        data_dict = {}
        for k, v in data_dict_set.items():
            data_list = [dict(t) for t in v]
            data_list = data_list[:self.splitsizes[k]]


            progress_bar = tqdm(range(len(data_list)), desc=f"Calculating : {k}")
            for data in (data_list):
                expr, result_int = self.solve(data)
                data["answer"] = str(result_int)
                progress_bar.update(1)
            data_dict[k] = Dataset.from_list(data_list)

        # make complete DatasetDict
        complete_data_dict = DatasetDict(data_dict)

        # save
        if self.save_args['save']:
            self.save_name = self.get_name()
            if self.save_args['save_in_staging']:
                save_path = os.path.join(self.staging_dir, self.save_args['save_dir'], self.save_name)
            else:
                save_path = os.path.join(self.save_args['save_dir'], self.save_name)
        
            complete_data_dict.save_to_disk(save_path)
            
            # check for exit after save
            if self.save_args['exit_after_save']:
                sys.exit()

        # return the full DatasetDict
        return complete_data_dict













# create equation datasets
class EquationSyntheticDatasetBuilder_bak:
    def __init__(self, generate_args, save_args, staging_dir):
        # splitsizes: dict[str, int],
        # digitranges: dict[str, dict[str, list[int]]], # {split, {operand_1 : [mindigit, maxdigit], operand_2 : [mindigit, maxdigit]}} minmax for each operands.
        # balance_digits: dict[str, bool],
        # shuffle: dict[str, bool],
        # operators: dict[str, dict[str, list[str]]], # {operator_1 : ["+", "-"], operator_2 : ["+"]} any of the two for the first operator, then + for the second operator
        # radix: int,
        # parenthesis: None = None, #TODO
        # seed: int | None = None,

        # generate args
        self.generate_args = generate_args
        self.splitsizes = generate_args["splitsizes"]
        self.digitranges = generate_args["digitranges"]
        self.balance_digits = generate_args["balance_digits"]
        self.shuffle = generate_args["shuffle"]
        self.operators = generate_args["operators"]
        self.seed = generate_args["seed"]
        self.radix = generate_args["radix"]
        self.parenthesis = generate_args["parenthesis"]

        # save args
        self.save_args = save_args
        self.operation_map = {
            "+": "a",
            "-": "s",
            "/": "fd",
            "//": "id",
            "*": "m",
            "%": "r"
        }

        # staging_dir
        self.staging_dir = staging_dir

        
        if self.seed is not None:
            random.seed(self.seed)

        # add sanity checks here
    def get_name(self, split="train"):

        operands = self.digitranges[split]
        operators = self.operators[split]
        operands = {int(k.split("_")[-1]): v for k, v in operands.items() if k.startswith("operand")}
        operators = {int(k.split("_")[-1]): v for k, v in operators.items() if k.startswith("operator")}  
        
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
        for i, operand_lengths in enumerate(sorted_operands):
            expression_parts.append(f"{operand_lengths[0]}-{operand_lengths[1]}")
            if i < len(sorted_operators):
                ops = []
                for unmapped_op in sorted_operators[i]:
                    mapped_op = self.operation_map[unmapped_op]
                    ops.append(mapped_op)
                sorted_ops = sorted(ops)
                expression_parts.extend(sorted_ops)
        name = "".join(expression_parts)
        return name

    def solve(self, data: dict):
        # Separate operands and operators
        operands = {int(k.split("_")[-1]): v for k, v in data.items() if k.startswith("operand")}
        operators = {int(k.split("_")[-1]): v for k, v in data.items() if k.startswith("operator")}

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
        expression = "".join(expression_parts)

        # Evaluate the expression safely
        result = eval(expression)
        return expression, result



    def _make_example(self, split):
        """Generate one synthetic example (row)."""
        out_dict = {}
        # sample operand
        for i, (operand_column, (min_digit, max_digit)) in enumerate(self.digitranges[split].items()):
            if self.balance_digits[split]:
                digit = random.randint(min_digit, max_digit)
                operand = random.randint(10**(digit-1), 10**digit-1)
            else:
                operand = random.randint(10**(min_digit-1), 10**(max_digit)-1)
            out_dict[f"{operand_column}"] = str(operand)

        # sample operator
        for i, (operator_column, operators) in enumerate(self.operators[split].items()):
            operator = random.choice(operators)
            out_dict[operator_column] = str(operator)

        # calculate eqn
        _, result_int =  self.solve(out_dict)
        out_dict["answer"] = str(result_int)

        return out_dict
    
    def _make_datalist(self, split, data_dict_set):
        splitsize = self.splitsizes[split]
        progress_bar = tqdm(range(splitsize), desc=split)
        loop_counter = 0
        data_split = []
        while len(data_split) < splitsize:
            # try splitsize*10 to sample data that meets the conditions
            if loop_counter > splitsize*10:
                print("Exiting to prevent an infinite loop when creating dataset!")
                break

            # sample data
            data_sample = self._make_example(split)

            # check for duplicates inside the split
            if data_sample in data_split:
                loop_counter += 1
                continue
            
            # check for duplicates across all the previous splits
            duplicate_found = False
            for k, v in data_dict_set.items():
                if data_sample in v:
                    duplicate_found = True
                    break
            
            if duplicate_found:
                loop_counter += 1
                continue

            # no duplicates found
            data_split.append(data_sample)
            progress_bar.update(1)
        
        if self.shuffle[split]:
            random.shuffle(data_split)

        data_dict_set[split] = data_split

        return data_dict_set
    

    def build(self) -> DatasetDict:
        """Build and return a DatasetDict with train/test/validation splits."""
        data_dict_set = {}
        priority = ["test", "validation", "train"]
        splits = self.splitsizes.keys()
        for pri in priority:
            if pri in splits: # create the test first (smallest first)
                split = pri
                splits = [x for x in splits if x != split] # remove running split from the queue
                data_dict_set = self._make_datalist(split, data_dict_set)

        # resuing for remainer of the splits
        for split in splits:
            data_dict_set = self._make_datalist(self, split, data_dict_set)

        # from data_dict_set make acutal datasets
        data_dict = {}
        for k, v in data_dict_set.items():
            data_dict[k] = Dataset.from_list(v)

        # make complete DatasetDict
        complete_data_dict = DatasetDict(data_dict)

        # save
        if self.save_args['save']:
            self.save_name = self.get_name()
            if self.save_args['save_in_staging_dir']:
                save_path = os.path.join(self.staging_dir, self.save_args['save_dir'], self.save_name)
            else:
                save_path = os.path.join(self.save_args['save_dir'], self.save_name)
        
            complete_data_dict.save_to_disk(save_path)
            
            # check for exit after save
            if self.save_args['exit_after_save']:
                sys.exit()

        # return the full DatasetDict
        return complete_data_dict










class PokemonSyntheticDatasetBuilder:
    def __init__(
        self,
        num_samples=1000,
        train_ratio=0.8,
        test_ratio=0.1,
        val_ratio=0.1,
        pokemons=None,
        types=None,
        level_range=(1, 100),
        seed=None,
    ):
        self.num_samples = num_samples
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.pokemons = pokemons or ["bulbasaur", "squirtle", "charmander", "pikachu"]
        self.types = types or ["grass", "water", "fire", "electric"]
        self.level_range = level_range

        if seed is not None:
            random.seed(seed)

        # sanity check: ratios must add up to 1
        total_ratio = train_ratio + test_ratio + val_ratio
        if not abs(total_ratio - 1.0) < 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    def _make_example(self):
        """Generate one synthetic example (row)."""
        return {
            "pokemon": random.choice(self.pokemons),
            "type": random.choice(self.types),
            "level": random.randint(*self.level_range),
        }

    def build(self) -> DatasetDict:
        """Build and return a DatasetDict with train/test/validation splits."""
        # Generate all rows at once (still efficient for millions of rows)
        data = [self._make_example() for _ in range(self.num_samples)]

        # Shuffle before splitting
        random.shuffle(data)

        # Split indexes
        n_train = int(self.train_ratio * self.num_samples)
        n_test = int(self.test_ratio * self.num_samples)
        n_val = self.num_samples - n_train - n_test

        train_data = data[:n_train]
        test_data = data[n_train:n_train + n_test]
        val_data = data[n_train + n_test:]

        return DatasetDict({
            "train": Dataset.from_list(train_data),
            "test": Dataset.from_list(test_data),
            "validation": Dataset.from_list(val_data),
        })