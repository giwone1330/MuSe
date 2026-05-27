import math
import random
import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
import importlib
import inspect
import os
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import sys
import datasets
import tokenizers
import inspect
import requests



def set_seed(seed: int, strict=False):
    # 1. Python / standard libs
    random.seed(seed)
    np.random.seed(seed)

    # 2. PyTorch — CPU and CUDA
    torch.manual_seed(seed)
    # If you are using multiple GPUs, also:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if strict: # strict deterministic
        # 3. cuDNN and backend settings
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # 4. Use deterministic algorithms
        # This makes PyTorch raise errors if a non-deterministic algorithm is used.
        torch.use_deterministic_algorithms(True, warn_only=True)

        # 5. (Optional) If you want to allow non-deterministic but reproducible behavior,
        # you can pass `warn_only=True`:
        # torch.use_deterministic_algorithms(True, warn_only=True)

        # 6. Environment variable for CUDA (if CUDA ≥ 10.2)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    else: # relaxed deterministic for efficiency
        # 3. cuDNN and backend settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def seed_worker(worker_id):
    # For DataLoader worker seeds
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def flatten_dict(d, parent_key='', sep='.', to_list_str = True):
    """
    Recursively flattens a nested dictionary/OmegaConf into a flat dictionary
    with dot-separated keys.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) or hasattr(v, 'items'):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    if to_list_str:
        items_list = [f"{k}:{v}" for k,v in items.items()]
        return items_list
    else:
        return items
    

def register_for_auto_classes(cfg, model_type=None): # don't need model_type
    # register config
    config_str = cfg.model.config._target_
    # config_str = config_str.replace("src.", ".")
    module_path, attr_name = config_str.rsplit(".", 1)
    module = importlib.import_module(module_path)
    # Get the attribute dynamically
    config_cls = getattr(module, attr_name)
    model_type = getattr(config_cls, "model_type")
    # model_type = getattr(config_cls, "model_type", None) or model_type
    AutoConfig.register(model_type, config_cls, exist_ok=True)
    getattr(config_cls, "register_for_auto_class")()

    # register model
    model_str = cfg.model._target_
    # model_str = model_str.replace("src.", ".")
    module_path, attr_name = model_str.rsplit(".", 1)
    module = importlib.import_module(module_path)
    # Get the attribute dynamically
    model_cls = getattr(module, attr_name)

    # Search for task specific AutoModel
    automodel_prefix = "AutoModelFor"
    import transformers as tfm
    for tf_name, tf_cls in inspect.getmembers(tfm, inspect.isclass):
        if tf_name.startswith("AutoModelFor"):
            task = tf_name[len("AutoModel"):] # ex) ForImageClassification
            if attr_name.endswith(task):
                tf_cls.register(config_cls, model_cls, exist_ok=True)
                getattr(model_cls, "register_for_auto_class")(tf_name)
                print(f"Registered class: {attr_name} to {tf_name}")
                return
    # If not found register to AutoModel
    AutoModel.register(config_cls, model_cls, exist_ok=True)
    getattr(model_cls, "register_for_auto_class")("AutoModel")
    print(f"Registered class: {attr_name} to AutoModel")
    # Also register to AutoModelForCausalLM
    try:
        AutoModelForCausalLM.register(config_cls, model_cls, exist_ok=True)
        getattr(model_cls, "register_for_auto_class")("AutoModelForCausalLM")
        print(f"Registered class: {attr_name} to AutoModelForCausalLM")
    except:
        print(f"Failed to register class: {attr_name} to AutoModelForCausalLM")
    return

#? not sure if this is needed
# def dynamic_local_imports(cfg):
#     # add all the _target_ to list
#     import_list = []
#     import_list.append(cfg.model.config._target_)
#     import_list.append(cfg.model._target_)

#     # iterate through all the import paths and import
#     for import_path in import_list:
#         if import_path.startswith("src."):
#             # import from this repo
#             module_path, attr_name = import_path.rsplit(".", 1)
#             module = importlib.import_module(module_path)
#             # Get the attribute dynamically
#             getattr(module, attr_name)
#         else:
#             # import from external library
#             pass


def get_run_name(s, prefix, suffix):
    """
    Returns the substring between prefix and suffix, or None if not found.
    """
    start = s.find(prefix)
    if start == -1:
        return None
    start += len(prefix)
    end = s.find(suffix, start)
    if end == -1:
        return None
    return s[start:end]

def exist_and_not_none(dic, *keys):
    """
    Recursively check that a nested key path exists and is not None.

    Example:
        exist_and_not_none(cfg, "dataset", "pre_processor")
    """
    current = dic
    for key in keys:
        if not isinstance(current, DictConfig) or key not in current or current[key] is None:
            return False
        current = current[key]
    return True

def get_tokenizer(cfg):
    # Keys to remove temorarily for instantiation
    keys_to_remove = ["non_inst"]
    cfg_tokenizer = OmegaConf.create({k: v for k, v in cfg.tokenizer.items() if k not in keys_to_remove})
    # instantiate backend_tokneizer
    raw_tokenizer = hydra.utils.instantiate(cfg_tokenizer, _convert_="partial")


    # configure pre/post_tokenizer, normalizers
    if exist_and_not_none(cfg.tokenizer, "non_inst", "setattr"):
        for k, v in cfg.tokenizer.non_inst.setattr.items():
            if v != None:
                if exist_and_not_none(v,"_target_"):
                    setattr(raw_tokenizer, k, hydra.utils.instantiate(v, _convert_="partial"))
                    # print(f"configured {k} for tokenizer")
                else:
                    setattr(raw_tokenizer, k, v)
    
    # add special tokens
    special_token_dict = {}
    if exist_and_not_none(cfg.tokenizer, "non_inst", "special_tokens"):
        if len(cfg.tokenizer.non_inst.special_tokens) != 0:
            for sp_token in cfg.tokenizer.non_inst.special_tokens:
                for k, v in sp_token.items():
                    if isinstance(v, str):
                        special_token_dict[k] = v
                    else:
                        special_token_dict[k] = hydra.utils.instantiate(v, _convert_="partial")
        
            special_token_list = list(special_token_dict.values())
            raw_tokenizer.add_special_tokens(special_token_list)


    # add post_processor : how to format encoded(split) tokens
    if exist_and_not_none(cfg.tokenizer, "non_inst", "post_processor"):
        template_dict = {"special_tokens": []}
        for t_k, t_v in cfg.tokenizer.non_inst.post_processor.templates.items():
            for k, v in special_token_dict.items():
                if k in t_v:
                    if isinstance(v, str):
                        sp_token = v
                    else:
                        sp_token = v.content
                    t_v = t_v.replace(k, sp_token)
                    sp_token_id = raw_tokenizer.token_to_id(sp_token)
                    template_dict["special_tokens"].append((sp_token, sp_token_id))
            template_dict[t_k]=t_v

        raw_tokenizer.post_processor = tokenizers.processors.TemplateProcessing(**template_dict)


    # tokenizer trainer
    if exist_and_not_none(cfg.tokenizer, "non_inst", "trainer"):
        tokenizer_trainer = hydra.utils.instantiate(cfg.tokenizer.non_inst.trainer, _convert_="partial")
        print("Training tokenizer...")
        raw_tokenizer.train("training/files/here", tokenizer_trainer)
        print("Tokenizer trained")
    
    # wrap raw_tokenizer(backend) to "pretrainedtokenizer(fast)"
    if (not isinstance(raw_tokenizer, transformers.PreTrainedTokenizer)) and (not isinstance(raw_tokenizer, transformers.PreTrainedTokenizerFast)):
        if exist_and_not_none(cfg, "non_inst", "wrapper_config"):
            wrapper_config = cfg.tokenizer.non_inst.wrapper_config
        else:
            wrapper_config = {}

        try:
            tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer, **wrapper_config, **special_token_dict)
        except:
            tokenizer = transformers.PreTrainedTokenizer(tokenizer_object=raw_tokenizer, **wrapper_config, **special_token_dict)
    else:
        tokenizer = raw_tokenizer

    return tokenizer

def get_model(cfg, tokenizer):
    # instantiate model and config
    # check for tokenizer - model config matches
    keys_to_remove = ["non_inst"]
    cfg_model = OmegaConf.create({k: v for k, v in cfg.model.items() if k not in keys_to_remove})

    # check for tokenizer - model config matches
    if cfg.env.auto_resolve.model:
        try:
            if cfg.model.config.vocab_size != len(tokenizer.vocab):
                cfg_model.config["vocab_size"]=len(tokenizer.vocab)
        except:
            print("Could not auto resolve model config: vocab_size")
        try:
            for k,v in tokenizer.special_tokens_map.items():
                # 'bos_token', '[BOS]'
                cfg_model.config[k+"_id"] = tokenizer.vocab[v]
        except:
            print("Could not auto resolve model config: special tokens")

    # instantiate model
    model = hydra.utils.instantiate(cfg_model, _convert_="partial")

    # configure model attributes
    if exist_and_not_none(cfg.model, "non_inst", "setattr"):
        for k, v in cfg.model.non_inst.setattr.items():
            if v != None:
                setattr(model, k, hydra.utils.instantiate(v, _convert_="partial"))
                # print(f"configured {k} for model")

    return model

def get_datasets(cfg, tokenizer):
    keys_to_remove = ["non_inst"]
    cfg_dataset = OmegaConf.create({k: v for k, v in cfg.dataset.items() if k not in keys_to_remove})
    # create
    if cfg_dataset._target_.startswith("src."):
        builder = hydra.utils.instantiate(cfg_dataset, staging_dir=cfg.state.staging_dir, _convert_="partial")
        if cfg_dataset.save_args.force_build:
            raw_datasets = builder.build()
        else:
            try:
                # try loading in if exists
                dataset_name = builder.get_name()
                data_path = os.path.join(cfg_dataset.save_args.save_dir, dataset_name)
                raw_datasets = datasets.load_from_disk(data_path)
                print(f"Successfuly loaded dataset from {data_path}")
            except:
                print(f"Failed to load dataset from {data_path}, building dataset...")
                raw_datasets = builder.build()
        if cfg_dataset.save_args.push_to_hub:
            raw_datasets.push_to_hub(f"{dataset_name}")
            if cfg_dataset.save_args.exit_after_push:
                    sys.exit()
    
    else:
        raw_datasets = hydra.utils.instantiate(cfg_dataset)
        # raw_datasets = load_dataset("wikitext", "wikitext-2-v1")
        # columns {'test': ['text'], 'train': ['text'], 'validation': ['text']}

    # instantiate preprocessor
    if exist_and_not_none(cfg.dataset, "non_inst", "pre_processor"):
        pre_processor = hydra.utils.instantiate(cfg.dataset.non_inst.pre_processor, tokenizer=tokenizer, _convert_="partial")
        # preprocess = formatting + tokenization + dataset column modification
        tokenized_datasets = pre_processor.process(raw_datasets)
        print(tokenized_datasets.column_names)
        tokenized_datasets.set_format("torch")

    else:
        tokenized_datasets = None
        print('pre_processor not provided, unable to tokenize dataset')

    if exist_and_not_none(cfg.dataset, "non_inst", "split_map", "train"):
        train_dataset = tokenized_datasets[cfg.dataset.non_inst.split_map.train]
    else:
        train_dataset = None
    if exist_and_not_none(cfg.dataset, "non_inst", "split_map", "eval"):
        eval_dataset = tokenized_datasets[cfg.dataset.non_inst.split_map.eval]
    else:
        eval_dataset = None

    return train_dataset, eval_dataset, raw_datasets, tokenized_datasets

def get_optimizer(cfg, model):
    if exist_and_not_none(cfg.trainer, "non_inst", "optimizer", "_target_"):
        # instantiate optimizer with model parameters
        optimizer = hydra.utils.instantiate(cfg.trainer.non_inst.optimizer, params=model.parameters(), _convert_="partial")
    else:
        optimizer = None

    return optimizer

def get_scheduler(cfg, optimizer, train_dataset):
    train_dataloader_len = math.ceil(len(train_dataset) / cfg.trainer.args.per_device_train_batch_size)
    cfg.state.num_training_steps_per_epoch = (train_dataloader_len // cfg.trainer.args.gradient_accumulation_steps)
    cfg.state.num_training_steps = cfg.state.num_training_steps_per_epoch * cfg.trainer.args.num_train_epochs

    if exist_and_not_none(cfg.trainer, "non_inst", "scheduler", "num_warmup_steps"):
        if cfg.trainer.non_inst.scheduler.num_warmup_steps > 1: # int: warmup steps
            cfg.state.num_warmup_steps = cfg.trainer.non_inst.scheduler.num_warmup_steps
        else: # float: warmup ratio
            cfg.state.num_warmup_steps = int(cfg.trainer.non_inst.scheduler.num_warmup_steps * cfg.state.num_training_steps)

    if exist_and_not_none(cfg.trainer, "non_inst", "scheduler", "_target_") and optimizer != None:
        scheduler = hydra.utils.instantiate(cfg.trainer.non_inst.scheduler, optimizer=optimizer, num_training_steps=cfg.state.num_training_steps, num_warmup_steps=cfg.state.num_warmup_steps, _convert_="partial")
    else:
        scheduler = None

    return scheduler







def get_dataloader(cfg, tokenized_dataset, data_collator, g, seed_worker):

    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset, batch_size=cfg.trainer.trainer_args.train_batch_size, collate_fn=data_collator, generator=g, worker_init_fn=seed_worker,num_workers=cfg.state.num_proc, pin_memory=True
    )
    return dataloader



def get_dataloaders(cfg, tokenized_datasets, tokenizer, accelerator, g, seed_worker):
    # Data collator
    # convert pad_token_id -> -100, can do dynamic padding per batch
    if exist_and_not_none(cfg.dataset, "data_collator"):
        print("instantiating data_collator")
        data_collator = hydra.utils.instantiate(cfg.dataset.data_collator, tokenizer=tokenizer, _convert_="partial")
        print("instantiated data_collator")
    else: # default data collator
        data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = torch.utils.data.DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=cfg.trainer.trainer_args.train_batch_size, collate_fn=data_collator, generator=g, worker_init_fn=seed_worker,num_workers=cfg.state.num_proc, pin_memory=True
    )
    eval_dataloader = torch.utils.data.DataLoader(
        tokenized_datasets["validation"], batch_size=cfg.trainer.trainer_args.eval_batch_size, collate_fn=data_collator, generator=g, worker_init_fn=seed_worker, num_workers=cfg.state.num_proc, pin_memory=True
    )
    try:
        test_dataloader = torch.utils.data.DataLoader(
            tokenized_datasets["test"], batch_size=1, collate_fn=data_collator, generator=g, worker_init_fn=seed_worker, num_workers=cfg.state.num_proc, pin_memory=True
        )
    except:
        test_dataloader = None
        print("No test dataset")
    return train_dataloader, eval_dataloader, test_dataloader



def get_trainer(cfg, model, optimizer, scheduler, tokenizer, train_dataset, eval_dataset=None, tokenized_datasets=None, hooks=None):
    # note that the train_dataset and eval_dataset should be raw_datasets before tokenization
    g = torch.Generator()
    g.manual_seed(cfg.env.seed)


    # Data collator
    # convert pad_token_id -> -100, can do dynamic padding per batch
    if exist_and_not_none(cfg.trainer, "non_inst", "data_collator"):
        data_collator = hydra.utils.instantiate(cfg.trainer.non_inst.data_collator, tokenizer=tokenizer, _convert_="partial")
    else: # default data collator
        data_collator = None
        # data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        # each trainer should handle data collator if None


    # check if custom trainer
    if cfg.trainer._target_.startswith("src."):

        # dataloaders
        train_dataloader, eval_dataloader, test_dataloader = get_dataloaders(cfg, tokenized_datasets, data_collator, tokenizer, g, seed_worker)

        # testing dataloader
        if cfg.env.debug:
            for batch in train_dataloader:
                break
            batch_example = {k: v.shape for k, v in batch.items()}
            print(f"batch k, v.shape : {batch_example}")
            batch_values = {k: v[0] for k, v in batch.items()}
            print(f"batch k, v[0] : {batch_values}")

            # test model output
            outputs = model(**batch)
            print(outputs.loss, outputs.logits.shape)


        keys_to_remove = ["accelerator", "optimizer", "scheduler"]
        cfg_trainer = OmegaConf.create({k: v for k, v in cfg.trainer.items() if k not in keys_to_remove})
        trainer = hydra.utils.instantiate(cfg_trainer,
                                        model=model,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        tokenizer=tokenizer,
                                        cfg=cfg,
                                        train_dataloader=train_dataloader,
                                        eval_dataloader=eval_dataloader,
                                        test_dataloader=test_dataloader,
                                        hooks=None,
                                        _recursive_=False,
                                        _convert_="none") # don't want to instantiate accelerator, optimizer, scheduler
    else:
        # use hf transformers trainer
        keys_to_remove = ["non_inst"]
        cfg_trainer = OmegaConf.create({k: v for k, v in cfg.trainer.items() if k not in keys_to_remove})
        trainer = hydra.utils.instantiate(cfg_trainer,
                                        model=model,
                                        train_dataset=train_dataset,
                                        eval_dataset=eval_dataset,
                                        data_collator=data_collator,
                                        processing_class=tokenizer,
                                        optimizers=(optimizer, scheduler),
                                        args={
                                            "output_dir": cfg.state.output_dir,
                                            "seed": cfg.env.seed,
                                        },
                                        _convert_="object")
    return trainer


def get_evaluator(cfg, model, tokenizer, datasets):
    keys_to_remove = ["non_inst"]
    cfg_evaluator = OmegaConf.create({k: v for k, v in cfg.evaluator.items() if k not in keys_to_remove})
    
    # evaluator
    evaluator = hydra.utils.instantiate(
        cfg_evaluator,
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
        _convert_="none",
        _recursive_=False
    )
    return evaluator


def send_telegram_message(bot_token, message, chat_id=None):
    try:
        if bot_token is None:
            print("TELEGRAM_BOT_TOKEN not set, unable to send telegram message")
            return

        # get chat id
        if chat_id is None:
            url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            chat_id = data['result'][0]['message']['chat']['id']

        # send message
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
        }

        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
    except:
        print("Failed to send Telegram complete message.")
        pass
