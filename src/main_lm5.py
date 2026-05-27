# python std library
import math
import os
import sys
import time
import inspect
import socket
import importlib

# ML / related libraries
import torch
import numpy as np
# import sklearn
# import timm
import tiktoken

# hf ecosystem
import huggingface_hub
import transformers
import datasets
import tokenizers
import accelerate
import evaluate
import diffusers

# tracking & plotting
import wandb
import matplotlib.pyplot as plt

# utils
from tqdm.auto import tqdm
import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

# for error handling
import traceback


# imported utils
from src.utils.utils import set_seed, seed_worker, flatten_dict, register_for_auto_classes, exist_and_not_none, get_tokenizer, get_model, get_datasets, get_dataloaders, get_dataloader, get_trainer, get_optimizer, get_scheduler, get_run_name, get_evaluator, send_telegram_message

import warnings
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")

##########################################
# temporary helper functions
dotenv.load_dotenv()

##########################################

#-----------------------------------------------------
# Anything to run prior to main script
#-----------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    #* dummy if statement for easy folding
    if True:
        print("Start of main")
        # register environment variables in .env file
        dotenv.load_dotenv()

        # Create .state and allow modification
        state = OmegaConf.create({})
        OmegaConf.set_struct(state, False)
        with open_dict(cfg):
            cfg.state = state

        cfg.state.username = os.getenv("USERNAME")
        cfg.state.project = os.getenv("PROJECT")
        # start measuring time
        cfg.state.script_start_time = time.time()
        if exist_and_not_none(cfg.env, 'device'):
            cfg.state.device = cfg.env.device
        else:
            cfg.state.device = "cuda" if torch.cuda.is_available() else "cpu"

        # set staging dir #? but note that writing to it is not allowed
        cfg.state.staging_dir = os.path.join("/staging", os.getenv("USERNAME"), os.getenv("PROJECT"))

        # check if the model is loaded from local source code, which then is usually custom model #? not sure if this is needed
        cfg.state.custom_model = str(cfg.model._target_).startswith("src.")

        # set thread count
        if cfg.env.num_proc == "Max":
            cfg.state.num_proc = os.cpu_count()
        else:
            cfg.state.num_proc = cfg.env.num_proc

        # define output_dir, custom_model
        if str(hydra.core.hydra_config.HydraConfig.get().mode) == "RunMode.RUN":
            cfg.state.run_mode = "single"
            cfg.state.output_dir = hydra.core.hydra_config.HydraConfig.get().run.dir # /runs/~~~/local/hydra
            cfg.state.output_dir = os.path.dirname(cfg.state.output_dir) # /runs/~~~/local

        else:
            # TODO need extra mappings for hydra sweep dir  to output
            cfg.state.run_mode = "sweep"
            cfg.state.output_dir = os.path.join(hydra.core.hydra_config.HydraConfig.get().sweep.dir, hydra.core.hydra_config.HydraConfig.get().sweep.subdir) #/runs/~~~/chtc/hydra/
            cfg.state.output_dir = os.path.dirname(cfg.state.output_dir) # /runs/~~~/local #! might not work
        
        # check local or chtc
        if "/chtc/hydra/" in cfg.state.output_dir:
            cfg.state.env = "chtc"
            cfg.state.run_name = get_run_name(cfg.state.output_dir, "runs/", "/local/")
        elif "/local" in cfg.state.output_dir: # changes made so that it alwasy saves to "local"
            cfg.state.env = "local"
            cfg.state.run_name = get_run_name(cfg.state.output_dir, "runs/", "/local")
        else:
            cfg.state.env = "cli"
            cfg.state.run_name = get_run_name(cfg.state.output_dir, "runs/", "/hydra")
        
        # override output dir if specified in env
        if exist_and_not_none(cfg.env, "output_dir"):
            cfg.state.output_dir = cfg.env.output_dir


        # update run name with chtc details
        if exist_and_not_none(cfg.env, 'chtc', 'process'):
            cfg.state.run_name = f"{cfg.state.run_name}_{cfg.env.chtc.cluster}_{cfg.env.chtc.process}_{cfg.env.chtc.model_config}"
        
        # resolve interpolations
        OmegaConf.resolve(cfg)

        #! Set all the variables that will be relavant above & inside cfg.state
        #! ####################################################        
        # Instantiate & initialize tracker
        if exist_and_not_none(cfg.env, "tracker", "wandb"):
            wandb.init(
                project=cfg.state.project,
                name=cfg.state.run_name,
                config=dict(cfg),
                **cfg.env.tracker.wandb
            )
        if exist_and_not_none(cfg, "tracker", "tensorboard"):
            hydra.utils.instantiate(cfg.tracker.tensorboard, _convert_="partial")


        # send telegram start message
        if cfg.state.device == "cuda":
            send_telegram_message(os.getenv("TELEGRAM_BOT_TOKEN"), f"Start : {cfg.state.run_name}", os.getenv("TELEGRAM_CHAT_ID"))
        else:
            #! notify the run performed on cuda
            send_telegram_message(os.getenv("TELEGRAM_BOT_TOKEN"), f"Start-CPU : {cfg.state.run_name}", os.getenv("TELEGRAM_CHAT_ID"))


        # print out the configs
        print(f"\n======= hydra configs =======\n")
        print(OmegaConf.to_yaml(cfg))  # prints out the final cfg after command line overrides
        print(f"\n============================\n")


        # Login to Hugging Face Hub with environemtn variable HF_TOKEN
        huggingface_hub.login(token=os.getenv('HF_TOKEN'))
        print("HF_login : Successful")
        
        if cfg.env.verbose:
            # print machine
            machine_name = os.environ.get("Machine", socket.gethostname())
            print(f"Machine Name: {machine_name}")
            # print gpu
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                print("No GPU found")
        
            print(f"device : {cfg.state.device}")
            print(f"Start time: {cfg.state.script_start_time}")
            print(f"Staging dir: {cfg.state.staging_dir}")
            print(f"Is custom model? : {cfg.state.custom_model}")
    # configuration and setup end
    

    # Setting Seed
    set_seed(cfg.env.seed, cfg.env.deterministic)

    #-----------------tokenizer------------------
    tokenizer = get_tokenizer(cfg)

    if cfg.env.verbose:
        print(f"Vocab size: {len(tokenizer.vocab)}")
        
    #-----------------model------------------
    model = get_model(cfg, tokenizer)

    #-----------------dataset------------------
    train_dataset, eval_dataset, raw_datasets, tokenized_datasets = get_datasets(cfg, tokenizer)

    # test dataset and tokenizer
    if cfg.env.debug:
        print(len(train_dataset))
        print(len(eval_dataset))
        print(tokenized_datasets)
        print(len(tokenized_datasets['train']))
        print(len(tokenized_datasets['validation']))
        print(len(tokenized_datasets['test']))

        for col in raw_datasets['train'].column_names:
            break
        text = raw_datasets['train'][col][3]
        tokens = tokenizer.tokenize(text)
        input_ids = tokenizer(text)["input_ids"]
        decoded = tokenizer.decode(input_ids)
        decoded_list = tokenizer.convert_ids_to_tokens(input_ids)
        print("Raw_text:", text)
        print("Tokens:", tokens)
        print("Input IDs:", input_ids)
        print("Decoded text:", decoded)
        print("Decoded text, nospace:", "".join(decoded_list))


    if exist_and_not_none(cfg, "trainer"):
        #-----------------optimizer------------------
        optimizer = get_optimizer(cfg, model)

        #-----------------scheduler------------------
        scheduler = get_scheduler(cfg, optimizer, train_dataset)

        if cfg.env.verbose:
            print(f"Scheduler num_training_steps: {cfg.state.num_training_steps}")

        #-----------------train------------------
        # move all below to trainer
        # using hf transformers trainer

        trainer = get_trainer(cfg, model, optimizer, scheduler, tokenizer, train_dataset, eval_dataset, tokenized_datasets, hooks=None)

        trainer.train()
        ########################################################
        model = trainer.model  # this is the trained model

    # evaluator
    if exist_and_not_none(cfg, "evaluator"):
        evaluator = get_evaluator(cfg, model, tokenizer, tokenized_datasets)
        evaluator.evaluate()
    
    # FHE inference with desilo
    if exist_and_not_none(cfg, "fhe"):
        
        pass

    # saving to huggingface hub
    if exist_and_not_none(cfg.env, "huggingface"):
        #-----------------automodel registeration------------------

        # register custom model to AutoModel -> better when planning to update to the hub
        if exist_and_not_none(cfg.env.huggingface, "model_name"):
            model_name = cfg.env.huggingface.model_name
        else:
            model_name = "placeholder"
        if cfg.state.custom_model:
            register_for_auto_classes(cfg, model_name) # enables the use of autoclasses for current model and config

        #-----------------upload to hub------------------

        # save & uploading
        os.path.join(cfg.state.output_dir, "model", model_name)
        model.save_pretrained(os.path.join(cfg.state.output_dir, "model", model_name))
        tokenizer.save_pretrained(os.path.join(cfg.state.output_dir, "model", model_name))

        repo_id = f"{os.getenv('HF_USERNAME')}/{model_name}"
        model.push_to_hub(repo_id, private=cfg.env.huggingface.private)
        tokenizer.push_to_hub(repo_id, private=cfg.env.huggingface.private)

        # model.config.save_pretrained(os.path.join(cfg.state.output_dir, "model", model_name))
        # model.push_to_hub(model_name, private=cfg.env.huggingface.private)
        # tokenizer.push_to_hub(model_name, private=cfg.env.huggingface.private)
        print('Model pushed to the hub successfully!')
    

    send_telegram_message(os.getenv("TELEGRAM_BOT_TOKEN"), f"Complete : {cfg.state.run_name}", os.getenv("TELEGRAM_CHAT_ID"))

    print("\n\nEnd of main\n\n")
    return

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc() # print the error log
        tb_string = traceback.format_exc()
        send_telegram_message(os.getenv("TELEGRAM_BOT_TOKEN"), f"Error!\n{tb_string}", os.getenv("TELEGRAM_CHAT_ID"))
