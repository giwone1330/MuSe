import os
import math
from typing import Optional, List, Callable
import json
import evaluate


import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from accelerate import Accelerator
import time
from tqdm.auto import tqdm
import hydra


import os
import glob
import shutil

def cleanup_checkpoints(base_dir, prefix="epoch_", keep_last=5):
    # Find all epoch_* directories
    checkpoints = sorted(
        glob.glob(os.path.join(base_dir, f"{prefix}*")),
        key=os.path.getmtime,   # sort by modification time
        reverse=True            # newest first
    )
    
    if len(checkpoints) <= keep_last:
        return  # Nothing to delete
    
    # Keep only the latest N, remove the rest
    for old_ckpt in checkpoints[keep_last:]:
        print(f"Removing old checkpoint: {old_ckpt}")
        shutil.rmtree(old_ckpt)


class TrainerHook:
    """
    Base hook class. Override the methods you need.
    """
    def on_step_end(self, trainer, **kwargs): pass
    def on_epoch_end(self, trainer, **kwargs): pass
    def on_eval_end(self, trainer, **kwargs): pass
    def on_save(self, trainer, **kwargs): pass

    # training hooks disabled for now, should put them in different file
    # class LoggingHook(TrainerHook):
    #     def on_step_end(self, trainer, step, batch, loss):
    #         if step % trainer.cfg.training.log_every == 0:
    #             lr = trainer.scheduler.get_last_lr()[0] if trainer.scheduler else trainer.optimizer.param_groups[0]["lr"]
    #             print(f"[Step {trainer.global_step}] Loss={loss.item()*trainer.cfg.training.gradient_accumulation_steps:.4f}, LR={lr:.5e}")

    # class EvalHook(TrainerHook):
    #     def on_step_end(self, trainer, step, batch, loss):
    #         if trainer.eval_dataloader and trainer.global_step % trainer.cfg.training.eval_steps == 0:
    #             trainer.evaluate()

class CustomTrainer:
    def __init__(self,
                 model,
                 accelerator,
                 optimizer,
                 scheduler,
                 tokenizer,
                 trainer_args, # comes directly from hydra
                 cfg,
                 train_dataloader,
                 eval_dataloader = None,
                 test_dataloader = None, 
                 hooks = []):

        # These should be already instantiated

        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.trainer_args = trainer_args
        self.cfg = cfg
        self.seed = cfg.env.seed
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.hooks = hooks

        self.global_step = 0
        self.starting_epoch = 0
        self.last_model_path = None

        # best model tracking
        self.best_model_score = None
        self.best_model_epoch = None




        self.epoch = 0
        
        self.metrics = None
        if "metrics" in self.trainer_args:
            if (self.trainer_args.metrics != None) and (len(self.trainer_args.metrics)>0):
                metric_dict={}
                for metric_name, metric in self.trainer_args.metrics.items():
                    metric_obj = hydra.utils.instantiate(metric, tokenizer=self.tokenizer, _convert_="partial")
                    metric_dict[metric_name]=metric_obj
                # combine
                self.metrics = evaluate.combine(metric_dict)



        # Prepare model, optimizer, dataloaders with accelerator
        self.model, self.optimizer, self.scheduler, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, self.train_dataloader
        )
        if self.eval_dataloader:
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)
        if self.test_dataloader:
            self.test_dataloader = self.accelerator.prepare(self.test_dataloader)


    # # ----------------- save_checkpoint -----------------

    # def save_checkpoint(self, epoch):
    #     """Saves a model checkpoint."""
    #     self.accelerator.wait_for_everyone()
        
    #     # Unwrap the model to save the raw model state, not the DDP wrapper
    #     unwrapped_model = self.accelerator.unwrap_model(self.model)
        
    #     # Save on the main process only
    #     if self.accelerator.is_main_process:
    #         save_dir = os.path.join(self.cfg.state.output_dir, "ckpt", f"epoch_{epoch}")
    #         try:
    #             if hasattr(unwrapped_model, "save_pretrained"):
    #                 unwrapped_model.save_pretrained(save_dir, save_function=self.accelerator.save)
    #             else:
    #                 torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    #         except Exception as e:
    #             print("Warning: save failed:", e)

    #         # You might want to save the tokenizer as well
    #         if self.tokenizer != None:
    #             self.tokenizer.save_pretrained(save_dir)
    #         print(f"Saved model checkpoint to {save_dir}")

    # ----------------- Evaluate Loop -----------------
    def evaluate(self):
        """Runs one evaluation loop."""
        self.model.eval()
        all_predictions = []
        all_references = []

        log_dict = {}
        log_dict["val_loss"] = 0

        progress_bar = tqdm(
            range(len(self.eval_dataloader)),
            disable=not self.accelerator.is_local_main_process,
            desc=f"EVAL  | epoch {self.epoch}/{self.trainer_args.epochs}"
        )

        for step, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                outputs = self.model(**batch)
            
            # get predictions
            predictions = outputs.logits.argmax(dim=-1)
            
            # gather loss - running average
            log_dict["val_loss"] = log_dict["val_loss"] * step/(step+1) + outputs.loss.item() /(step+1)

            # Pad generated tokens and labels to the same length for batch_decode
            predicted_tokens = self.accelerator.pad_across_processes(
                predictions, dim=1, pad_index=self.tokenizer.pad_token_id
            )
            labels = batch["input_ids"]
            if labels is not None:
                labels[labels == -100] = self.tokenizer.pad_token_id
                labels = self.accelerator.pad_across_processes(
                    labels, dim=1, pad_index=self.tokenizer.pad_token_id
                )

            # Gather all predictions and references from all processes
            predicted_tokens, labels = self.accelerator.gather_for_metrics((predicted_tokens, labels))

            # Decode predictions and references
            preds = self.tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)
            refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            all_predictions.extend(preds)
            all_references.extend(refs)

            progress_bar.update(1)

        # Compute metric on the main process
        if self.accelerator.is_main_process:
            # verbose
            self.accelerator.print("\n--- Eval Examples ---")
            for i in range(min(5, len(all_predictions))):
                self.accelerator.print(f"Reference: {all_references[i]}")
                self.accelerator.print(f"Prediction: {all_predictions[i]}")
                self.accelerator.print(f"Metrics: {self.metrics.compute(predictions=[all_predictions[i]], references=[all_references[i]])}")
                self.accelerator.print("-" * 20)


            if self.metrics != None:
                metric_results = self.metrics.compute(predictions=all_predictions, references=all_references)
                log_dict.update(metric_results)
                
            self.accelerator.log(
                log_dict,
                step=self.global_step
            )
            self.accelerator.print(f"EVAL  | epoch {self.epoch}/{self.trainer_args.epochs}:", log_dict)

        # return the evaluation to the main training script
        return log_dict



    # ----------------- Training Loop -----------------
    def train(self):
        """Runs the full training loop."""

        # Resuming training #
        # We need to load the checkpoint back in before training here with `load_state`
        # The total number of epochs is adjusted based on where the state is being loaded from,
        if "resume_from_checkpoint" in self.trainer_args:
            if self.trainer_args.resume_from_checkpoint is not None or self.trainer_args.resume_from_checkpoint != "":
                self.accelerator.print(f"Resumed from checkpoint: {self.trainer_args.resume_from_checkpoint}")
                self.accelerator.load_state(self.trainer_args.resume_from_checkpoint)
                path = os.path.basename(self.trainer_args.resume_from_checkpoint)
            # else:
            #     # Get the most recent checkpoint
            #     dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            #     dirs.sort(key=os.path.getctime)
            #     path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                self.starting_epoch = int(training_difference.replace("epoch_", ""))
                self.resume_step = None
            else:
                self.resume_step = int(training_difference.replace("step_", ""))
                self.starting_epoch = self.resume_step // self.cfg.state.num_training_steps_per_epoch
                self.resume_step -= self.starting_epoch * self.cfg.state.num_training_steps_per_epoch

        # training loop
        for epoch in range(self.starting_epoch, self.trainer_args.epochs):
            self.epoch = epoch
            ################ epoch start ###################
            # measure the performance of the scratch model
            if self.trainer_args.eval_on_train_start and (epoch == 0):
                self.evaluate()

            progress_bar = tqdm(
                range(self.cfg.state.num_training_steps_per_epoch),
                disable=not self.accelerator.is_local_main_process,
                desc=f"TRAIN | epoch {epoch+1}/{self.trainer_args.epochs}"
            )
            self.model.train()

            # Skipping to designed step if possible
            if "resume_from_checkpoint" in self.trainer_args:            
                if self.trainer_args.resume_from_checkpoint and epoch == self.starting_epoch and self.resume_step is not None:
                    # We need to skip steps until we reach the resumed step only if we are not using a stateful dataloader
                    if not self.trainer_args.use_stateful_dataloader:
                        self.active_dataloader = self.accelerator.skip_first_batches(self.train_dataloader, self.resume_step)
                    else:
                        self.active_dataloader = self.train_dataloader
                    self.global_step += self.resume_step
                else:
                    # After the first iteration though, we need to go back to the original dataloader
                    self.active_dataloader = self.train_dataloader
            else:
                # if no loading is required, use original dataloader
                self.active_dataloader = self.train_dataloader
            
            for step, batch in enumerate(self.active_dataloader):
                ################ step start ###################

                # forward & backward pass on the batch, one step
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Log training loss
                    self.accelerator.log({
                        "train_loss": loss.item(),
                        "epoch": epoch
                    }, step=self.global_step)
                    
                    # Backward pass, handled by accelerate
                    self.accelerator.backward(loss)
                    
                    # Optimizer and scheduler steps
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                self.global_step += 1
                progress_bar.update(1)

                # time limit check
                if time.time()-self.cfg.state.script_start_time > self.cfg.env.system.time_limit:
                    self.accelerator.print("Reached time limit, running evaluate before termination")
                    eval_result = self.evaluate()
                    output_dir = os.path.join(self.cfg.state.output_dir, "ckpt", f"step_{self.global_step}")
                    self.accelerator.save_state(output_dir)

                    self.accelerator.end_training()
                    return # some form of info on the model (path/to/best, path/to/last)
                ################ step end ###################

            ################ epoch end ###################
            self.epoch += 1
            
            # Evaluate at the end of every epoch of the multiple of eval_iters
            if (self.epoch) % self.trainer_args.eval_interval == 0:
                eval_result = self.evaluate()

                # if best model
                if self.trainer_args.best_model_metric in eval_result:
                    eval_result_value = eval_result[self.trainer_args.best_model_metric]
                    is_better = False
                    if self.trainer_args.best_model_metric_strategy == "min":
                        is_better = (self.best_model_score == None) or (eval_result_value < self.best_model_score)
                    else: # higher
                        is_better = (self.best_model_score == None) or (eval_result_value > self.best_model_score)
                    if is_better:
                        self.best_model_score = eval_result_value
                        output_dir = os.path.join(self.cfg.state.output_dir, "ckpt", "best")
                        self.accelerator.save_state(output_dir)
                        self.accelerator.print(f"New best model with {self.trainer_args.best_model_metric}={self.best_model_score:.4f} at epoch {self.epoch}, saved to {output_dir}")
                        self.best_model_epoch = self.epoch
                        # maybe need to save info about the best model epoch/step
                    else: # if no improvement
                        if self.best_model_epoch is not None:
                            if (self.epoch - self.best_model_epoch)//self.trainer_args.eval_interval >= self.trainer_args.early_stopping_patience:
                                self.accelerator.print(f"No improvement in {self.trainer_args.best_model_metric} for {self.trainer_args.early_stopping_patience} evaluations, early stopping at epoch {self.epoch}")
                                break


            # Save model checkpoint
            output_dir = os.path.join(self.cfg.state.output_dir, "ckpt", f"epoch_{self.epoch}")
            self.accelerator.save_state(output_dir)
            self.last_model_path = output_dir
            # keep only last 5 checkpoints
            cleanup_checkpoints(os.path.join(self.cfg.state.output_dir, "ckpt"), prefix="epoch_", keep_last=5)

                
        # end training + tracker finalize
        # self.test() # tests with .generate
        return # some form of info on the model (path/to/best, path/to/last)
    
    
    def end(self):
        self.accelerator.end_training()

    def get_best_model(self):
        if self.best_model_epoch is not None:
            best_model_path = os.path.join(self.cfg.state.output_dir, "ckpt", "best")
            self.accelerator.load_state(best_model_path)
            self.accelerator.print(f"loaded best model from {best_model_path}")
            return best_model_path
        else:
            self.accelerator.print(f"failed to load best model, no best model found")
            return None
        
    def get_last_model(self):
        if self.last_model_path is not None:
            self.accelerator.load_state(self.last_model_path)
            self.accelerator.print(f"loaded last model from {self.last_model_path}")
            return self.last_model_path
        else:
            self.accelerator.print(f"failed to load last model, no last model found")
            return None

    # ----------------- Test Loop -----------------
    def test(self, prefix="test_",use_best_model = True, test_dataloader = None):
        """Runs inference and generation on the test set."""
        if test_dataloader is None:
            test_dataloader = self.test_dataloader
        if test_dataloader is None:
            self.accelerator.print("No test dataloader provided. Skipping test.")
            return
        
        if use_best_model:
            self.get_best_model()
        else: # use last model
            self.get_last_model()

        self.model.eval()
        all_predictions = []
        all_references = []

        progress_bar = tqdm(
            range(len(test_dataloader)),
            disable=not self.accelerator.is_local_main_process,
            desc="TEST  | Running generation"
        )

        for step, batch in enumerate(test_dataloader):
            with torch.no_grad():
                # It's recommended to unwrap the model for generation
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                
                # Generation config can be passed from your main config file
                # For example: **self.cfg.generation_config
                generated_tokens = unwrapped_model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **self.trainer_args.generate_args
                )

            # Pad generated tokens and labels to the same length for gathering
            generated_tokens = self.accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id
            )
            
            labels = batch["input_ids"]
            if labels is not None:
                labels[labels == -100] = self.tokenizer.pad_token_id
                labels = self.accelerator.pad_across_processes(
                    labels, dim=1, pad_index=self.tokenizer.pad_token_id
                )

            # Gather all predictions and references from all processes
            generated_tokens, labels = self.accelerator.gather_for_metrics((generated_tokens, labels))

            # Decode predictions and references
            preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            all_predictions.extend(preds)
            all_references.extend(refs)

            self.accelerator.print(f"Reference: {refs[0]}")
            self.accelerator.print(f"Prediction: {preds[0]}")
            self.accelerator.print(f"Metrics: {self.metrics.compute(predictions=preds, references=refs)}")
            self.accelerator.print("-" * 20)
            progress_bar.update(1)


        # On the main process, you can now save or process the results
        if self.accelerator.is_main_process:
            self.accelerator.print("\n--- Generation results ---")
            for i in range(len(all_predictions)):
                self.accelerator.print(f"Reference: {all_references[i]}")
                self.accelerator.print(f"Prediction: {all_predictions[i]}")
                self.accelerator.print(f"Metrics: {self.metrics.compute(predictions=[all_predictions[i]], references=[all_references[i]])}")
                self.accelerator.print("-" * 20)
            
            # You can save the results to a file here
            output_dir = self.cfg.state.output_dir
            with open(os.path.join(output_dir, f"{prefix}predictions.txt"), "w") as f:
                for pred in all_predictions:
                    f.write(pred + "\n")
            with open(os.path.join(output_dir, f"{prefix}references.txt"), "w") as f:
                for ref in all_references:
                    f.write(ref + "\n")
            self.accelerator.print(f"Saved test predictions and references to {output_dir}")

            # Compute and save metrics
            if self.metrics is not None:
                metric_results = self.metrics.compute(predictions=all_predictions, references=all_references)
                self.accelerator.print(f"TEST  | generated:", metric_results)
                test_metric_results = {f"{prefix}{k}": v for k, v in metric_results.items()}
                self.accelerator.log(
                    test_metric_results,
                    step=self.global_step
                )


        return all_predictions, all_references
    
    # ----------------- Test Loop -----------------
    def test_on_train(self, use_best_model = False, test_dataloader = None):
        """Runs inference and generation on the test set."""
        if test_dataloader is None:
            test_dataloader = self.test_dataloader
        if test_dataloader is None:
            self.accelerator.print("No test dataloader provided. Skipping test.")
            return
        
        if use_best_model:
            self.get_best_model()
        else: # use last model
            self.get_last_model()

        self.model.eval()
        all_predictions = []
        all_references = []

        progress_bar = tqdm(
            range(len(test_dataloader)),
            disable=not self.accelerator.is_local_main_process,
            desc="TEST  | Running generation"
        )

        for step, batch in enumerate(test_dataloader):
            with torch.no_grad():
                # It's recommended to unwrap the model for generation
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                
                # Generation config can be passed from your main config file
                # For example: **self.cfg.generation_config
                generated_tokens = unwrapped_model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **self.trainer_args.generate_args
                )

            # Pad generated tokens and labels to the same length for gathering
            generated_tokens = self.accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id
            )
            
            labels = batch["input_ids"]
            if labels is not None:
                labels[labels == -100] = self.tokenizer.pad_token_id
                labels = self.accelerator.pad_across_processes(
                    labels, dim=1, pad_index=self.tokenizer.pad_token_id
                )

            # Gather all predictions and references from all processes
            generated_tokens, labels = self.accelerator.gather_for_metrics((generated_tokens, labels))

            # Decode predictions and references
            preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            all_predictions.extend(preds)
            all_references.extend(refs)

            self.accelerator.print(f"Reference: {refs[0]}")
            self.accelerator.print(f"Prediction: {preds[0]}")
            self.accelerator.print(f"Metrics: {self.metrics.compute(predictions=preds, references=refs)}")
            self.accelerator.print("-" * 20)
            progress_bar.update(1)


        # On the main process, you can now save or process the results
        if self.accelerator.is_main_process:
            self.accelerator.print("\n--- Generation results ---")
            for i in range(len(all_predictions)):
                self.accelerator.print(f"Reference: {all_references[i]}")
                self.accelerator.print(f"Prediction: {all_predictions[i]}")
                self.accelerator.print(f"Metrics: {self.metrics.compute(predictions=[all_predictions[i]], references=[all_references[i]])}")
                self.accelerator.print("-" * 20)
            
            # You can save the results to a file here
            output_dir = self.cfg.state.output_dir
            with open(os.path.join(output_dir, "test_predictions.txt"), "w") as f:
                for pred in all_predictions:
                    f.write(pred + "\n")
            with open(os.path.join(output_dir, "test_references.txt"), "w") as f:
                for ref in all_references:
                    f.write(ref + "\n")
            self.accelerator.print(f"Saved test predictions and references to {output_dir}")

            # Compute and save metrics
            if self.metrics is not None:
                metric_results = self.metrics.compute(predictions=all_predictions, references=all_references)
                self.accelerator.print(f"TEST  | generated:", metric_results)
                test_metric_results = {f"test_{k}": v for k, v in metric_results.items()}
                self.accelerator.log(
                    test_metric_results,
                    step=self.global_step
                )


        return all_predictions, all_references
    
































class CustomTrainer_bak2:
    def __init__(self,
                 model,
                 accelerator,
                 optimizer,
                 scheduler,
                 tokenizer,
                 trainer_args, # comes directly from hydra
                 cfg,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader = None, 
                 hooks = []):

        # These should be already instantiated

        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.trainer_args = trainer_args
        self.cfg = cfg
        self.seed = cfg.env.seed
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.hooks = hooks

        self.global_step = 0
        self.starting_epoch = 0
        
        

        self.metrics = None
        if "metrics" in self.trainer_args:
            if self.trainer_args.metrics != None:
                self.metrics = hydra.utils.instantiate(self.cfg.trainer_args.metrics)


        # Prepare model, optimizer, dataloaders with accelerator
        self.model, self.optimizer, self.scheduler, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, self.train_dataloader
        )
        if self.eval_dataloader:
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)


    # ----------------- Evaluate Loop -----------------
    def save_checkpoint(self, epoch):
        """Saves a model checkpoint."""
        self.accelerator.wait_for_everyone()
        
        # Unwrap the model to save the raw model state, not the DDP wrapper
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # Save on the main process only
        if self.accelerator.is_main_process:
            save_dir = os.path.join(self.cfg.state.output_dir, "ckpt", f"epoch_{epoch}")
            try:
                if hasattr(unwrapped_model, "save_pretrained"):
                    unwrapped_model.save_pretrained(save_dir, save_function=self.accelerator.save)
                else:
                    torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
            except Exception as e:
                print("Warning: save failed:", e)

            # You might want to save the tokenizer as well
            if self.tokenizer != None:
                self.tokenizer.save_pretrained(save_dir)
            print(f"Saved model checkpoint to {save_dir}")

    # ----------------- Evaluate Loop -----------------
    def evaluate(self, epoch):
        """Runs one evaluation loop."""
        self.model.eval()

        log_dict = {}
        log_dict["val_loss"] = 0

        progress_bar = tqdm(
            range(len(self.eval_dataloader)),
            disable=not self.accelerator.is_local_main_process,
            desc=f"EVAL  | epoch {epoch+1}/{self.trainer_args.epochs}"
        )

        for step, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                outputs = self.model(**batch)
            
            # gather loss - running average
            log_dict["val_loss"] = log_dict["val_loss"] * step/(step+1) + outputs.loss.item() /(step+1)

            if self.eval_metrics != None:
                # gather predictions
                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = self.accelerator.gather((predictions, batch["labels"]))

                self.eval_metrics.add_batch(
                    predictions=predictions,
                    references=references,
                )

            progress_bar.update(1)

        # Compute metric on the main process
        if self.accelerator.is_main_process:
            if self.eval_metrics != None:
                metric_results = self.eval_metric.compute()
                log_dict.update(metric_results)
                

            self.accelerator.log(
                log_dict,
                step=self.global_step
            )
            print(f"Epoch {epoch} | logged {log_dict}")
        
        # Save model checkpoint
        self.save_checkpoint(epoch)


        """Runs one evaluation loop."""
        # the argument epoch represents the actual number of epochs done
        self.model.eval()
        all_predictions = []
        all_references = []
        log_dict = {}
        log_dict["val_loss"] = 0

        progress_bar = tqdm(
            range(len(self.eval_dataloader)),
            disable=not self.accelerator.is_local_main_process,
            desc=f"EVAL  | epoch {epoch}/{self.trainer_args.epochs}"
        )

        for step, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                outputs = self.model(**batch)
            
            # gather loss - running average
            log_dict["val_loss"] = log_dict["val_loss"] * step/(step+1) + outputs.loss.item() /(step+1)

            predictions = outputs.logits.argmax(dim=-1)

            # Pad generated tokens and labels to the same length for gathering
            predicted_tokens = self.accelerator.pad_across_processes(
                predictions, dim=1, pad_index=self.tokenizer.pad_token_id
            )
            
            labels = batch["labels"]
            if labels is not None:
                labels = self.accelerator.pad_across_processes(
                    labels, dim=1, pad_index=self.tokenizer.pad_token_id
                )

            # Gather all predictions and references from all processes
            predicted_tokens, labels = self.accelerator.gather_for_metrics((predicted_tokens, labels))

            # Decode predictions and references
            preds = self.tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)
            refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # all_predictions.append(gathered_predictions.cpu())
            # all_references.append(gathered_references.cpu())
            all_predictions.extend(preds)
            all_references.extend(refs)

            progress_bar.update(1)


        # Compute metric on the main process
        if self.accelerator.is_main_process:
            if self.metrics != None:
                metric_results = self.metrics.compute()
                log_dict.update(metric_results)
                

            self.accelerator.log(
                log_dict,
                step=self.global_step
            )
            print(f"Epoch {epoch} | logged {log_dict}")
        
        # Save model checkpoint
        self.save_checkpoint(epoch)


    # ----------------- Training Loop -----------------
    def train(self):
        """Runs the full training loop."""
        # progress_bar = tqdm(
        #     range(self.cfg.state.num_training_steps),
        #     disable=not self.accelerator.is_local_main_process
        # )


        # Resuming training #
        # We need to load the checkpoint back in before training here with `load_state`
        # The total number of epochs is adjusted based on where the state is being loaded from,
        # as we assume continuation of the same training script
        if "resume_from_checkpoint" in self.cfg.trainer_args:
            if self.trainer_args.resume_from_checkpoint is not None or self.trainer_args.resume_from_checkpoint != "":
                self.accelerator.print(f"Resumed from checkpoint: {self.trainer_args.resume_from_checkpoint}")
                self.accelerator.load_state(self.trainer_args.resume_from_checkpoint)
                path = os.path.basename(self.trainer_args.resume_from_checkpoint)
            # else:
            #     # Get the most recent checkpoint
            #     dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            #     dirs.sort(key=os.path.getctime)
            #     path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                self.starting_epoch = int(training_difference.replace("epoch_", ""))
                self.resume_step = None
            else:
                self.resume_step = int(training_difference.replace("step_", ""))
                self.starting_epoch = self.resume_step // self.cfg.state.num_training_steps_per_epoch
                self.resume_step -= self.starting_epoch * self.cfg.state.num_training_steps_per_epoch

        for epoch in range(self.starting_epoch, self.trainer_args.epochs):
            ################ epoch start ###################
            if self.trainer_args.eval_on_train_start and (epoch == 0):
                self.evaluate(0)

            progress_bar = tqdm(
                range(self.cfg.state.num_training_steps_per_epoch),
                disable=not self.accelerator.is_local_main_process,
                desc=f"TRAIN | epoch {epoch+1}/{self.trainer_args.epochs}"
            )
            self.model.train()

            if self.trainer_args.resume_from_checkpoint and epoch == self.starting_epoch and self.resume_step is not None:
                # We need to skip steps until we reach the resumed step only if we are not using a stateful dataloader
                if not self.trainer_args.use_stateful_dataloader:
                    self.active_dataloader = self.accelerator.skip_first_batches(self.train_dataloader, self.resume_step)
                else:
                    self.active_dataloader = self.train_dataloader
                self.global_step += self.resume_step
            
            for step, batch in enumerate(self.active_dataloader):
                ################ step start ###################

                # The `accelerate.accumulate` context manager handles gradient accumulation
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Log training loss
                    self.accelerator.log({
                        "train_loss": loss.item(),
                        "epoch": epoch
                    }, step=self.global_step)
                    
                    # Backward pass, handled by accelerate
                    self.accelerator.backward(loss)
                    
                    # Optimizer and scheduler steps
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()


                self.global_step += 1
                progress_bar.update(1)

                if time.time()-self.cfg.state.script_start_time > self.cfg.env.system.time_limit:
                    self.accelerator.print("Reached time limit, running evaluate before termination")
                    self.evaluate(epoch)
                    self.accelerator.end_training()
                    return
                ################ step end ###################

            # Evaluate at the end of every epoch of the multiple of eval_iters
            if (epoch+1) % self.trainer_args.eval_interval == 0:
                self.evaluate(epoch+1)

            ################ epoch end ###################

        # end training + tracker finalize
        self.accelerator.end_training()
        return

    # ----------------- Test Loop -----------------
    def test(self):
        """Runs inference and generation on the test set."""
        if self.test_dataloader is None:
            self.accelerator.print("No test dataloader provided. Skipping test.")
            return

        self.model.eval()
        all_predictions = []
        all_references = []

        progress_bar = tqdm(
            range(len(self.test_dataloader)),
            disable=not self.accelerator.is_local_main_process,
            desc="TEST  | Running generation"
        )

        for step, batch in enumerate(self.test_dataloader):
            with torch.no_grad():
                # It's recommended to unwrap the model for generation
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                
                # Generation config can be passed from your main config file
                # For example: **self.cfg.generation_config
                generated_tokens = unwrapped_model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=50  # Example max length
                )

            # Pad generated tokens and labels to the same length for gathering
            generated_tokens = self.accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id
            )
            
            labels = batch["labels"]
            if labels is not None:
                labels = self.accelerator.pad_across_processes(
                    labels, dim=1, pad_index=self.tokenizer.pad_token_id
                )

            # Gather all predictions and references from all processes
            generated_tokens, labels = self.accelerator.gather_for_metrics((generated_tokens, labels))

            # Decode predictions and references
            preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            all_predictions.extend(preds)
            all_references.extend(refs)

            progress_bar.update(1)

        # On the main process, you can now save or process the results
        if self.accelerator.is_main_process:
            print("\n--- Generation Examples ---")
            for i in range(min(5, len(all_predictions))):
                print(f"Reference: {all_references[i]}")
                print(f"Prediction: {all_predictions[i]}")
                print("-" * 20)
            
            # You can save the results to a file here
            output_dir = self.cfg.state.output_dir
            with open(os.path.join(output_dir, "test_predictions.txt"), "w") as f:
                for pred in all_predictions:
                    f.write(pred + "\n")
            with open(os.path.join(output_dir, "test_references.txt"), "w") as f:
                for ref in all_references:
                    f.write(ref + "\n")
            print(f"Saved test predictions and references to {output_dir}")

            # Compute and save metrics
            if self.metrics is not None:
                import json
                metric_results = self.metrics.compute(predictions=all_predictions, references=all_references)
                print("\n--- Test Metrics ---")
                # Pretty print the metrics
                print(json.dumps(metric_results, indent=4))

                # Save metrics to a json file
                with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
                    json.dump(metric_results, f, indent=4)
                print(f"Saved test metrics to {output_dir}")


        return all_predictions, all_references
        




























#!################################################################
class CustomTrainer_backup:
    def __init__(self,
                 model,
                 accelerator,
                 optimizer,
                 scheduler,
                 tokenizer,
                 trainer_args, # comes directly from hydra
                 cfg,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader = None, 
                 hooks = []):

        # These should be already instantiated

        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.trainer_args = trainer_args
        self.cfg = cfg
        self.seed = cfg.env.seed
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.hooks = hooks

        self.global_step = 0
        

        self.eval_metrics = None
        if "metrics" in self.trainer_args:
            if self.trainer_args.metrics != None:
                self.eval_metrics = hydra.utils.instantiate(self.cfg.trainer_args.metrics)


        # Prepare model, optimizer, dataloaders with accelerator
        self.model, self.optimizer, self.scheduler, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, self.train_dataloader
        )
        if self.eval_dataloader:
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)


    # ----------------- Evaluate Loop -----------------
    def save_checkpoint(self, epoch):
        """Saves a model checkpoint."""
        self.accelerator.wait_for_everyone()
        
        # Unwrap the model to save the raw model state, not the DDP wrapper
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # Save on the main process only
        if self.accelerator.is_main_process:
            save_dir = os.path.join(self.cfg.state.output_dir, "ckpt", f"epoch_{epoch}")
            try:
                if hasattr(unwrapped_model, "save_pretrained"):
                    unwrapped_model.save_pretrained(save_dir, save_function=self.accelerator.save)
                else:
                    torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
            except Exception as e:
                print("Warning: save failed:", e)

            # You might want to save the tokenizer as well
            if self.tokenizer != None:
                self.tokenizer.save_pretrained(save_dir)
            print(f"Saved model checkpoint to {save_dir}")

    # ----------------- Evaluate Loop -----------------
    def evaluate(self, epoch):
        """Runs one evaluation loop."""
        self.model.eval()

        log_dict = {}
        log_dict["val_loss"] = 0

        progress_bar = tqdm(
            range(len(self.eval_dataloader)),
            disable=not self.accelerator.is_local_main_process,
            desc=f"EVAL  | epoch {epoch+1}/{self.trainer_args.epochs}"
        )

        for step, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                outputs = self.model(**batch)
            
            # gather loss - running average
            log_dict["val_loss"] = log_dict["val_loss"] * step/(step+1) + outputs.loss.item() /(step+1)

            if self.eval_metrics != None:
                # gather predictions
                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = self.accelerator.gather((predictions, batch["labels"]))

                self.eval_metrics.add_batch(
                    predictions=predictions,
                    references=references,
                )

            progress_bar.update(1)


        # Compute metric on the main process
        if self.accelerator.is_main_process:
            if self.eval_metrics != None:
                metric_results = self.eval_metric.compute()
                log_dict.update(metric_results)
                

            self.accelerator.log(
                log_dict,
                step=self.global_step
            )
            print(f"Epoch {epoch} | logged {log_dict}")
        
        # Save model checkpoint
        self.save_checkpoint(epoch)


    # ----------------- Training Loop -----------------
    def train(self):
        """Runs the full training loop."""
        # progress_bar = tqdm(
        #     range(self.cfg.state.num_training_steps),
        #     disable=not self.accelerator.is_local_main_process
        # )

        for epoch in range(self.trainer_args.epochs):
            if self.trainer_args.eval_on_train_start and (epoch == 0):
                self.evaluate(0)

            progress_bar = tqdm(
                range(self.cfg.state.num_training_steps_per_epoch),
                disable=not self.accelerator.is_local_main_process,
                desc=f"TRAIN | epoch {epoch+1}/{self.trainer_args.epochs}"
            )
            self.model.train()
            
            for step, batch in enumerate(self.train_dataloader):
                # The `accelerate.accumulate` context manager handles gradient accumulation
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Log training loss
                    self.accelerator.log({
                        "train_loss": loss.item(),
                        "epoch": epoch
                    }, step=self.global_step)
                    
                    # Backward pass, handled by accelerate
                    self.accelerator.backward(loss)
                    
                    # Optimizer and scheduler steps
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()



                self.global_step += 1
                progress_bar.update(1)

                if time.time()-self.cfg.state.script_start_time > self.cfg.env.system.time_limit:
                    self.accelerator.print("Reached time limit, running evaluate before termination")
                    self.evaluate(epoch)
                    self.accelerator.end_training()
                    return

            # Evaluate at the end of every epoch of the multiple of eval_iters
            if (epoch+1) % self.trainer_args.eval_interval == 0:
                self.evaluate(epoch+1)

        # end training + tracker finalize
        self.accelerator.end_training()
        return
