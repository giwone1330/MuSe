import os
import math
from typing import Optional, List, Callable

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from accelerate import Accelerator
import time
from tqdm.auto import tqdm
import hydra


class EquationTester():
    def __init__(self,
                 model,
                 accelerator,
                 tokenizer,
                 cfg,
                 tester_args,
                 test_dataloader=None):
        
        # These should be already instantiated

        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator


        self.tester_args = tester_args
        self.cfg = cfg
        self.seed = cfg.env.seed
        self.test_dataloader = test_dataloader

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
        log_dict["epoch"] = epoch
        log_dict["val_loss"] = 0

        progress_bar = tqdm(
            range(len(self.eval_dataloader)),
            disable=not self.accelerator.is_local_main_process,
            desc=f"EVAL  | Epoch {epoch+1}/{self.trainer_args.epochs}"
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
    def test(self):

        progress_bar = tqdm(
            range(self.cfg.state.num_training_steps_per_epoch),
            disable=not self.accelerator.is_local_main_process,
            desc=f"TEST | "
        )
        self.model.eval()
        
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



        # end training + tracker finalize
        self.accelerator.end_training()
        return