import time
from transformers import TrainerCallback

class TimeLimitCallback(TrainerCallback):
    def __init__(self, max_seconds):
        self.max_seconds = max_seconds
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() - self.start_time > self.max_seconds:
            control.should_training_stop = True
        return control





class TrainerHook_template():
    """
    Base hook class. Override the methods you need.
    """
    def on_step_end(self, trainer, **kwargs): pass
    def on_epoch_end(self, trainer, **kwargs): pass
    def on_eval_end(self, trainer, **kwargs): pass
    def on_save(self, trainer, **kwargs): pass


    