# PILOT: stub. The released quiet-star-train.py imports this module, which is not in the repo; the
# callback is imported but never used, so an empty TrainerCallback keeps the import working.
from transformers import TrainerCallback


class EarlyStoppingCallback(TrainerCallback):
    pass
