import os
import builtins
import logging as base_logging
from nemo.utils import logging

def do_nothing(*args, **kwargs):
    pass

builtins.print = do_nothing

logging.setLevel(logging.ERROR)
base_logging.getLogger("lightning").setLevel(base_logging.ERROR)
base_logging.getLogger().setLevel(base_logging.ERROR)


def get_rank():
    return int(os.getenv("SLURM_PROCID", 0))

class IgnoreRegexPatternFilter(base_logging.Filter):
    def __init__(self, patterns):
        self.patterns = patterns

    def filter(self, record):
        message = record.getMessage()
        return not any(pattern in message for pattern in self.patterns)

base_logging.getLogger().addFilter(IgnoreRegexPatternFilter(["Loaded ViT-H-14 model config.", "Loading pretrained ViT-H-14 weights (laion2b_s32b_b79k)."]))
