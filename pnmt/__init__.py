""" Main entry point of the ONMT library """
import pnmt.inputters
import pnmt.encoders
import pnmt.decoders
import pnmt.models
import pnmt.utils
import pnmt.modules
from pnmt.trainer import Trainer
import sys
import pnmt.utils.optimizers
pnmt.utils.optimizers.Optim = pnmt.utils.optimizers.Optimizer
sys.modules["onmt.Optim"] = pnmt.utils.optimizers

# For Flake
__all__ = [pnmt.inputters, pnmt.encoders, pnmt.decoders, pnmt.models,
           pnmt.utils, pnmt.modules, "Trainer"]

__version__ = "2.2.0"
