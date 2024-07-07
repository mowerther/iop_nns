"""
Select the NN module to use.
Cannot be in .common because of circular imports.
"""
from .bnn_dc import BNN_DC
from .bnn_mcd import BNN_MCD
from .ens import Ensemble
from .rnn import RNN_MCD
from .. import constants as c

nns = {c.bnn_dc: BNN_DC,
       c.bnn_mcd: BNN_MCD,
       c.ensemble: Ensemble,
       c.rnn: RNN_MCD,}

def select_nn(name: str | c.Parameter):
    return nns[name]
