"""
Select the NN module to use.
Cannot be in .common because of circular imports.
"""
from . import bnn_mcd, rnn
from .. import constants as c

nns = {c.bnn_mcd: bnn_mcd,
       c.rnn: rnn,
       }

def select_nn(name: str | c.Parameter):
    return nns[name]
