from ..trainer.algs.MEND import MEND
from ..trainer.algs.SERAC import SERAC_MULTI
from ..trainer.algs.ft import FT
from ..trainer.algs.OURS import OURS
from ..trainer.algs.WISE import WISEMultimodal

SERAC = SERAC_MULTI


ALG_TRAIN_DICT = {
    'MEND': MEND,
    'SERAC': SERAC,
    'SERAC_MULTI': SERAC_MULTI,
    'FT': FT,
    'ft': FT,
    'lora': FT,
    'LORA': FT,
    'OURS': OURS,
    'WISE': WISEMultimodal
}
