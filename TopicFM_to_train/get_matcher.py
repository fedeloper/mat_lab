from .src.config.default import get_cfg_defaults
import pytorch_lightning as pl
from .src.lightning_trainer.trainer import PL_Trainer
def get_matcher():
    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    # tune when testing
    threshold = None
    if threshold is not None:
        config.LOFTR.MATCH_COARSE.THR = threshold
    # lightning module
    matcher = PL_Trainer(config, pretrained_ckpt="../TopicFM/pretrained/model_best.ckpt")

    return matcher.eval()