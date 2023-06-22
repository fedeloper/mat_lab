


from MatchFormer.config.defaultmf import get_cfg_defaults
from MatchFormer.model.lightning_loftr import PL_LoFTR



import pytorch_lightning as pl

def get_matcher():
    config = get_cfg_defaults()
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility

    matcher = PL_LoFTR(config, pretrained_ckpt="../MatchFormer/weights/outdoor-lite-SEA.ckpt", dump_dir=".")

    return matcher.eval()
