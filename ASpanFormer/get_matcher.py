import pytorch_lightning as pl

from .src.utils.profiler import build_profiler

from .src.lightning.lightning_aspanformer import PL_ASpanFormer

from .src.config.default import get_cfg_defaults

def get_matcher():

    print("hello")
    #args = parse_args()
    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    #config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility

    # tune when testing
    threshold = None
    if threshold is not None:
        config.LOFTR.MATCH_COARSE.THR = threshold

    # lightning module
    ckpt_path="weights/outdoor.ckpt"
    profiler = build_profiler("inference")
    matcher = PL_ASpanFormer(config, pretrained_ckpt="../ASpanFormer/weights/outdoor.ckpt", dump_dir=".",profiler=profiler)



    return matcher.cuda().eval()