from ALIKE.alike import ALike, configs
from DFM.python.DeepFeatureMatcher import DeepFeatureMatcher


def get_matcher(device = 'cuda'):

    return DeepFeatureMatcher().eval()