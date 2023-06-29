from ALIKE.alike import ALike, configs


def get_matcher(device = 'cuda'):

    return  ALike(**configs["alike-t"],
           device=device,
           top_k=-1,
           scores_th=0.2,
           n_limit=5000).eval()