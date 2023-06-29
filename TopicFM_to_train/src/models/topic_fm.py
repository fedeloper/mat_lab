from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .backbone import build_backbone
from .modules import LocalFeatureTransformer, FinePreprocess, TopicFormer
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching


class TopicFM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone(config)
        self.loftr_coarse = TopicFormer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching()


    def forward(self, image0,image1)-> Tuple[Tensor,Tensor,Tensor,Tensor]:
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        #image0 , image1 = torch.unsqueeze(images[:,0,:,:], 1),torch.unsqueeze(images[:,1,:,:], 1)

        bs = image0.size(0)
        hw0_i= image0.shape[2:]
        hw1_i = image1.shape[2:]


        if hw0_i == hw1_i:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(torch.cat([image0, image1], dim=0))

            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(bs), feats_f.split(bs)
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(image0), self.backbone(image1)


        hw0_c= feat_c0.shape[2:]
        hw1_c= feat_c1.shape[2:]
        hw0_f= feat_f0.shape[2:]
        hw1_f= feat_f1.shape[2:]


        # 2. coarse-level loftr module
        feat_c0 = feat_c0.permute(0, 2, 3, 1).reshape(feat_c0.shape[0], -1, feat_c0.shape[1])
        feat_c1 = feat_c1.permute(0, 2, 3, 1).reshape(feat_c1.shape[0], -1, feat_c1.shape[1])

        mask_c0 = mask_c1 = None  # mask is useful in training


        feat_c0, feat_c1, conf_matrix, topic_matrix = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)


        # 3. match coarse-level
        gt_mask,m_bids,mkpts0_c,mkpts1_c,mconf,b_ids,i_ids, j_ids = self.coarse_matching(conf_matrix,hw0_c,hw1_c, hw0_i)

        # 4. fine-level refinement

        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, hw0_f, hw0_c, b_ids, i_ids, j_ids)


        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)


        # 5. match fine-level
        mk0,mk1 = self.fine_matching(feat_f0_unfold, feat_f1_unfold,hw0_i,hw0_f,mkpts0_c,mkpts1_c,mconf)


        return (mk0,mk1,mconf,m_bids)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
