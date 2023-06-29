from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

INF = 1e9



def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch
    
    Args:
        p_m0, p_m1 (Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand

class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # general config
        self.thr = config['thr']
        self.border_rm = config['border_rm']
        # -- # for trainig fine-level LoFTR
        self.train_coarse_percent = config['train_coarse_percent']
        self.train_pad_num_gt_min = config['train_pad_num_gt_min']

        # we provide 2 options for differentiable matching
        self.match_type = config['match_type']
        if self.match_type == 'dual_softmax':
            self.temperature = config['dsmax_temperature']
        elif self.match_type == 'sinkhorn':
            try:
                from .superglue import log_optimal_transport
            except ImportError:
                raise ImportError("download superglue.py first!")
            self.log_optimal_transport = log_optimal_transport
            self.bin_score = nn.Parameter(
                Tensor(config['skh_init_bin_score'], requires_grad=True))
            self.skh_iters = config['skh_iters']
            self.skh_prefilter = config['skh_prefilter']
        else:
            raise NotImplementedError()

    def forward(self, conf_matrix:Tensor,hw0_c:List[int],hw1_c:List[int], hw0_i:List[int] ) -> Tuple[Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor]:
        """
        Args:
            data (dict)
        Update:
            data (dict): {
                'b_ids' (Tensor): [M'],
                'i_ids' (Tensor): [M'],
                'j_ids' (Tensor): [M'],
                'gt_mask' (Tensor): [M'],
                'mkpts0_c' (Tensor): [M, 2],
                'mkpts1_c' (Tensor): [M, 2],
                'mconf' (Tensor): [M]}
            NOTE: M' != M during training.
        """

        # predict coarse matches from conf_matrix
        return self.get_coarse_match(conf_matrix,hw0_c,hw1_c, hw0_i )

    def mask_border(self,m: Tensor, b: int, v:bool):
        """ Mask borders with value
        Args:
            m (): [N, H0, W0, H1, W1]
            b (int)
            v (m.dtype)
        """
        if b <= 0:
            return

        m[:, :b] = v
        m[:, :, :b] = v
        m[:, :, :, :b] = v
        m[:, :, :, :, :b] = v
        m[:, -b:] = v
        m[:, :, -b:] = v
        m[:, :, :, -b:] = v
        m[:, :, :, :, -b:] = v

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix:Tensor,hw0_c:List[int],hw1_c:List[int], hw0_i:List[int] ) ->Tuple[Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor] :
        """
        Args:
            conf_matrix (Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (Tensor): [M'],
                'i_ids' (Tensor): [M'],
                'j_ids' (Tensor): [M'],
                'gt_mask' (Tensor): [M'],
                'm_bids' (Tensor): [M],
                'mkpts0_c' (Tensor): [M, 2],
                'mkpts1_c' (Tensor): [M, 2],
                'mconf' (Tensor): [M]}
        """



        _device = conf_matrix.device
        # 1. confidence thresholding
        mask = conf_matrix > self.thr
        mask = mask.reshape(-1, hw0_c[0], hw0_c[1], hw1_c[0], hw1_c[1])
        mask = mask.to(torch.float)
        #if 'mask0' not in data:
        self.mask_border(mask, self.border_rm, False)
        #else:
        #    mask_border_with_padding(mask, self.border_rm, False,
        #                             data.mask0, data.mask1)
        mask = mask.reshape(-1, hw0_c[0] * hw0_c[1], hw1_c[0] * hw1_c[1])

        # 2. mutual nearest
        mask = mask \
            * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
            * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

        # 3. find all valid coarse matches
        # this only works when at most one `True` in each row
        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        mconf = conf_matrix[b_ids, i_ids, j_ids]

        # 4. Random sampling of training samples for fine-level LoFTR
        # (optional) pad samples with gt coarse-level matches

        # These matches select patches that feed into fine-level network

        # 4. Update with matches in original image resolution
        scale = hw0_i[0] / hw0_c[0]
        scale0 = scale
        scale1 = scale
        mkpts0_c = torch.stack(
            [i_ids % hw0_c[1], i_ids // hw0_c[1]],
            dim=1) * scale0
        mkpts1_c = torch.stack(
            [j_ids % hw1_c[1], j_ids // hw1_c[1]],
            dim=1) * scale1

        # These matches is the current prediction (for visualization)

        gt_mask = mconf == 0
        m_bids =b_ids[mconf != 0]  # mconf == 0 => gt matches
        mkpts0_c= mkpts0_c[mconf != 0]
        mkpts1_c= mkpts1_c[mconf != 0]
        mconf= mconf[mconf != 0]


        return gt_mask,m_bids,mkpts0_c,mkpts1_c,mconf,b_ids,i_ids, j_ids
