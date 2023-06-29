import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FinePreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.cat_c_feat = config['fine_concat_coarse_feat']
        self.W = int((self.config['fine_window_size']))
        self.kernel_ = (self.W,self.W)
        self.padding_ = self.floordiv(self.W,2)
        d_model_c = self.config['coarse']['d_model']
        d_model_f = self.config['fine']['d_model']
        self.d_model_f = d_model_f
        #if self.cat_c_feat:
        #    self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
         #   self.merge_feat = nn.Linear(2*d_model_f, d_model_f, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def square(self,a:int)->int:
        return int(pow(a,2))
    def floordiv(self,a:int,b:int)->int:
        return int(a//b)
    def forward(self, feat_f0:Tensor, feat_f1:Tensor, hw0_f:List[int],hw0_c:List[int],b_ids:Tensor,i_ids:Tensor,j_ids:Tensor) -> Tuple[Tensor,Tensor]:

        W = self.W
        stride = 4#self.floordiv(hw0_f[0], hw0_c[0])

        if b_ids.shape[0] == 0:
            feat0 = torch.empty(0, self.square(W), self.d_model_f, device=feat_f0.device)
            feat1 = torch.empty(0, self.square(W), self.d_model_f, device=feat_f0.device)
            return feat0, feat1

        # 1. unfold(crop) all local windows

        feat_f0_unfold = F.unfold(feat_f0, kernel_size=self.kernel_, stride=stride, padding=self.padding_)
        n, _, l = feat_f0_unfold.shape  # get n and l from the shape of feat_f0_unfold
        c = _ // (self.square(W))
        feat_f0_unfold = feat_f0_unfold.permute(0, 2, 1)  # shape: [n, l, c*ww]
        feat_f0_unfold = feat_f0_unfold.reshape(n, l, self.square(W), c)  # shape: [n, l, ww, c]
        #feat_f0_unfold = feat_f0_unfold.permute(0, 2, 1, 3)  # shape: [n, ww, l, c]

        feat_f1_unfold = F.unfold(feat_f1, kernel_size=self.kernel_, stride=stride, padding=self.padding_)

        n, _, l = feat_f1_unfold.shape  # get n and l from the shape of feat_f0_unfold
        c = _ // (self.square(W))
        feat_f1_unfold = feat_f1_unfold.permute(0, 2, 1)  # shape: [n, l, c*ww]
        feat_f1_unfold = feat_f1_unfold.reshape(n, l, self.square(W), c)  # shape: [n, l, ww, c]
        #feat_f1_unfold = feat_f1_unfold.permute(0, 2, 1, 3)  # shape: [n, ww, l, c]



        # 2. select only the predicted matches
        feat_f0_unfold = feat_f0_unfold[b_ids, i_ids]  # [n, ww, cf]
        feat_f1_unfold = feat_f1_unfold[b_ids, j_ids]

        # option: use coarse-level feature as context: concat and linear
        #if self.cat_c_feat:
        #    feat_c_win = self.down_proj(torch.cat([feat_c0[b_ids, i_ids],
        #                                           feat_c1[b_ids, j_ids]], 0))  # [2n, c]
        #    feat_cf_win = self.merge_feat(torch.cat([
        #        torch.cat([feat_f0_unfold, feat_f1_unfold], 0),  # [2n, ww, cf]
        #        feat_c_win.unsqueeze(1).repeat(1, self.square(W), 1),  # [2n, ww, cf]
        #    ], -1))
        #    feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)

        return feat_f0_unfold, feat_f1_unfold
