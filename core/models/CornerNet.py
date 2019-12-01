import torch
import torch.nn as nn

from .py_utils import TopPool, BottomPool, LeftPool, RightPool

from .py_utils.utils import convolution, residual, corner_pool
from .py_utils.losses import CornerNet_Loss
from .py_utils.modules import hg_module, hg, hg_net


# 做池化层
def make_pool_layer(dim):
    return nn.Sequential()


# 做hg层 [看样子和残差有关],modules是模块数目
def make_hg_layer(inp_dim, out_dim, modules):
    layers = [residual(inp_dim, out_dim, stride=2)]
    layers += [residual(out_dim, out_dim) for _ in range(1, modules)]
    return nn.Sequential(*layers)


class model(hg_net):
    def _pred_mod(self, dim):
        return nn.Sequential(
            convolution(3, 256, 256, with_bn=False),
            nn.Conv2d(256, dim, (1, 1))
        )

    def _merge_mod(self):
        return nn.Sequential(
            nn.Conv2d(256, 256, (1, 1), bias=False),
            nn.BatchNorm2d(256)
        )

    def __init__(self):
        stacks = 2
        pre = nn.Sequential(
            # 为啥卷积核选择7啊
            convolution(7, 3, 128, stride=2),
            residual(128, 256, stride=2)
        )
        hg_mods = nn.ModuleList([
            hg_module(
                5, [256, 256, 384, 384, 384, 512], [2, 2, 2, 2, 2, 4],
                # 这里初始化了一个空nn序列
                make_pool_layer=make_pool_layer,
                # 赋值上面的函数
                make_hg_layer=make_hg_layer
            ) for _ in range(stacks)
        ])
        # 依次出n个卷基层,n-1残差层
        cnvs = nn.ModuleList([convolution(3, 256, 256) for _ in range(stacks)])
        inters = nn.ModuleList([residual(256, 256) for _ in range(stacks - 1)])
        # n-1 归一化层
        cnvs_ = nn.ModuleList([self._merge_mod() for _ in range(stacks - 1)])
        inters_ = nn.ModuleList([self._merge_mod() for _ in range(stacks - 1)])

        # 上面几个模块我们依次放下来做forward
        hgs = hg(pre, hg_mods, cnvs, inters, cnvs_, inters_)

        # 新建细节的模块[corner的池化层]
        tl_modules = nn.ModuleList([corner_pool(256, TopPool, LeftPool) for _ in range(stacks)])
        br_modules = nn.ModuleList([corner_pool(256, BottomPool, RightPool) for _ in range(stacks)])

        tl_heats = nn.ModuleList([self._pred_mod(80) for _ in range(stacks)])
        br_heats = nn.ModuleList([self._pred_mod(80) for _ in range(stacks)])
        for tl_heat, br_heat in zip(tl_heats, br_heats):
            torch.nn.init.constant_(tl_heat[-1].bias, -2.19)
            torch.nn.init.constant_(br_heat[-1].bias, -2.19)

        tl_tags = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])
        br_tags = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])

        tl_offs = nn.ModuleList([self._pred_mod(2) for _ in range(stacks)])
        br_offs = nn.ModuleList([self._pred_mod(2) for _ in range(stacks)])

        super(model, self).__init__(
            hgs, tl_modules, br_modules, tl_heats, br_heats,
            tl_tags, br_tags, tl_offs, br_offs
        )

        self.loss = CornerNet_Loss(pull_weight=1e-1, push_weight=1e-1)
