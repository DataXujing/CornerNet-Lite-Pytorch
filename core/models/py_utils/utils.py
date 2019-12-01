import torch
import torch.nn as nn

# # 根据 ind 和 mask 提取 feature
def _gather_feat(feat, ind, mask=None):
    """
        获取特征函数
        @param 这里我们
    """
    # 维度[要是下面传过来的话是最后一个维度]的大小
    dim = feat.size(2)
    # index 也添加了最后的维度，再拓展到上面的dim的维度上
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


# NMS——非极大值抑制 #heatmap上利用maxpool等效实现nms
def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


# 传入的是tl_regr和ind 转置并获取特征
def _tranpose_and_gather_feat(feat, ind):
    # permute将维度换位 contiguous用于整理内存，之后才能使用view[虽然现在能直接用reshape换位置了]
    feat = feat.permute(0, 2, 3, 1).contiguous()
    # 重整了特征视图
    feat = feat.view(feat.size(0), -1, feat.size(3))
    # 看看获取的是啥特征吧
    feat = _gather_feat(feat, ind)
    return feat


# 在heatmap上提取topk corners
def _topk(scores, K=20):
    batch, cat, height, width = scores.size()

    # # 在batch的每张图上挑选出topK corners，topk_scores指示某点分数，topk_inds指示某点序号(index)
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int() # corner 类别，feature map的通道数=类别数

    topk_inds = topk_inds % (height * width) #映射到feature map的通道上
    topk_ys = (topk_inds / width).int().float()  #y坐标
    topk_xs = (topk_inds % width).int().float()  #x坐标
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


# 解码 话说这里tl可能代表左上角吧，那相对的br就是右下角
"""大致总结一下decode的流程： 
对tl_heat and br_heat依次做sigmoid、nms、tokk得到tl_scores, tl_inds, tl_clses, tl_ys, tl_xs(same as br_...);
利用tl_inds来获得tl_regr，再利用tl_regr来调整tl_ys and tl_xs(same as br_...);
有了更新后的tl_ys, tl_xs, br_ys, br_xs，便可以得到bboxes;
利用tl_inds来获得tl_tag，利用br_inds来获得br_tag，此时就可以计算L1距离了;
利用公式 scores = (tl_scores + br_scores) / 2 计算scores;
将不符合条件的scores都设为-1（包括不是一类的、L1距离大于阈值的、右下角点坐标小于左上角点的）;
挑选scores最大的num_dets个框保留下来，并取出它们的bboxes, scores, tl_scores, br_scores, clses返回。
"""
def _decode(
        tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr,
        K=100, kernel=1, ae_threshold=1, num_dets=1000, no_border=False
):
    batch, cat, height, width = tl_heat.size()

    # 使用的激励函数sigmoid
    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)

    # perform nms on heatmaps
    # 然后进行非极大值抑制[池化]
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)

    # 获得其中topK个的分数、等与坐标
    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)

    # 拓展坐标
    tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
    tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
    br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
    br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)

    # 如果没有边界[默认False] ，那么考虑其中tl在整个左上角和br在整个右下角的情况
    if no_border:
        tl_ys_binds = (tl_ys == 0)
        tl_xs_binds = (tl_xs == 0)
        br_ys_binds = (br_ys == height - 1)
        br_xs_binds = (br_xs == width - 1)

    # 这个可能是说有回归的情况
    if tl_regr is not None and br_regr is not None:
        # 为啥要颠倒啊
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
        tl_regr = tl_regr.view(batch, K, 1, 2)
        br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
        br_regr = br_regr.view(batch, 1, K, 2)

        tl_xs = tl_xs + tl_regr[..., 0]
        tl_ys = tl_ys + tl_regr[..., 1]
        br_xs = br_xs + br_regr[..., 0]
        br_ys = br_ys + br_regr[..., 1]

    # all possible boxes based on top k corners (ignoring class)
    # 所有可能的框都基于最顶的k的顶点
    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)

    # 拿了啥特征回来啊 tag？？？？？
    tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)
    tl_tag = tl_tag.view(batch, K, 1)
    br_tag = _tranpose_and_gather_feat(br_tag, br_inds)
    br_tag = br_tag.view(batch, 1, K)

    # 这里对两个tag做差求绝对值了
    dists = torch.abs(tl_tag - br_tag)

    # 将得分拓展到K,K的范围
    tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
    br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)
    scores = (tl_scores + br_scores) / 2

    # reject boxes based on classes
    # 这边要只取那些不一样的
    tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)
    br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)
    cls_inds = (tl_clses != br_clses)

    # reject boxes based on distances
    # 上面的绝对值小于阈值的扔了扔了
    dist_inds = (dists > ae_threshold)

    # reject boxes based on widths and heights
    # 右下点肯定要比左上点的右下一些吧
    width_inds = (br_xs < tl_xs)
    height_inds = (br_ys < tl_ys)

    # 没有边界的话，边界那边设成-1
    if no_border:
        scores[tl_ys_binds] = -1
        scores[tl_xs_binds] = -1
        scores[br_ys_binds] = -1
        scores[br_xs_binds] = -1

    scores[cls_inds] = -1
    scores[dist_inds] = -1
    scores[width_inds] = -1
    scores[height_inds] = -1

    # 分数最后在整理一下，找到topK个最大的
    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    # bboxes也要整理
    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses = tl_clses.contiguous().view(batch, -1, 1)
    clses = _gather_feat(clses, inds).float()

    tl_scores = tl_scores.contiguous().view(batch, -1, 1)
    tl_scores = _gather_feat(tl_scores, inds).float()
    br_scores = br_scores.contiguous().view(batch, -1, 1)
    br_scores = _gather_feat(br_scores, inds).float()

    detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)
    return detections


# 上采样
class upsample(nn.Module):
    def __init__(self, scale_factor):
        super(upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor)


# 就是个加法
class merge(nn.Module):
    def forward(self, x, y):
        return x + y


# 卷积层 conv + bn +rule
class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu


# 残差网络类
class residual(nn.Module):
    def __init__(self, inp_dim, out_dim, k=3, stride=1):
        super(residual, self).__init__()
        # 默认卷积核的尺度k是3，步长stride是1
        p = (k - 1) // 2

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(p, p), stride=(stride, stride), bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (k, k), padding=(p, p), bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)

        # 如果步长stride不为1，且输入维度和输出维度不同，我们就放一个残差网络在这
        self.skip = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    # 依次进行下去就是了
    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        skip = self.skip(x)
        return self.relu(bn2 + skip)


# corner的池化层[我觉得这可能就是他的改进之处了吧]
# 角点池化的具体实现是用c++写的，关于池化的代码在./core/models/pyutils/_cpools/下
class corner_pool(nn.Module):
    def __init__(self, dim, pool1, pool2):
        super(corner_pool, self).__init__()
        self._init_layers(dim, pool1, pool2)

    def _init_layers(self, dim, pool1, pool2):
        # 新建两卷积层,核是3x3的，不管输入维度，反正输出128
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)

        # 新建一卷积层，核是3x3，拓展了1
        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1 = nn.BatchNorm2d(dim)

        # 这个是核是1x1，就没得啥拓展的事情了
        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        # 又来了一个卷积层，为啥啊
        self.conv2 = convolution(3, dim, dim)

        # 池化1和池化2
        self.pool1 = pool1()
        self.pool2 = pool2()

    def forward(self, x):
        # 这两个池化层分别先池化
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1 = self.pool1(p1_conv1)

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2 = self.pool2(p2_conv1)

        # 然后我们把俩池化层加起来
        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1 = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2
