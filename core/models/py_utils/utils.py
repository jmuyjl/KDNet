import torch
import torch.nn as nn
import os
import random
import cv2
import numpy as np 

def draw_proposals(position_matrix, img, extract=5):
    if os.path.exists("/home/jhsu/yjl/lite/draw_proposals/"):
        pass
    else:
        os.mkdir("/home/jhsu/yjl/lite/draw_proposals/")
    # red, blue, green, yellow, black
    # (14, 3, 511, 511)
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                    dtype=np.float32).reshape(3, 1, 1)
    img = (img[0].detach().cpu().numpy() * std + mean) * 255
    # 再把图片transpose成标准的样子
    img = img.astype(np.uint8).transpose(1, 2, 0)
    img = img.copy()
    # img = np.array(img.transpose(2, 0).cpu())
    a = [(0, 0, 255), (255, 0, 0), (0, 100, 0), (0, 255, 255), (0, 0, 0)]
    # print(80*"K")
    # print(type(img))
    # print(img.shape)
    for idx, position in enumerate(position_matrix):
        y = position[0].item()
        x = position[1].item()
        # print(x, y)
        # if x <= extract // 2:
        #     x = extract // 2
        # elif x + extract // 2 >= img.shape[1] - 1:
        #     x = img.shape[1] - extract // 2 - 1
        # if y <= extract // 2:
        #     y = extract // 2
        # elif y + extract // 2 >= img.shape[1] - 1:
        #     y = img.shape[1] - extract // 2 - 1
        # print(x, y)
        # color = BGR
        img = cv2.circle(img, center=(x*8, y*8), radius=extract, color=a[idx // 50], thickness=-1)
        # img = cv2.rectangle(img, (x*8 - extract*8, y*8 - extract*8), (x*8 + extract*8, y*8 + extract*8), color=a[idx // 50])
    count = random.randint(0, 10000)
    cv2.imwrite("/home/jhsu/yjl/lite/draw_proposals/iamge_" + str(count) + ".jpg", img)
    print("save img ok!!!")

def save_tensor(a, BATCH, C, H, W):
    print(a.size())
    a = a.numpy()
    with open("/home/jhsu/yjl/lite/core/models/py_utils/save_tensor.txt", 'w') as f:
        f.write("[")
        for batch, n in enumerate(a):
            f.write("[")
            for x, i in enumerate(n):
                f.write("[")
                for y, j in enumerate(i):
                    f.write("[")
                    for z, k in enumerate(j):
                        if z == W:
                            f.write(str(k))
                        else:
                            f.write(str(k) + ", ")
                    if y == H and x != C and batch != BATCH:
                        f.write("]]," + "\n")
                    elif y == H and x == C and batch != BATCH:
                        f.write("]]]," + "\n")
                    elif batch == BATCH and y == H and x == C:
                        f.write("]]]]" + "\n")
                    else:
                        f.write("]," + "\n")
                f.write("\n")
    print(" save done！")

def extract_fa(fm, x, y, extract):
    x = int(x)
    y = int(y)
    # x轴坐标变换
    if x <= extract // 2:
        x = extract // 2
    elif x + extract // 2 >= fm.size(1) - 1:
        x = fm.size(1) - extract // 2 - 1
        
    # y轴坐标变换
    if y <= extract // 2:
        y = extract // 2
    elif y + extract // 2 >= fm.size(2) - 1:
        y = fm.size(2) - extract // 2 - 1
    # 截取
    if extract == 3:
        for_fa = fm[:, x - 1:x + extract - 1, y - 1:y + extract - 1]
    elif extract == 5:
        for_fa = fm[:, x - 2:x + extract - 2, y - 2:y + extract - 2]
    else:
        for_fa = fm[:, x - 3:x + extract - 3, y - 3:y + extract - 3]
    # print(for_fa.size())
    return for_fa


def extract_fg(x, y, w, h):
    '''
    传入一个kp
    return一个3*3*7的tensor
    '''
    for_fg = torch.zeros([4], dtype=torch.int32)
    for_fg[0] = x
    for_fg[1] = y
    for_fg[2] = w
    for_fg[3] = h
    return for_fg


def update_heatmap(fa_new, fm, fg, extract):
    '''
    fa_new是[num_kps, 256, 7, 7]
    fm是[256, 128, 128]
    fg是[num_kps, 4]
    extract是截取区域
    '''
    # 虽然这个empty_fm现在不需要梯度，但是算完之后就要梯度了
    empty_fm = torch.zeros([fm.size(0), fm.size(1), fm.size(2)]).cuda()

    for fa_extract, fa_position in zip(fa_new, fg):
        x = fa_position[0]
        y = fa_position[1]

        # x轴坐标变换
        if x <= extract // 2:
            x = extract // 2
        elif x + extract // 2 >= fm.size(1) - 1:
            x = fm.size(1) - extract // 2 - 1
            
        # y轴坐标变换
        if y <= extract // 2:
            y = extract // 2
        elif y + extract // 2 >= fm.size(2) - 1:
            y = fm.size(2) - extract // 2 - 1

        if extract == 3:
            empty_fm[:, x - 1:x + extract - 1, y - 1:y + extract - 1] = fa_extract
        elif extract == 5:
            empty_fm[:, x - 2:x + extract - 2, y - 2:y + extract - 2] = fa_extract
        else:
            empty_fm[:, x - 3:x + extract - 3, y - 3:y + extract - 3] = fa_extract

    empty_fm = empty_fm.detach()
    return empty_fm
    
    
def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _topk(scores, K=20):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

# 先sigmoid变成响应值，其次nms(maxpooling)，找topk
def _decode(
    tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr, 
    K=200, kernel=1, ae_threshold=1, num_dets=1000, no_border=False
):
    batch, cat, height, width = tl_heat.size()

    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)

    # perform nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)

    ## 在top left以及bottom right,找到最大的前K个点，并记录下他们的得分，位置，类别，坐标等信息，下面返回的结果分别代表的是：
    ## 类别得分，位置索引，类别，y坐标，x坐标
    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)

    tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
    tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
    br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
    br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)

    if no_border:
        tl_ys_binds = (tl_ys == 0)
        tl_xs_binds = (tl_xs == 0)
        br_ys_binds = (br_ys == height - 1)
        br_xs_binds = (br_xs == width  - 1)

    if tl_regr is not None and br_regr is not None:
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
        tl_regr = tl_regr.view(batch, K, 1, 2)
        br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
        br_regr = br_regr.view(batch, 1, K, 2)

        tl_xs = tl_xs + tl_regr[..., 0]
        tl_ys = tl_ys + tl_regr[..., 1]
        br_xs = br_xs + br_regr[..., 0]
        br_ys = br_ys + br_regr[..., 1]

    # all possible boxes based on top k corners (ignoring class)
    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)

    tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)
    tl_tag = tl_tag.view(batch, K, 1)
    br_tag = _tranpose_and_gather_feat(br_tag, br_inds)
    br_tag = br_tag.view(batch, 1, K)
    dists  = torch.abs(tl_tag - br_tag)

    tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
    br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)
    scores    = (tl_scores + br_scores) / 2

    # reject boxes based on classes
    tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)
    br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)
    cls_inds = (tl_clses != br_clses)

    # reject boxes based on distances
    dist_inds = (dists > ae_threshold)

    # reject boxes based on widths and heights
    width_inds  = (br_xs < tl_xs)
    height_inds = (br_ys < tl_ys)

    if no_border:
        scores[tl_ys_binds] = -1
        scores[tl_xs_binds] = -1
        scores[br_ys_binds] = -1
        scores[br_xs_binds] = -1

    scores[cls_inds]    = -1
    scores[dist_inds]   = -1
    scores[width_inds]  = -1
    scores[height_inds] = -1

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses  = tl_clses.contiguous().view(batch, -1, 1)
    clses  = _gather_feat(clses, inds).float()

    tl_scores = tl_scores.contiguous().view(batch, -1, 1)
    tl_scores = _gather_feat(tl_scores, inds).float()
    br_scores = br_scores.contiguous().view(batch, -1, 1)
    br_scores = _gather_feat(br_scores, inds).float()

    detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)
    return detections

class upsample(nn.Module):
    def __init__(self, scale_factor):
        super(upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor)

class merge(nn.Module):
    def forward(self, x, y):
        return x + y

class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class residual(nn.Module):
    def __init__(self, inp_dim, out_dim, k=3, stride=1):
        super(residual, self).__init__()
        p = (k - 1) // 2

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(p, p), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (k, k), padding=(p, p), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)
        
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)

class corner_pool(nn.Module):
    def __init__(self, dim, pool1, pool2):
        super(corner_pool, self).__init__()
        self._init_layers(dim, pool1, pool2)

    def _init_layers(self, dim, pool1, pool2):
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1()
        self.pool2 = pool2()

    def forward(self, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1    = self.pool1(p1_conv1)

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2    = self.pool2(p2_conv1)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2
