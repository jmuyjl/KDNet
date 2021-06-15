# encoding=utf-8
import torch
import torch.nn as nn
import numpy as np 
import time
import math
import torch.nn.functional as F

from .utils import residual, upsample, merge, _decode
from .utils import extract_fa, extract_fg, update_heatmap
from .visualize import visualize

def _make_layer(inp_dim, out_dim, modules):
    layers  = [residual(inp_dim, out_dim)]
    layers += [residual(out_dim, out_dim) for _ in range(1, modules)]
    return nn.Sequential(*layers)

def _make_layer_revr(inp_dim, out_dim, modules):
    layers  = [residual(inp_dim, inp_dim) for _ in range(modules - 1)]
    layers += [residual(inp_dim, out_dim)]
    return nn.Sequential(*layers)

def _make_pool_layer(dim):
    return nn.MaxPool2d(kernel_size=2, stride=2)

def _make_unpool_layer(dim):
    return upsample(scale_factor=2)

def _make_merge_layer(dim):
    return merge()

class hg_module(nn.Module):
    def __init__(
        self, n, dims, modules, make_up_layer=_make_layer,
        make_pool_layer=_make_pool_layer, make_hg_layer=_make_layer,
        make_low_layer=_make_layer, make_hg_layer_revr=_make_layer_revr,
        make_unpool_layer=_make_unpool_layer, make_merge_layer=_make_merge_layer
    ):
        super(hg_module, self).__init__()

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.n    = n
        self.up1  = make_up_layer(curr_dim, curr_dim, curr_mod)
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(curr_dim, next_dim, curr_mod)
        self.low2 = hg_module(
            n - 1, dims[1:], modules[1:],
            make_up_layer=make_up_layer,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            make_low_layer=make_low_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer
        ) if n > 1 else make_low_layer(next_dim, next_dim, next_mod)
        self.low3 = make_hg_layer_revr(next_dim, curr_dim, curr_mod)
        self.up2  = make_unpool_layer(curr_dim)
        self.merg = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        merg = self.merg(up1, up2)
        return merg

class hg(nn.Module):
    def __init__(self, pre, hg_modules, cnvs, inters, cnvs_, inters_):
        super(hg, self).__init__()

        self.pre  = pre
        self.hgs  = hg_modules
        #######################################
        # for p in self.hgs.parameters():
        #     p.requires_grad = False
        #######################################
        self.cnvs = cnvs

        self.inters  = inters
        self.inters_ = inters_
        self.cnvs_   = cnvs_

    def forward(self, x):
        inter = self.pre(x)
        
        cnvs  = []
        hgs   = []
        for ind, (hg_, cnv_) in enumerate(zip(self.hgs, self.cnvs)):
            hg  = hg_(inter)
            cnv = cnv_(hg)
            cnvs.append(cnv)
            hgs.append(hg)

            if ind < len(self.hgs) - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = nn.functional.relu_(inter)
                inter = self.inters[ind](inter)
        return cnvs, hgs

class hg_net(nn.Module):
    def __init__(
        self, hg, tl_modules, br_modules, tl_heats, br_heats, 
        tl_tags, br_tags, tl_offs, br_offs, cnvs, cnvs_, 
        region=7, num_kps=50, fc_dim=1024, cnv_dim=256, out_dim=5
    ):
        super(hg_net, self).__init__()

        self._decode = _decode

        self.hg = hg

        self.tl_modules = tl_modules
        self.br_modules = br_modules

        self.tl_heats = tl_heats
        self.br_heats = br_heats

        self.tl_tags = tl_tags
        self.br_tags = br_tags
        
        self.tl_offs = tl_offs
        self.br_offs = br_offs
        
        self.cnvs = cnvs
        self.cnvs_ = cnvs_
        ####################  relation added by yjl 2019/10/11   ####################
        self.region = region
        self.fc_dim = fc_dim
        self.num_kps = num_kps
        self.cnv_dim = cnv_dim
        self.out_dim = out_dim
        
        self.fc1 = nn.Linear(self.region ** 2 * self.cnv_dim, self.fc_dim)
        self.fc2 = nn.Linear(self.fc_dim, self.region ** 2 * self.cnv_dim)
        
        # self.conv1 = nn.Conv2d(self.cnv_dim, 20, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # self.conv2 = nn.Conv2d(20, self.cnv_dim, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.extract_fa = extract_fa
        self.extract_fg = extract_fg
        self.update_heatmap = update_heatmap
        self.relationship = RelationModule(self.num_kps)
        ####################  relation added by yjl 2019/10/11   ####################

    def _train(self, xs, ys):
        '''
        xs列表维度为1 
        ys列表维度为7
        
        '''
        # print(80*"*")
        # print(len(xs))
        # print(len(ys))
        # print(80*"*")
        # torch.Size([13, 3, 511, 511])
        # torch.Size([14, 3, 511, 511])
        image = xs[0]
        # type(cnvs) = "list" = 2
        cnvs, hgs = self.hg(image)
        '''
        cnvs[0]: (13, 256, 64, 64)
        cnvs[1]: (13, 256, 64, 64)
        '''
        
        # 下面所有列表len=2
        # tl_modules[0]: (13, 256, 64, 64)
        # tl_modules[1]: (13, 256, 64, 64)
        tl_modules = [tl_mod_(cnv) for tl_mod_, cnv in zip(self.tl_modules, cnvs)]
        br_modules = [br_mod_(cnv) for br_mod_, cnv in zip(self.br_modules, cnvs)]
        # tl_heats[0]: (13, 5, 64, 64)
        # tl_heats[1]: (13, 5, 64, 64),数值是-2.+
        tl_heats   = [tl_heat_(tl_mod) for tl_heat_, tl_mod in zip(self.tl_heats, tl_modules)]
        br_heats   = [br_heat_(br_mod) for br_heat_, br_mod in zip(self.br_heats, br_modules)]
        # tl_tags    = [tl_tag_(tl_mod)  for tl_tag_,  tl_mod in zip(self.tl_tags,  tl_modules)]
        # br_tags    = [br_tag_(br_mod)  for br_tag_,  br_mod in zip(self.br_tags,  br_modules)]
        # tl_offs    = [tl_off_(tl_mod)  for tl_off_,  tl_mod in zip(self.tl_offs,  tl_modules)]
        # br_offs    = [br_off_(br_mod)  for br_off_,  br_mod in zip(self.br_offs,  br_modules)]
        
        ####################  relation added by yjl 2019/10/10   ####################
        gt_tl_heats = ys[0]
        gt_br_heats = ys[1]
        hgs_tl = []
        hgs_br = []
        
        for i in range(len(hgs)):
            # start = time.clock()
            hgs_tl.append(self.relation(tl_heats[i], gt_tl_heats, hgs[i].detach()))
            # end = time.clock()
            # print('Relation Module running time: %s seconds' % (end - start))
            hgs_br.append(self.relation(br_heats[i], gt_br_heats, hgs[i].detach()))
        
        
        cnv_tl = []
        cnv_br = []
        #################新的feature-map已经出来###################
        for ind, cnv_ in enumerate(self.cnvs):
            cnv_tl_, cnv_br_ = cnv_(hgs_tl[ind]), cnv_(hgs_br[ind])
            cnv_tl.append(cnv_tl_)
            cnv_br.append(cnv_br_)
        tl_modules = [tl_mod_(cnv) for tl_mod_, cnv in zip(self.tl_modules, cnv_tl)]
        br_modules = [br_mod_(cnv) for br_mod_, cnv in zip(self.br_modules, cnv_br)]
        tl_heats   = [tl_heat_(tl_mod) for tl_heat_, tl_mod in zip(self.tl_heats, tl_modules)]
        br_heats   = [br_heat_(br_mod) for br_heat_, br_mod in zip(self.br_heats, br_modules)]
        tl_tags    = [tl_tag_(tl_mod)  for tl_tag_,  tl_mod in zip(self.tl_tags,  tl_modules)]
        br_tags    = [br_tag_(br_mod)  for br_tag_,  br_mod in zip(self.br_tags,  br_modules)]
        tl_offs    = [tl_off_(tl_mod)  for tl_off_,  tl_mod in zip(self.tl_offs,  tl_modules)]
        br_offs    = [br_off_(br_mod)  for br_off_,  br_mod in zip(self.br_offs,  br_modules)]
        ####################  relation added by yjl 2019/10/10   ####################
        return [tl_heats, br_heats, tl_tags, br_tags, tl_offs, br_offs]

    def _test(self, *xs, **kwargs):
        image = xs[0]
        cnvs, hgs = self.hg(image)

        tl_mod = self.tl_modules[-1](cnvs[-1])
        br_mod = self.br_modules[-1](cnvs[-1])
        
        
        tl_heat, br_heat = self.tl_heats[-1](tl_mod), self.br_heats[-1](br_mod)
        # tl_tag,  br_tag  = self.tl_tags[-1](tl_mod),  self.br_tags[-1](br_mod)
        # tl_off,  br_off  = self.tl_offs[-1](tl_mod),  self.br_offs[-1](br_mod)
        ####################  test_relation added by yjl 2019/10/10   ####################
        
        hgs_tl = self.test_relation(tl_heat, hgs[-1])
        hgs_br = self.test_relation(tl_heat, hgs[-1])

        cnv_tl = self.cnvs[-1](hgs_tl)
        cnv_br = self.cnvs[-1](hgs_br)
        
        tl_mod = self.tl_modules[-1](cnv_tl)
        br_mod = self.br_modules[-1](cnv_br)
        
        tl_heat, br_heat = self.tl_heats[-1](tl_mod), self.br_heats[-1](br_mod)
        tl_tag,  br_tag  = self.tl_tags[-1](tl_mod),  self.br_tags[-1](br_mod)
        tl_off,  br_off  = self.tl_offs[-1](tl_mod),  self.br_offs[-1](br_mod)
        
        ####################  test_relation added by yjl 2019/10/10   ####################
        
        
        ##########   visualize   ##########
        # if 'debug' in kwargs:
        #     if kwargs['debug']:
        #         visualize(image, tl_heat, br_heat)
        #     del kwargs['debug']
        # visualize(image, tl_heat, br_heat)
        ##########   visualize   ##########

        outs = [tl_heat, br_heat, tl_tag, br_tag, tl_off, br_off]
        return self._decode(*outs, **kwargs), tl_heat, br_heat, tl_tag, br_tag
    # # train
    # def forward(self, xs, ys, test=False, **kwargs):
    #     if not test:
    #         return self._train(xs, ys, **kwargs)
    #     return self._test(*xs, **kwargs)
    # # test
    def forward(self, *xs, test=False, **kwargs):
        return self._test(*xs, **kwargs)
        
    def relation(self, pred_heatmap, gt_heatmap, feature_map):
        '''
        torch.Size([13, 5, 64, 64])
        torch.Size([13, 5, 64, 64])
        torch.Size([13, 256, 64, 64])
        '''
        start_time = time.clock()
        # kps_num_each_channel = torch.zeros([14, self.out_dim], dtype=torch.int32)
        for batch_index, tl_heat in enumerate(pred_heatmap):
            kp_vector_cat = torch.cuda.FloatTensor()
            gt_kp_vector_cat = torch.cuda.FloatTensor()
            fg_cat = torch.cuda.IntTensor()
            gt_fg_cat = torch.cuda.IntTensor()
            kp_position_cat = torch.cuda.LongTensor()
            gt_kp_position_cat = torch.cuda.LongTensor()
            each_1024_new_cat = torch.cuda.FloatTensor()
            
            for channel_index, tl_heat_one_channel in enumerate(tl_heat):
                gt_kp_0 = gt_heatmap[batch_index][channel_index] == 1.0
                gt_kp_position = torch.nonzero(gt_kp_0)
                gt_kp_position_cat = torch.cat((gt_kp_position_cat, gt_kp_position), 0)
                del gt_kp_0
                if gt_kp_position.size(0) == 0:
                    continue
                else:
                    tl_heat_one_channel_1 = tl_heat_one_channel.reshape(-1)
                    _, kp_position = torch.topk(tl_heat_one_channel_1, self.num_kps)
                    kp_x, kp_y = kp_position / tl_heat_one_channel.size(1), kp_position % tl_heat_one_channel.size(1)
                    kp_position = torch.cat((kp_x, kp_y), 0).reshape(2, -1).t()
                    kp_position_cat = torch.cat((kp_position_cat, kp_position), 0)
            if gt_kp_position_cat.size(0) != 0:
                for index, _ in enumerate(kp_position_cat):
                    # 直接利用slice
                    for_fa = self.extract_fa(feature_map[batch_index], kp_position_cat[index][0],
                                             kp_position_cat[index][1], self.region)
                    kp_vector = self.fc1(for_fa.reshape(-1))
                    # for_fa_cnv = self.conv1()
                    kp_vector_cat = torch.cat((kp_vector_cat, kp_vector), 0)
                    for_fg = self.extract_fg(kp_position_cat[index][0], kp_position_cat[index][1], self.region,
                                             self.region)
                    fg_cat = torch.cat((fg_cat, for_fg.cuda()), 0)

                for gt_index, _ in enumerate(gt_kp_position_cat):
                    gt_for_fa = self.extract_fa(feature_map[batch_index], gt_kp_position_cat[gt_index][0],
                                                gt_kp_position_cat[gt_index][1], self.region)

                    gt_kp_vector = self.fc1(gt_for_fa.reshape(-1))
                    
                    gt_kp_vector_cat = torch.cat((gt_kp_vector_cat, gt_kp_vector), 0)
                    gt_for_fg = self.extract_fg(gt_kp_position_cat[gt_index][0], gt_kp_position_cat[gt_index][1],
                                                self.region, self.region)
                    gt_fg_cat = torch.cat((gt_fg_cat, gt_for_fg.cuda()), 0)
                kp_vector_cat = torch.cat((kp_vector_cat, gt_kp_vector_cat), 0)
                del gt_kp_vector_cat
                fg_cat = torch.cat((fg_cat, gt_fg_cat), 0)
                del gt_fg_cat

                fa = kp_vector_cat.reshape([kp_position_cat.size(0) + gt_kp_position_cat.size(0), -1])
                del kp_vector_cat
                fg = fg_cat.reshape([kp_position_cat.size(0) + gt_kp_position_cat.size(0), -1])
                del fg_cat
                # fa_new_vector = [num_kps, 1024]
                fa_new_vector = self._heatmap_relation(fa, fg)
                for vector_index, each_1024 in enumerate(fa_new_vector):
                    each_1024_new = self.fc2(each_1024.reshape(-1))
                    each_1024_new_cat = torch.cat((each_1024_new_cat,
                                                   each_1024_new), 0)
                fa_new = each_1024_new_cat.reshape(
                    [fa_new_vector.size(0), self.cnv_dim, self.region, self.region])
                
                # print(fa_new_vector.size(0))
                fa_new = fa_new[:kp_position_cat.size(0), :, :, :]
                # print(fa_new.size())
                
                empty_fm = self.update_heatmap(fa_new, feature_map[batch_index], fg, self.region)
                ################################
                # 更新的
                feature_map[batch_index] = feature_map[batch_index] + empty_fm
                ################################
                # print("feature_map update ok !!!")
                del gt_kp_position_cat
                del kp_position_cat
                
            # end_time = time.clock()
            # a = a + (end_time - start_time)
            # print('Relation Module running time: %s seconds' % (end_time - start_time))
        # print(kps_num_each_channel)
        
        end_time = time.clock()
        # a = a + (end_time - start_time)
        # print('Relation Module running time: %s seconds' % (end_time - start_time))
        # print(a)
        return feature_map
        
    def _heatmap_relation(self, fa, fg):
        '''
        fa:[num_kps, 45]
        fg:[num_kps, 4]
        '''
        # relationship = RelationModule(20)
        # [num_kps, 4]-->[num_kps, nongt_dim, 4]
        position_matrix = self.relationship.extract_position_matrix(fg, self.num_kps)
        # [num_kps, nongt_dim, 4]-->[num_kps, nongt_dim, 64]
        position_embedding = self.relationship.extract_position_embedding(position_matrix, feat_dim=64)
        fc_all_1 = fa + self.relationship(position_embedding, fa)
        fc_all_1_relu = F.relu(fc_all_1)
        return fc_all_1_relu
        
    def test_relation(self, pred_heatmap, feature_map):
        for batch_index, tl_heat in enumerate(pred_heatmap):
            
            kp_vector_cat = torch.cuda.FloatTensor()
            fg_cat = torch.cuda.IntTensor()
            kp_position_cat = torch.cuda.LongTensor()
            each_1024_new_cat = torch.cuda.FloatTensor()
            
            for channel_index, tl_heat_one_channel in enumerate(tl_heat):
                # 直接每个通道取top-50
                tl_heat_one_channel_1 = tl_heat_one_channel.reshape(-1)
                _, kp_position = torch.topk(tl_heat_one_channel_1, self.num_kps)
                # 对列取余
                kp_x, kp_y = kp_position / tl_heat_one_channel.size(1), kp_position % tl_heat_one_channel.size(1)
                kp_position = torch.cat((kp_x, kp_y), 0).reshape(2, -1).t()
                # print(kp_position)
                kp_position_cat = torch.cat((kp_position_cat, kp_position), 0)
            # print(80*"*")
            # print(kp_position_cat)
            for index, _ in enumerate(kp_position_cat):
                # 直接利用slice
                for_fa = self.extract_fa(feature_map[batch_index], kp_position_cat[index][0], kp_position_cat[index][1],
                                         self.region)
                kp_vector = self.fc1(for_fa.reshape(-1))
                kp_vector_cat = torch.cat((kp_vector_cat, kp_vector), 0)
                for_fg = self.extract_fg(kp_position_cat[index][0], kp_position_cat[index][1], self.region, self.region)
                fg_cat = torch.cat((fg_cat, for_fg.cuda()), 0)

            fa = kp_vector_cat.reshape([kp_position_cat.size(0), -1])
            del kp_vector_cat
            fg = fg_cat.reshape([kp_position_cat.size(0), -1])
            del fg_cat

            fa_new_vector = self._heatmap_relation(fa, fg)
            for vector_index, each_1024 in enumerate(fa_new_vector):
                each_1024_new = self.fc2(each_1024.reshape(-1))
                each_1024_new_cat = torch.cat((each_1024_new_cat,
                                               each_1024_new), 0)
            fa_new = each_1024_new_cat.reshape(
                [fa_new_vector.size(0), self.cnv_dim, self.region, self.region])

            # fa_new = fa_new_vector.reshape([self.num_kps, self.out_dim, self.region, self.region])
            empty_fm = self.update_heatmap(fa_new, feature_map[batch_index], fg, self.region)
            ################################
            # 更新的
            feature_map[batch_index] = feature_map[batch_index] + empty_fm
            # pred_heatmap[batch_index] = tl_heat
        return feature_map
        
class RelationModule(nn.Module):
    def __init__(self, nongt_dim, feat_dim=1024, emb_dim=64, dim=(1024, 1024, 1024), group=16):
        """
        Relation Module init
        :param nongt_dim: int
        :param feat_dim: int
        :param emb_dim: int
        :param dim: a 3-tuple of (query, key, output)
        :param group: The variable 'fc_dim' shares values with it
        """
        super(RelationModule, self).__init__()

        self.group = group
        self.fc_dim = group
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.nongt_dim = nongt_dim
        self.dim_group = (dim[0] // group, dim[1] // group, dim[2] // group)

        self.fc_q = nn.Linear(feat_dim, feat_dim).cuda()
        self.fc_k = nn.Linear(feat_dim, feat_dim).cuda()
        self.fc_pos_feat = nn.Linear(emb_dim, self.fc_dim).cuda()
        self.conv_cat = nn.Conv2d(group * feat_dim, feat_dim, kernel_size=1, groups=group).cuda()

    def forward(self, position_embedding: torch.Tensor, roi_feat: int):
        """
        Relation Module Forward
        :param position_embedding: [num_rois, nongt_dim, emb_dim], emb_dim = 64
        :param roi_feat: [num_rois, feat_dim]
        :return: output: [num_rois, feat_dim]
        """
        pos_embedding_reshape = position_embedding.reshape(position_embedding.shape[0] * position_embedding.shape[1],
                                                           *(position_embedding.shape[2:]))
        pos_feat = self.fc_pos_feat(pos_embedding_reshape)
        pos_feat_relu = F.relu(pos_feat)
        aff_weight = pos_feat_relu.reshape(-1, self.nongt_dim, self.fc_dim)
        aff_weight = aff_weight.transpose(2, 1)

        nongt_roi_feat = roi_feat[:self.nongt_dim, :]
        q_data = self.fc_q(roi_feat)
        q_data_batch = q_data.reshape(-1, self.group, self.dim_group[0]).transpose(1, 0)
        k_data = self.fc_k(nongt_roi_feat)
        k_data_batch = k_data.reshape(-1, self.group, self.dim_group[1]).transpose(1, 0)
        v_data = nongt_roi_feat

        aff = torch.matmul(q_data_batch, k_data_batch.transpose(2, 1))
        aff_scale = (1.0 / math.sqrt(float(self.dim_group[1]))) * aff
        aff_scale = aff_scale.transpose(1, 0)
        weighted_aff = torch.log(aff_weight.clamp(min=1e-6)) + aff_scale
        aff_softmax = F.softmax(weighted_aff, 2)
        aff_softmax_reshape = aff_softmax.reshape(aff_softmax.shape[0] * aff_softmax.shape[1], *(aff_softmax.shape[2:]))
        output_t = torch.mm(aff_softmax_reshape, v_data)
        output_t = output_t.reshape(-1, self.fc_dim * self.feat_dim, 1, 1)
        linear_out = self.conv_cat(output_t)
        output = linear_out.reshape(linear_out.shape[0], linear_out.shape[1]).cuda()
        return output

    @staticmethod
    def extract_position_matrix(bbox: torch.Tensor, nongt_dim: int) -> torch.Tensor:
        """
        Convert bbox matrix to position matrix
        :param bbox: [num_rois, 4]
        :param nongt_dim: int
        :return: pos_matrix: [num_rois, nongt_dim, 4]
        """
        # 四个均是 [num_kps, 1]
        bbox = bbox.float()
        center_x, center_y, bbox_w, bbox_h = bbox.chunk(chunks=4, dim=1)
        delta_x = (center_x - center_x.t()) / bbox_w
        delta_x = delta_x.abs().clamp(min=1e-3)
        delta_x = torch.log(delta_x)
        delta_y = (center_y - center_y.t()) / bbox_h
        delta_y = delta_y.abs().clamp(min=1e-3)
        delta_y = torch.log(delta_y)
        delta_w = bbox_w / bbox_w.t()
        delta_w = torch.log(delta_w)
        delta_h = bbox_h / bbox_h.t()
        delta_h = torch.log(delta_h)
        concat_list = [d[:, :nongt_dim].unsqueeze(2) for d in (delta_x, delta_y, delta_w, delta_h)]
        pos_matrix = torch.cat(concat_list, 2)
        # torch.Size([num_kps, nongt_dim, 4])
        return pos_matrix

    @staticmethod
    def extract_position_embedding(position_mat: torch.Tensor, feat_dim: int, wave_length=1000) -> torch.Tensor:
        """
        Convert position matrix to position embedding matrix
        :param position_mat: [num_rois, nongt_dim, 4]
        :param feat_dim: int
        :param wave_length: int, the default is 1000
        :return:
        """
        feat_range = torch.arange(0., feat_dim / 8.)
        dim_mat = torch.pow(torch.full((1,), wave_length), (8. / feat_dim) * feat_range)
        dim_mat = dim_mat.reshape(1, 1, 1, -1).cuda()
        pos_mat = torch.unsqueeze(100.0 * position_mat, 3).cuda()
        div_mat = pos_mat / dim_mat
        sin_mat = torch.sin(div_mat)
        cos_mat = torch.cos(div_mat)
        embedding = torch.cat((sin_mat, cos_mat), 3)
        # torch.Size([num_kps, num_kps, 64])
        embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], feat_dim)
        return embedding