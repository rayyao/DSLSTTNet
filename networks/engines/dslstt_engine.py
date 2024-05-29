import numpy as np
import torch.nn as nn
import torch

from utils.image import one_hot_mask

from networks.layers.basic import seq_to_2d
from networks.engines.vgb_engine import VGBEngine, VGBInferEngine


class DSLSTTEngine(VGBEngine):
    def __init__(self,
                 lstn_model,
                 gpu_id=0,
                 long_term_mem_gap=9999,
                 short_term_mem_skip=1,
                 layer_loss_scaling_ratio=2.):
        super().__init__(lstn_model, gpu_id, long_term_mem_gap,
                         short_term_mem_skip)
        self.layer_loss_scaling_ratio = layer_loss_scaling_ratio
        # self.dw_com = nn.Linear(640, 512).cuda()
        # # self.up_v = nn.Linear(128, 512).cuda()
        # self.linear1 = nn.Linear(640, 640).cuda()
        # self.linear2 = nn.Linear(128, 128).cuda()
        # self.linear3 = nn.Linear(512, 512).cuda()
        # self.sigmoid = nn.Sigmoid()

    def update_short_term_memory(self, curr_mask, curr_id_emb=None, skip_long_term_update=False):

        if curr_id_emb is None:
            if len(curr_mask.size()) == 3 or curr_mask.size()[0] == 1:
                curr_one_hot_mask = one_hot_mask(curr_mask, self.max_obj_num)
            else:
                curr_one_hot_mask = curr_mask
            curr_id_emb = self.assign_identity(curr_one_hot_mask)

        lsab_curr_memories = self.curr_lsab_output[1]
        lsab_curr_memories_2d = []
        for layer_idx in range(len(lsab_curr_memories)):
            curr_k, curr_v, curr_id_k, curr_id_v = lsab_curr_memories[layer_idx]
            # # curr_v = self.dw_v(curr_v)
            # com = torch.cat([curr_k, curr_v], dim=-1)
            # com = self.linear1(com)
            # com = self.sigmoid(com)
            # # com1, com2 = torch.split(com, [128, 512], dim=2)
            # com = self.dw_com(com)
            # # curr_k = self.linear2(curr_k)
            # b2 = self.linear3(curr_v)
            # # b3 = self.linear3(curr_v)
            # curr_v = com * curr_v + b2
            # # curr_v = self.up_v(curr_v)
            curr_id_k, curr_id_v = self.lstn.LSAB.layers[layer_idx].fuse_key_value_id(curr_id_k, curr_id_v, curr_id_emb)
            lsab_curr_memories[layer_idx][2], lsab_curr_memories[layer_idx][3] = curr_id_k, curr_id_v
            local_curr_id_k = seq_to_2d(curr_id_k, self.enc_size_2d) if curr_id_k is not None else None
            local_curr_id_v = seq_to_2d(curr_id_v, self.enc_size_2d)
            lsab_curr_memories_2d.append([
                seq_to_2d(curr_k, self.enc_size_2d),
                seq_to_2d(curr_v, self.enc_size_2d), local_curr_id_k,
                local_curr_id_v
            ])

        self.short_term_memories_list.append(lsab_curr_memories_2d)
        self.short_term_memories_list = self.short_term_memories_list[-self.short_term_mem_skip:]
        self.short_term_memories = self.short_term_memories_list[0]

        # 计算内积
        a = lsab_curr_memories[layer_idx][0]
        b = self.long_term_memories[layer_idx][0]
        # 切片操作
        b = b[b.size(0)-a.size(0):]

        reshaped_a = a.view(-1, 128)
        reshaped_b = b.view(-1, 128)

        similarity_matrix = torch.matmul(reshaped_a.transpose(0, 1), reshaped_b)

        # 计算归一化因子
        norm_tensor1 = torch.norm(reshaped_a,  dim=0, keepdim=True)
        norm_tensor2 = torch.norm(reshaped_b,  dim=0, keepdim=True)

        # # 计算内积
        # a = lsab_curr_memories[layer_idx][0]
        # b = self.long_term_memories[layer_idx][0]
        #
        # reshaped_a = a.view(-1, 128)
        # reshaped_b = b.view(-1, 128)
        #
        # similarity_matrix = torch.matmul(reshaped_a, reshaped_b.transpose(0, 1))
        #
        # # 计算归一化因子
        # norm_tensor1 = torch.norm(reshaped_a, dim=1)
        # norm_tensor2 = torch.norm(reshaped_b, dim=1)

        similarity_scores = similarity_matrix / (norm_tensor1.view(-1, 1) * norm_tensor2)
        max_similarity = torch.max(similarity_scores)

        # 判断最大相似值是否大于0.9
        if max_similarity.item() <= self.long_term_mem_gap:

        # if self.frame_step - self.last_mem_step >= self.long_term_mem_gap:
            if not skip_long_term_update:
                self.update_long_term_memory(lsab_curr_memories)
            self.last_mem_step = self.frame_step


class DSLSTTInferEngine(VGBInferEngine):
    def __init__(self,
                 lstn_model,
                 gpu_id=0,
                 long_term_mem_gap=9999,
                 short_term_mem_skip=1,
                 max_lstn_obj_num=None):
        super().__init__(lstn_model, gpu_id, long_term_mem_gap,
                         short_term_mem_skip, max_lstn_obj_num)

    def add_reference_frame(self, img, mask, obj_nums, frame_step=-1):
        if isinstance(obj_nums, list):
            obj_nums = obj_nums[0]
        self.obj_nums = obj_nums
        dslstt_num = max(np.ceil(obj_nums / self.max_lstn_obj_num), 1)
        while (dslstt_num > len(self.lstn_engines)):
            new_engine = DSLSTTEngine(self.lstn, self.gpu_id,
                                     self.long_term_mem_gap,
                                     self.short_term_mem_skip)
            new_engine.eval()
            self.lstn_engines.append(new_engine)

        separated_masks, separated_obj_nums = self.separate_mask(
            mask, obj_nums)
        img_embs = None
        for lstn_engine, separated_mask, separated_obj_num in zip(
                self.lstn_engines, separated_masks, separated_obj_nums):
            lstn_engine.add_reference_frame(img,
                                           separated_mask,
                                           obj_nums=[separated_obj_num],
                                           frame_step=frame_step,
                                           img_embs=img_embs)
            if img_embs is None:  # reuse image embeddings
                img_embs = lstn_engine.curr_enc_embs

        self.update_size()
