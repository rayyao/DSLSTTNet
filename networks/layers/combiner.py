import torch.nn as nn
# import ltr.models.layers.filter as filter_layer
import math
# from pytracking import TensorList
import torch


class Combiner(nn.Module):
    """ Target model constituting a single conv layer, along with the few-shot learner used to obtain the target model
        parameters (referred to as filter), i.e. weights of the conv layer
    """
    def __init__(self, num_filters):
        super().__init__()

        # self.tm_1 = nn.Linear(num_filters, num_filters, bias=False)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.t_conv = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        self.norm = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()
        # self.tm_2 = nn.Linear(num_filters, num_filters, bias=False)
        # self.tm_2 = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        # self.norm_2 = nn.BatchNorm1d(num_filters)
        # self.relu_2 = nn.ReLU()
        # self.tm_3 = nn.Linear(num_filters, num_filters, bias=False)
        # self.tm_3 = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        # self.tr_1 = nn.Linear(num_filters, num_filters, bias=False)
        # self.tr_2 = nn.Linear(num_filters, num_filters, bias=False)
        # self.tr_3 = nn.Linear(num_filters, num_filters, bias=False)
        # self.tr_1 = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        # self.tr_2 = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        # self.tr_3 = nn.Conv1d(num_filters, num_filters, kernel_size=1)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, mask_encoding_pred_tm, mask_encoding_pred_tr):
        """ the mask should be 3d"""
        assert mask_encoding_pred_tm.dim() == 3
        assert mask_encoding_pred_tr.dim() == 3

        num_sequences = mask_encoding_pred_tm.shape[1]
        HW, C = mask_encoding_pred_tm.shape[0], mask_encoding_pred_tm.shape[2]

        if mask_encoding_pred_tm.dim() == 3 and mask_encoding_pred_tr.dim() == 3:

            # mask_encoding_pred_tm = mask_encoding_pred_tm.view(HW, -1, C)
            # mask_encoding_pred_tm = mask_encoding_pred_tm.permute(1, 2, 0)
            # # tm_res = self.gap(mask_encoding_pred_tm)
            # tm_res = self.t_conv(mask_encoding_pred_tm)
            # # tm_res = self.t_conv(tm_res)
            # tm_res = self.norm(tm_res)
            # tm_res = self.relu(tm_res)
            # tm_res = self.t_conv(tm_res)
            # tm_res = self.norm(tm_res)
            # tm_res = self.relu(tm_res)
            # tm_res = self.t_conv(tm_res)
            # gate_wei_tm = self.sigmoid(tm_res)
            #
            # mask_encoding_pred_tr = mask_encoding_pred_tr.view(HW, -1, C)
            # mask_encoding_pred_tr = mask_encoding_pred_tr.permute(1, 2, 0)
            # # tr_res = self.gap(mask_encoding_pred_tr)
            # tr_res = self.t_conv(mask_encoding_pred_tr)
            # # tr_res = self.t_conv(tr_res)
            # tr_res = self.norm(tr_res)
            # tr_res = self.relu(tr_res)
            # tr_res = self.t_conv(tr_res)
            # tr_res = self.norm(tr_res)
            # tr_res = self.relu(tr_res)
            # tr_res = self.t_conv(tr_res)
            # gate_wei_tr = self.sigmoid(tr_res)
            #
            # mask_encoding_pred_tm = gate_wei_tm * mask_encoding_pred_tm + mask_encoding_pred_tm
            # mask_encoding_pred_tm = mask_encoding_pred_tm.permute(2, 0, 1)
            # mask_encoding_pred_tm = mask_encoding_pred_tm.view(HW, num_sequences, C)
            #
            # mask_encoding_pred_tr = gate_wei_tr * mask_encoding_pred_tr + mask_encoding_pred_tr
            # mask_encoding_pred_tr = mask_encoding_pred_tr.permute(2, 0, 1)
            # mask_encoding_pred_tr = mask_encoding_pred_tr.view(HW, num_sequences, C)

            mask_encoding_pred_tm = mask_encoding_pred_tm.view(HW, -1, C)
            mask_encoding_pred_tm = mask_encoding_pred_tm.permute(0, 2, 1)
            # tm_res = self.gap(mask_encoding_pred_tm)
            tm_res = self.t_conv(mask_encoding_pred_tm)
            tm_res = self.t_conv(tm_res)
            tm_res = self.norm(tm_res)
            tm_res = self.relu(tm_res)
            tm_res = self.t_conv(tm_res)
            tm_res = self.norm(tm_res)
            tm_res = self.relu(tm_res)
            tm_res = self.t_conv(tm_res)
            gate_wei_tm = self.sigmoid(tm_res)

            mask_encoding_pred_tr = mask_encoding_pred_tr.view(HW, -1, C)
            mask_encoding_pred_tr = mask_encoding_pred_tr.permute(0, 2, 1)
            # tr_res = self.gap(mask_encoding_pred_tr)
            tr_res = self.t_conv(mask_encoding_pred_tr)
            tr_res = self.t_conv(tr_res)
            tr_res = self.norm(tr_res)
            tr_res = self.relu(tr_res)
            tr_res = self.t_conv(tr_res)
            tr_res = self.norm(tr_res)
            tr_res = self.relu(tr_res)
            tr_res = self.t_conv(tr_res)
            gate_wei_tr = self.sigmoid(tr_res)

            mask_encoding_pred_tm = gate_wei_tm * mask_encoding_pred_tm + mask_encoding_pred_tm
            mask_encoding_pred_tm = mask_encoding_pred_tm.permute(0, 2, 1)
            mask_encoding_pred_tm = mask_encoding_pred_tm.view(HW, num_sequences, C)

            mask_encoding_pred_tr = gate_wei_tr * mask_encoding_pred_tr + mask_encoding_pred_tr
            mask_encoding_pred_tr = mask_encoding_pred_tr.permute(0, 2, 1)
            mask_encoding_pred_tr = mask_encoding_pred_tr.view(HW, num_sequences, C)

        mask_encodings = mask_encoding_pred_tm + mask_encoding_pred_tr

        return mask_encodings