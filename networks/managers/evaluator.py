import os
import time
import datetime as datetime
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dslstt.dataloaders.eval_demo_data import VSDTest
from dslstt.dataloaders.eval_datasets import ViSha_Test
import dataloaders.video_transforms as tr

from utils.image import flip_tensor, save_mask
from utils.checkpoint import load_network

from networks.models import build_vos_model
from networks.engines import build_engine
import cv2




class EvaluatorVisha(object):
    def __init__(self, cfg, rank=0):
        self.gpu = rank
        self.rank = rank
        self.cfg = cfg

        torch.cuda.set_device(self.gpu)
        self.model = build_vos_model(cfg.MODEL_VSD, cfg).cuda(self.gpu)

        # load checkpionts
        self.ckpt = str(cfg.TEST_CKPT_STEP)
        ckpt_path = os.path.join(cfg.TEST_CKPT_PATH, 'save_step_%s.pth' % self.ckpt)
        self.model, removed_dict = load_network(self.model, ckpt_path, self.gpu)

        # prepare dataset
        self.prepare_dataset()


    def prepare_dataset(self):
        cfg = self.cfg
        eval_transforms = transforms.Compose([
            tr.MultiRestrictSize(cfg.TEST_MAX_SHORT_EDGE,
                                 cfg.TEST_MAX_LONG_EDGE, cfg.TEST_FLIP,
                                 cfg.TEST_MULTISCALE, cfg.MODEL_ALIGN_CORNERS),
            tr.MultiToTensor()
        ])

        # _frame = cv2.imread("/data/wangyh/data4/video_shadow_detection/32/demo1.jpg")
        # _frame = cv2.resize(_frame, (600, 680), interpolation=cv2.INTER_CUBIC)
        # _frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)
        # cfg.IMAGE_LIST = [_frame] * 10
        # _label = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)
        # cfg.MASK_LIST = [_label]
        eval_name = '{}_ckpt_{}'.format(cfg.EXP_NAME, self.ckpt)
        visha_data_name = cfg.TEST_DATASET_PATH.split('/')[-1]
        self.save_log = os.path.join(cfg.DIR_EVALUATION, cfg.EXP_NAME, visha_data_name)
        self.result_root = os.path.join(cfg.DIR_EVALUATION, cfg.EXP_NAME, visha_data_name, eval_name)
        self.dataset = ViSha_Test(root=cfg.TEST_DATASET_PATH, root2=cfg.TEST_DATASET_PATH2, transform=eval_transforms, result_root=self.result_root)


    def evaluating(self):
        cfg = self.cfg
        self.model.eval()
        total_time = 0
        total_frame = 0
        total_sfps = 0
        video_num = 0
        total_video_num = len(self.dataset)

        all_engines = []
        start_eval_time = time.time()
        with torch.no_grad():
            for seq_idx, seq_dataset in enumerate(self.dataset):
                video_num += 1
                for engine in all_engines:
                    engine.restart_engine()

                seq_name = seq_dataset.seq_name
                print('GPU {} - Processing Seq {} [{}/{}]:'.format(self.gpu, seq_name, video_num, total_video_num))
                torch.cuda.empty_cache()
                seq_dataloader = DataLoader(seq_dataset, batch_size=1, shuffle=False, num_workers=cfg.TEST_WORKERS, pin_memory=True)
                
                seq_total_time = 0
                seq_total_frame = 0
                seq_timers = []
                seq_pred_masks = {'dense': [], 'sparse': []}
                for frame_idx, samples in enumerate(seq_dataloader):
                    all_preds = []
                    aug_num = len(samples)

                    for aug_idx in range(aug_num):
                        if len(all_engines) <= aug_idx:
                            all_engines.append(
                                build_engine(cfg.MODEL_ENGINE,
                                                phase='eval',
                                                lstn_model=self.model,
                                                gpu_id=self.gpu,
                                                long_term_mem_gap=self.cfg.TEST_LONG_TERM_MEM_GAP,
                                                short_term_mem_skip=self.cfg.TEST_SHORT_TERM_MEM_SKIP
                                                ))
                            all_engines[-1].eval()

                        if aug_num > 1:  # if use test-time augmentation
                            torch.cuda.empty_cache()  # release GPU memory

                        engine = all_engines[aug_idx]
                        sample = samples[aug_idx]

                        obj_nums = sample['meta']['obj_num']
                        ori_height = sample['meta']['height']
                        ori_width = sample['meta']['width']
                        obj_idx = sample['meta']['obj_idx']
                        imgname = sample['meta']['current_name']
                        obj_nums = [int(obj_num) for obj_num in obj_nums]
                        obj_idx = [int(_obj_idx) for _obj_idx in obj_idx]
                        # image to cuda
                        current_img = sample['current_img'].type(torch.FloatTensor)
                        current_img = current_img.cuda(self.gpu, non_blocking=True)
                        sample['current_img'] = current_img
                        # label to cuda
                        if 'current_label' in sample.keys():
                            current_label = sample['current_label'].cuda(self.gpu, non_blocking=True).float()
                        else:
                            current_label = None

                        #############################################################
                        # start forward testing
                        if frame_idx == 0:
                            _current_label = F.interpolate(current_label, size=current_img.size()[2:], mode="nearest")
                            engine.add_reference_frame(current_img, _current_label, frame_step=0, obj_nums=obj_nums)
                        else:
                            if aug_idx == 0:
                                seq_timers.append([])
                                now_timer = torch.cuda.Event(enable_timing=True)
                                now_timer.record()
                                seq_timers[-1].append(now_timer)

                            engine.match_propogate_one_frame(current_img)  # forward
                            pred_logit = engine.decode_current_logits((ori_height, ori_width))  # output logits
                            pred_prob = torch.softmax(pred_logit, dim=1)  # 经过softmax
                            all_preds.append(pred_prob)


                    if frame_idx > 0:
                        all_pred_probs = [torch.mean(pred, dim=0, keepdim=True) for pred in all_preds]
                        all_pred_labels = [torch.argmax(prob, dim=1, keepdim=True).float() for prob in all_pred_probs]

                        cat_all_preds = torch.cat(all_preds, dim=0)
                        pred_prob = torch.mean(cat_all_preds, dim=0, keepdim=True)
                        pred_label = torch.argmax(pred_prob, dim=1, keepdim=True).float()


                        if not cfg.MODEL_USE_PREV_PROB:
                            for aug_idx in range(len(samples)):
                                engine = all_engines[aug_idx]
                                current_label = all_pred_labels[aug_idx]
                                current_label = F.interpolate(current_label, size=engine.input_size_2d, mode="nearest")
                                engine.update_memory(current_label)

                        now_timer = torch.cuda.Event(enable_timing=True)
                        now_timer.record()
                        seq_timers[-1].append((now_timer))

                        if cfg.TEST_FRAME_LOG:
                            torch.cuda.synchronize()
                            one_frametime = seq_timers[-1][0].elapsed_time(seq_timers[-1][1]) / 1e3
                            obj_num = obj_nums[0]
                            print('GPU {} - Frame: {} - Obj Num: {}, Time: {}ms'.format(self.gpu, "video_frame", obj_num, int(one_frametime * 1e3)))
                        # Save result
                        seq_pred_masks['dense'].append({
                            'path': os.path.join(self.result_root, seq_name, imgname[0].split('.')[0] + '.png'),
                            'mask': pred_label,
                            'obj_idx': obj_idx
                        })


                # Save result
                for mask_result in seq_pred_masks['dense'] + seq_pred_masks['sparse']:
                    save_mask(mask_result['mask'].squeeze(0).squeeze(0), mask_result['path'], mask_result['obj_idx'])
                del (seq_pred_masks)

                for timer in seq_timers:
                    torch.cuda.synchronize()
                    one_frametime = timer[0].elapsed_time(timer[1]) / 1e3
                    seq_total_time += one_frametime
                    seq_total_frame += 1
                del (seq_timers)

                seq_avg_time_per_frame = seq_total_time / seq_total_frame
                total_time += seq_total_time
                total_frame += seq_total_frame
                total_avg_time_per_frame = total_time / total_frame
                total_sfps += seq_avg_time_per_frame
                max_mem = torch.cuda.max_memory_allocated(device=self.gpu) / (1024.**3) 
                # log
                log = "GPU {} - Seq {} - FPS: {:.2f}. All-Frame FPS: {:.2f}, Max Mem: {:.2f}G" \
                    .format(self.gpu, "video", 1. / seq_avg_time_per_frame, 1. / total_avg_time_per_frame,  max_mem)
                print(log)

        if self.rank == 0:
            end_eval_time = time.time()
            total_eval_time = str(datetime.timedelta(seconds=int(end_eval_time - start_eval_time)))
            log_time = "Total evaluation time: {}".format(total_eval_time)
            print(log_time)