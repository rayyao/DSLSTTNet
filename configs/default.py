import os
import importlib


class DefaultEngineConfig():
    def __init__(self, exp_name='default', model='swin_tiny'):
        model_cfg = importlib.import_module('configs.models.'+model).ModelConfig()
        self.__dict__.update(model_cfg.__dict__)  # add model config
        self.EXP_NAME = exp_name + '_' + self.MODEL_NAME

        # initial dataset
        self.DATA_VISHA_REPEAT = 5
        self.DATA_SEQ_LEN = 5
        self.DATA_RANDOM_GAP_DAVIS = 12
        self.DATA_RANDOM_REVERSE_SEQ = True
        # transform
        self.DATA_WORKERS = 16
        self.DATA_MIN_SCALE_FACTOR = 0.7
        self.DATA_MAX_SCALE_FACTOR = 1.3
        self.DATA_SHORT_EDGE_LEN = 480
        self.DATA_RANDOMCROP = (465, 465) if self.MODEL_ALIGN_CORNERS else (464, 464)
        self.DATA_RANDOMFLIP = 0.5

        # load pretrain network
        self.PRETRAIN_MODEL = ''

        # train step
        self.TRAIN_TOTAL_STEPS = 20000

        # weight decay
        self.TRAIN_WEIGHT_DECAY = 0.07
        self.TRAIN_WEIGHT_DECAY_EXCLUSIVE = {
            # 'encoder.': 0.01
        }
        self.TRAIN_WEIGHT_DECAY_EXEMPTION = [
            'absolute_pos_embed',
            'relative_position_bias_table',
            'relative_emb_v',
            'conv_out'
        ]

        # lr
        self.TRAIN_LR = 1e-4
        self.TRAIN_LR_MIN = 2e-5 if 'swin_base' in self.MODEL_ENCODER else 1e-5
        self.TRAIN_LR_POWER = 0.9
        self.TRAIN_LR_ENCODER_RATIO = 0.1
        self.TRAIN_LR_WARM_UP_RATIO = 0.05
        self.TRAIN_LR_COSINE_DECAY = False
        self.TRAIN_LR_RESTART = 1
        self.TRAIN_LR_UPDATE_STEP = 1
        self.TRAIN_AUX_LOSS_WEIGHT = 1.0
        self.TRAIN_AUX_LOSS_RATIO = 1.0
        self.TRAIN_OPT = 'adamw'  # ["adamw", "sgd"]
        self.TRAIN_SGD_MOMENTUM = 0.9
        self.TRAIN_GPUS = 1
        self.TRAIN_BATCH_SIZE = 2  # # # # #
        self.TRAIN_TOP_K_PERCENT_PIXELS = 0.15
        self.TRAIN_SEQ_TRAINING_FREEZE_PARAMS = ['patch_wise_id_bank']
        self.TRAIN_SEQ_TRAINING_START_RATIO = 0.5
        self.TRAIN_HARD_MINING_RATIO = 0.5
        self.TRAIN_EMA_RATIO = 0.1
        self.TRAIN_CLIP_GRAD_NORM = 5.
        self.TRAIN_MAX_KEEP_CKPT = 30  # 最大保存的权重数量，如果超过这个数量，会自动删除之前保存的权重
        self.TRAIN_RESUME_CKPT = None
        self.TRAIN_RESUME_STEP = 0
        self.TRAIN_ENCODER_FREEZE_AT = 2
        self.TRAIN_LSAB_EMB_DROPOUT = 0.
        self.TRAIN_LSAB_ID_DROPOUT = 0.
        self.TRAIN_LSAB_DROPPATH = 0.1
        self.TRAIN_LSAB_DROPPATH_SCALING = False
        self.TRAIN_LSAB_DROPPATH_LST = False
        self.TRAIN_LSAB_LT_DROPOUT = 0.
        self.TRAIN_LSAB_ST_DROPOUT = 0.

        # about log
        self.TRAIN_IMGLOG_STEP = 500  # log image every x steps
        self.TRAIN_LOG_STEP = 20
        self.TRAIN_SAVE_STEP = 10000


        # GPU distribution
        self.DIST_ENABLE = True
        self.DIST_BACKEND = "nccl"  # "gloo"
        self.DIST_URL = "tcp://127.0.0.1:13241"
        self.DIST_START_GPU = 0


        # about test
        self.TEST_GPU_NUM = 1
        self.TEST_FRAME_LOG = False
        self.TEST_DATASET_PATH = ''
        self.TEST_DATASET_PATH2 = ''
        self.TEST_CKPT_PATH = None
        self.TEST_CKPT_STEP = None
        self.TEST_FLIP = False
        self.TEST_MULTISCALE = [1]
        self.TEST_MAX_SHORT_EDGE = None
        self.TEST_MAX_LONG_EDGE = 800 * 1.3
        self.TEST_WORKERS = 4

    def init_dir(self):
        self.DIR_VISHA = ''
        self.DIR_VISHA2 = ''
        self.DIR_RESULT = os.path.join('./checkpoints', self.EXP_NAME)
        self.DIR_CKPT = os.path.join(self.DIR_RESULT, 'ckpt')
        self.DIR_EMA_CKPT = os.path.join(self.DIR_RESULT, 'ema_ckpt')
        self.DIR_IMG_LOG = './img_logs_dslstt'
        self.DIR_EVALUATION = './results'

        for path in [
                self.DIR_RESULT, self.DIR_CKPT, self.DIR_EMA_CKPT,
                self.DIR_EVALUATION, self.DIR_IMG_LOG
            ]:
            if not os.path.isdir(path):
                try:
                    os.makedirs(path)
                except Exception as inst:
                    print(inst)
                    print('Failed to make dir: {}.'.format(path))
