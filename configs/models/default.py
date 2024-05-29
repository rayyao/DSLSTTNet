from .default_vgb import DefaultModelConfig as BaseConfig


class DefaultModelConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = 'DSLSTTDefault'
        self.MODEL_VSD = 'dslstt'
        self.MODEL_ENGINE = 'dslsttengine'
        self.MODEL_DECODER_INTERMEDIATE_LSAB = False
        self.MODEL_SELF_HEADS = 1
        self.MODEL_ATT_HEADS = 1
