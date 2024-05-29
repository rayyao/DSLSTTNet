from networks.engines.vgb_engine import VGBEngine, VGBInferEngine
from networks.engines.dslstt_engine import DSLSTTEngine, DSLSTTInferEngine

def build_engine(name, phase='train', **kwargs):
    if name == 'vgbengine':
        if phase == 'train':
            return VGBEngine(**kwargs)
        elif phase == 'eval':
            return VGBInferEngine(**kwargs)
        else:
            raise NotImplementedError
    elif name == 'dslsttengine':
        if phase == 'train':
            return DSLSTTEngine(**kwargs)
        elif phase == 'eval':
            return DSLSTTInferEngine(**kwargs)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
