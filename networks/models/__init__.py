from networks.models.vgb import VGB
from networks.models.dslstt import DSLSTT


def build_vos_model(name, cfg, **kwargs):
    if name == 'vgb':
        return VGB(cfg, encoder=cfg.MODEL_ENCODER, **kwargs)
    if name == 'dslstt':
        return DSLSTT(cfg, encoder=cfg.MODEL_ENCODER, **kwargs)
    else:
        raise NotImplementedError
