from .default import DefaultEngineConfig


class EngineConfig(DefaultEngineConfig):
    def __init__(self, exp_name='default', model='swin_tiny'):
        super().__init__(exp_name, model)
        
        self.init_dir()
