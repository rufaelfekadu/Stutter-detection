from yacs.config import CfgNode as CN


_C = CN()

_C.epochs = 10
_C.task_type = 'mtl'
_C.model = 'lstm'
_C.optimizer = 'adam'
_C.lr = 0.001
_C.batch_size = 32
_C.device = 'cuda'
_C.seed = 42
_C.log_interval = 10
_C.log_dir = 'outputs/logs'
_C.save_dir = 'outputs/checkpoints'
_C.data_dir = 'datasets'

