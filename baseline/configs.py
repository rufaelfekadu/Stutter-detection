from yacs.config import CfgNode as CN


_C = CN()

_C.name = 'stutter-detection'

# dataset configs
_C.data_path = '/Users/apple/Documents/Projects/RA/stutter-detection/datasets/sep28k/clips'
_C.label_path = '/Users/apple/Documents/Projects/RA/stutter-detection/datasets/sep28k/SEP-28k_labels_new.csv'
_C.data_ckpt = '/Users/apple/Documents/Projects/RA/stutter-detection/datasets/sep28k/sep28k.pt'

# data loader configs
_C.num_workers = 4

# model configs
_C.model = 'lstm'
_C.input_size = 40
_C.hidden_size = 64
_C.num_layers = 1
_C.output_size = 6
_C.dropout = 0.5


# training configs
_C.epochs = 10
_C.tasks = ['t1', 't2']
_C.optimizer = 'adam'
_C.lr = 0.001
_C.batch_size = 32
_C.device = 'mps'
_C.seed = 42
_C.log_interval = 10
_C.log_dir = '../outputs/logs'
_C.save_dir = '../outputs/checkpoints'

