from yacs.config import CfgNode as CN


_C = CN()

_C.name = 'stutter-detection'
_C.seed = 42
_C.tasks = ['t1', 't2']

# dataset configs
_C.data = CN()

_C.data.name = 'sep28k'
_C.data.root = '/Users/apple/Documents/Projects/RA/stutter-detection/datasets/sep28k/clips'
_C.data.label_path = '/Users/apple/Documents/Projects/RA/stutter-detection/datasets/sep28k/SEP-28k_labels_new.csv'
_C.data.ckpt = '/Users/apple/Documents/Projects/RA/stutter-detection/datasets/sep28k/sep28k.pt'

# data loader configs


# model configs
_C.model = CN()
_C.model.name = 'lstm'
_C.model.input_size = 40
_C.model.hidden_size = 64
_C.model.num_layers = 1
_C.model.output_size = 6
_C.model.dropout = 0.5
_C.model.seq_len = 301

# convlstm
_C.model.emb_dim = 64

# training configs
_C.solver = CN()
_C.solver.device = 'mps'
_C.solver.epochs = 10

_C.solver.optimizer = 'adam'
_C.solver.lr = 0.001

_C.solver.es_patience = 5

_C.solver.batch_size = 32
_C.solver.num_workers = 4

_C.solver.log_interval = 10

# logging and saving
_C.output = CN()
_C.output.log_dir = '../outputs/logs'
_C.output.save_dir = '../outputs/checkpoints'

