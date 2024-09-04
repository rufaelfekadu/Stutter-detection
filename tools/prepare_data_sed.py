from stutter.utils.data import construct_labels
import glob
import numpy as np
import torch
import argparse



def main(args):

    label_paths = glob.glob(args.label_paths)
    n_frames =args.num_frames
    n_classes = args.num_classes

    labels = np.zeros((len(label_paths), n_frames, n_classes))
    for i, label_path in enumerate(label_paths):
        labels[i] = construct_labels(label_path, n_frames, n_classes)

    labels = torch.tensor(labels, dtype=torch.float32)
    torch.save({
        'labels': labels,
        'label_paths': label_paths,
    }, 'datasets/fluencybank/ds_30/reading/sed_labels.pt')

if __name__ == '__main__':

    label_paths= 'datasets/fluencybank/ds_30/reading/label/*/sed/*ref.txt'
    
    parser = argparse.ArgumentParser(description='Prepare SED labels for FluencyBank')
    parser.add_argument('--label_paths', type=str, default=label_paths, help='dataset name')
    parser.add_argument('--num_frames', type=str, default='reading', help='task name')
    parser.add_argument('--num_classes', type=int, default=5, help='number of classes')
    args = parser.parse_args()
    
    main(args)
