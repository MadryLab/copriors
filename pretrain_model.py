import os
import git
import torch as ch
import torch
from tensorboardX import SummaryWriter
import argparse
from datasets.co_training import BaseTrainer
import json
from datasets.datasets import COPRIOR_DATASETS
from models.model_utils import make_and_restore_model
import os
from utils.logging import log_images_hook
from cox.store import Store
def main(args):
    # MAKE DATASET AND LOADERS
    dataset_name = args.dataset
    data_path = args.data_path

    ds_class = COPRIOR_DATASETS[dataset_name](data_path)
    classes = ds_class.CLASS_NAMES
    train_ds = ds_class.get_dataset('train')
    val_ds = ds_class.get_dataset('val' if args.use_val else 'test')
    
    model, model_args, checkpoint = make_and_restore_model(arch_name=args.arch, ds_class=ds_class,
                                                           resume_path=args.resume_path,
                                                           train_args=args,
                                                           additional_transform=args.additional_transform,
                                                           out_dir=args.out_dir)
    writer = SummaryWriter(args.out_dir)
    store = Store(args.out_dir)
    metadata_keys = {'dataset': str,
                     'use_val': bool,
                     'arch': str,
                     'epochs': int,
                     'lr': float,
                     'step_lr': int,
                     'step_lr_gamma': float,
                     'additional_transform': str}
    store.add_table('metadata', metadata_keys)
    args_dict = args.__dict__
    store['metadata'].append_row({k: args_dict[k] for k in metadata_keys.keys()})

    for mode in ['train', 'val']:
        store.add_table(mode, {'loss': float,
                               'acc': float,
                               'epoch': int})

    base_trainer = BaseTrainer(train_dataset=train_ds, unlabelled_dataset=None,
                               val_dataset=val_ds, writer=writer, store=store,
                               log_hook=log_images_hook, spurious=args.spurious)
    base_trainer.run(model_args, model, val_iters=25, checkpoint_iters=50,
                     checkpoint=checkpoint)
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='name of dataset')
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--use_val', action='store_true', default=False,
                                     help='use a training fold for validation')
    parser.add_argument('--out-dir', type=str, help='path to dump output')
    parser.add_argument('--arch', type=str, help='name of model architecture')
    parser.add_argument('--resume_path', type=str, default=None, help='path to load a previous checkpoint')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--step_lr', type=int, default=50, help='epochs before LR drop')
    parser.add_argument('--step_lr_gamma', type=float, default=0.1, help='LR drop multiplier')
    parser.add_argument('--additional-transform', type=str, help='type of additional transform')
    parser.add_argument('--spurious', type=str, default=None,
                                      help='add a spurious correlation to the'
                                           'training dataset')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args.__dict__)
    if os.path.exists('job_parameters.json'):
        with open('job_parameters.json') as f:
            job_params = json.load(f)
        for k, v in job_params.items():
            assert args.__contains__(k) # catch typos
            args.__setattr__(k, v)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'env.json'), 'w') as f:
        json.dump(args.__dict__, f)
    main(args)
