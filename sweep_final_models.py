import os
import git
import torch as ch
import torch
from tensorboardX import SummaryWriter
import argparse
from datasets.co_training import GTSingleStep
import json
import numpy as np
from datasets.datasets import COPRIOR_DATASETS
from models.model_utils import make_and_restore_model, load_model_from_checkpoint_dir
import os
from utils.logging import log_images_hook
from robustness.tools import folder
from cox.store import Store

def main(args):
    store = Store(args.out_dir)

    metadata_keys = {'dataset': str,
                     'indices_dir': str,
                     'use_gt': bool,
                     'dedup': bool,
                     'resample': bool,
                     'fraction': float,
                     'arch': str,
                     'epochs': int,
                     'lr': float,
                     'step_lr': int,
                     'step_lr_gamma': float,
                     'additional_transform': str,
                     'spurious': str}
    store.add_table('metadata', metadata_keys)
    args_dict = args.__dict__
    store['metadata'].append_row({k: args_dict[k] for k in metadata_keys.keys()})
    for mode in ['train', 'val']:
        store.add_table(mode,
                        {'loss': float,
                         'acc': float,
                         'epoch': int})
    store.add_table('main',
                    {'era': int,
                     'acc': float})

    # MAKE DATASET AND LOADERS
    dataset_name = args.dataset
    data_path = args.data_path

    ds_class = COPRIOR_DATASETS[dataset_name](data_path)
    classes = ds_class.CLASS_NAMES
    train_ds = ds_class.get_dataset('train')
    val_ds = ds_class.get_dataset('test')
    unlabelled_ds = ds_class.get_dataset('unlabeled')

    main_outdir = os.path.join(args.out_dir, f'total')
    os.makedirs(main_outdir, exist_ok=True)
    main_writer = SummaryWriter(main_outdir)

    for era in [1, 2, 5, 10, 15, 20]:
        print("ERA", era)
        indices_path = os.path.join(args.indices_dir, f'indices_era_{era}.pt')
        labels_path =  os.path.join(args.indices_dir, f'ys_era_{era}.pt')
        if not os.path.exists(indices_path):
            break
        indices = ch.load(indices_path)
        if args.use_gt:
            labels=None
            if args.dedup:
                indices = np.unique(indices)
            if args.resample: # TODO not fully kosher
                indices = np.random.choice(np.arange(max(indices) + 1),
                                        len(indices),
                                        replace=False)
        else:
            labels = ch.load(labels_path)
        
        out_dir = os.path.join(args.out_dir, f'era_{era}')
        os.makedirs(out_dir, exist_ok=True)
        model, model_args, checkpoint = make_and_restore_model(arch_name=args.arch, ds_class=ds_class,
                                                            resume_path=args.resume_path, train_args=args,
                                                            additional_transform=args.additional_transform,
                                                            out_dir=out_dir)
        writer = SummaryWriter(out_dir)

        gt_single_step = GTSingleStep(train_dataset=train_ds,
                                    unlabelled_dataset=unlabelled_ds,
                                    val_dataset=val_ds, store=store,
                                    writer=writer, log_hook=log_images_hook,
                                    start_fraction=args.fraction, indices=indices,
                                    out_dir=out_dir,
                                    labels=labels, spurious=args.spurious)
        _, prec1 = gt_single_step.run(epochs=args.epochs, model=model, model_args=model_args,
                                     out_dir=out_dir, val_iters=25, checkpoint_iters=50)
        main_writer.add_scalar(f'total_val_prec1', prec1, era)
        store['main'].append_row({'era': era, 'acc': prec1})
    
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='name of dataset')
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--out-dir', type=str, help='path to dump output')
    parser.add_argument('--indices-dir', type=str, default=None)
    parser.add_argument('--use-gt', action='store_true')
    parser.add_argument('--dedup', action='store_true')
    parser.add_argument('--resample', action='store_true')
    parser.add_argument('--fraction', type=float, default=0.05)

    parser.add_argument('--arch', type=str, help='name of model architecture')
    parser.add_argument('--resume_path', type=str, default=None, help='path to load a previous checkpoint')
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--step_lr', type=int, default=50, help='epochs between LR drops')
    parser.add_argument('--step_lr_gamma', type=float, default=0.1, help='LR drop multiplier')
    parser.add_argument('--additional-transform', type=str, default='NONE', help='type of additional transform')
    parser.add_argument('--eras', type=int, default=10)
    parser.add_argument('--spurious', type=str, default=None,
                                      help='add a spurious correlation to the'
                                           'training dataset')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if os.path.exists('job_parameters.json'):
        with open('job_parameters.json') as f:
            job_params = json.load(f)
        for k, v in job_params.items():
            assert args.__contains__(k) 
            args.__setattr__(k, v)

    os.makedirs(args.out_dir, exist_ok=True)
    print(args.__dict__)
    with open(os.path.join(args.out_dir, 'env.json'), 'w') as f:
        json.dump(args.__dict__, f)
    main(args)
