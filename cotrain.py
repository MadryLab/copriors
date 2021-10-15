import os
import git
import torch as ch
import torch
from tensorboardX import SummaryWriter
import argparse
from datasets.co_training import CoTrainDatasetTrainer
import json
from datasets.datasets import COPRIOR_DATASETS
from models.model_utils import make_and_restore_model, load_model_from_checkpoint_dir
import os
from utils.logging import log_images_hook
from cox.store import Store
def main(args):
    store = Store(args.out_dir)

    metadata_keys = {'dataset': str,
                     'use_val': bool,
                     'input_dirs': str,
                     'fraction': float,
                     'epochs_per_era': int,
                     'eras': int,
                     'strategy': str,
                     'pure': bool,
                     'arch': str,
                     'epochs': int,
                     'lr': float,
                     'step_lr': int,
                     'step_lr_gamma': float,
                     'additional_transform': str,
                     'spurious': str,
                     'use_gt': bool}
    store.add_table('metadata', metadata_keys)
    args_dict = args.__dict__
    store['metadata'].append_row({k: args_dict[k] for k in metadata_keys.keys()})
    for mode in ['train', 'val']:
        store.add_table(mode,
                        {'loss': float,
                         'acc': float,
                         'epoch': int})
        for i in range(args.eras): # hacky but backward compatible
            store.add_table(f'{i}_cotrain_{mode}', 
                            {'loss': float,
                             'acc': float,
                             'epoch': int})
    store.add_table('unlabeled',
                    {'model': int,
                     'era': int,
                     'acc': float,
                     'selected_acc': float,
                     'num_points': int})

    # MAKE DATASET AND LOADERS
    dataset_name = args.dataset
    data_path = args.data_path

    ds_class = COPRIOR_DATASETS[dataset_name](data_path)
    classes = ds_class.CLASS_NAMES
    train_ds = ds_class.get_dataset('train')
    val_ds = ds_class.get_dataset('val' if args.use_val else 'test')
    unlabelled_ds = ds_class.get_dataset('unlabeled')

    input_models = []
    input_model_args = []
    for input_dir in args.input_dirs:
        model, model_args, _ = load_model_from_checkpoint_dir(input_dir, ds_class, args.out_dir)
        input_models.append(model)
        input_model_args.append(model_args)

    model, model_args, checkpoint = make_and_restore_model(arch_name=args.arch, ds_class=ds_class,
                                                           resume_path=args.resume_path, train_args=args,
                                                           additional_transform=args.additional_transform,
                                                           out_dir=args.out_dir)
    writer = SummaryWriter(args.out_dir)
   
    cotrainer = CoTrainDatasetTrainer(train_dataset=train_ds,
                                      unlabelled_dataset=unlabelled_ds,
                                      val_dataset=val_ds, writer=writer,
                                      store=store, log_hook=log_images_hook,
                                      start_fraction=args.fraction,
                                      out_dir=args.out_dir,
                                      strategy=args.strategy,
                                      pure=args.pure,
                                      spurious=args.spurious,
                                      use_gt=args.use_gt,
                                      num_classes=len(ds_class.CLASS_NAMES))
    cotrainer.co_train_models(input_model_args=input_model_args,
                              input_models=input_models,
                              co_training_eras=args.eras,
                              co_training_epochs_per_era=args.epochs_per_era,
                              final_model=model,
                              final_model_args=model_args,
                              val_iters=25, checkpoint_iters=50)
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='name of dataset')
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--out-dir', type=str, help='path to dump output')
    parser.add_argument('--use_val', action='store_true', default=False,
                                     help='use a training fold for validation')
    
    parser.add_argument('--input-dirs', type=str, action='append')
    parser.add_argument('--fraction', type=float, default=0.1)
    parser.add_argument('--epochs_per_era', type=int, default=300)
    parser.add_argument('--eras', type=int, default=5)
    parser.add_argument('--strategy', type=str, default='STANDARD_CONSTANT')
    parser.add_argument('--pure', action='store_true', help='use pure cotraining')

    # final model arguments
    parser.add_argument('--arch', type=str, help='name of model architecture')
    parser.add_argument('--resume_path', type=str, default=None, help='path to load a previous checkpoint')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--step_lr', type=int, default=50, help='epochs before LR drop')
    parser.add_argument('--step_lr_gamma', type=float, default=0.1, help='LR drop multiplier')
    parser.add_argument('--additional-transform', type=str, default='NONE', help='type of additional transform')

    parser.add_argument('--spurious', type=str, default=None,
                                      help='add a spurious correlation to the'
                                           'training dataset')
    parser.add_argument('--use-gt', action='store_true', help='use gt labels')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if os.path.exists('job_parameters.json'):
        with open('job_parameters.json') as f:
            job_params = json.load(f)
        for k, v in job_params.items():
            assert args.__contains__(k) # catch typos)
            args.__setattr__(k, v)

    os.makedirs(args.out_dir, exist_ok=True)
    print(args.__dict__)
    with open(os.path.join(args.out_dir, 'env.json'), 'w') as f:
        json.dump(args.__dict__, f)
    main(args)
