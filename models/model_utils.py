from robustness import imagenet_models
from models.bagnet_custom import bagnetcustom32, bagnetcustom96thin
import torch 
import dill
import json
import os
from collections import namedtuple
from argparse import Namespace

TrainArgsWrapper = namedtuple('TrainArgsWrapper', [
                              'epochs', 'lr', 'momentum', 'weight_decay', 'custom_lr_multiplier', 'step_lr',
                              'regularizer', 'step_lr_gamma', 'classes', 'out_dir', 'additional_transform', 'ds_stats',
                              'batch_size', 'workers', 'mixed_precision'])

MODELS = imagenet_models.__dict__
CUSTOM_MODELS = {
    'bagnetcustom32': bagnetcustom32,
    'bagnetcustom96thin': bagnetcustom96thin,
}
MODELS.update(CUSTOM_MODELS)

COMMON_MODEL_ARGS = {
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'custom_lr_multiplier': None,
    'regularizer': None,
    'workers': 10,
    'mixed_precision': False,
}

BATCH_SIZE = { # img size to batch size
    96: 64,
    32: 128,
}

PREFIX_MODEL_ARGS = {
    'bagnet17': {'step_lr': 100, 'lr': 0.01, 'step_lr_gamma': 0.5},
    'bagnetcustom': {'step_lr': 100, 'lr': 0.01, 'step_lr_gamma': 0.5},
    'bagnet9': {'step_lr': 200, 'lr': 0.05, 'step_lr_gamma': 0.5},
    'patchnet': {'step_lr': 100, 'lr': 0.01, 'step_lr_gamma': 0.5},
}

DEFAULT_MODEL_ARGS = {
    'step_lr': 50,
    'lr': 0.01,
    'step_lr_gamma': 0.5,
}

def load_model_from_checkpoint_dir(input_dir, ds_class, out_dir):
    with open(os.path.join(input_dir, 'env.json'), 'r') as f:
        env = json.load(f)
        args = Namespace(**env)
    ckpt = os.path.join(input_dir, 'checkpoint.pt.best')
    input_name = os.path.basename(os.path.normpath(input_dir))
    out_dir = os.path.join(out_dir, input_name)
    os.makedirs(out_dir, exist_ok=True)
    return make_and_restore_model(arch_name=args.arch, ds_class=ds_class, resume_path=ckpt,
                                  train_args=args, additional_transform=args.additional_transform,
                                  out_dir=out_dir)
    

def make_model(arch_name, num_classes=10):
    model = MODELS[arch_name](num_classes=num_classes)
    model_args = {}
    model_args.update(COMMON_MODEL_ARGS)
    model_args.update(DEFAULT_MODEL_ARGS)
    for k,v in PREFIX_MODEL_ARGS.items():
        if k in arch_name:
            model_args.update(v)
            break
    return model, model_args

def make_and_restore_model(arch_name, ds_class, resume_path, train_args, additional_transform, out_dir):
    classes = ds_class.CLASS_NAMES
    model, model_args = make_model(arch_name, len(classes))
    model_args['epochs'] = train_args.epochs
    model_args['lr'] = train_args.lr
    model_args['step_lr'] = train_args.step_lr
    model_args['step_lr_gamma'] = train_args.step_lr_gamma
    model_args['additional_transform'] = additional_transform
    model_args['out_dir'] = out_dir
    model_args['classes'] = classes
    model_args['ds_stats'] = (ds_class.MEAN, ds_class.STD, ds_class.SIZE)
    model_args['batch_size'] = BATCH_SIZE.get(ds_class.SIZE, 64)
    print(model_args)
    with open(os.path.join(out_dir, 'args.json'), 'w') as outfile:
        json.dump(model_args, outfile)

    model_args_wrapper = TrainArgsWrapper(**model_args)

    # optionally resume from a checkpoint
    checkpoint = None
    if resume_path and os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path, pickle_module=dill)
        
        # Makes us able to load models saved with legacy versions
        state_dict_path = 'model'
        if not ('model' in checkpoint):
            state_dict_path = 'state_dict'

        sd = checkpoint[state_dict_path]
        sd = {k[len('module.'):]:v for k,v in sd.items()}
        model.load_state_dict(sd)
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
    elif resume_path:
        error_msg = "=> no checkpoint found at '{}'".format(resume_path)
        raise ValueError(error_msg)

    model = model.cuda()
    return model, model_args_wrapper, checkpoint
