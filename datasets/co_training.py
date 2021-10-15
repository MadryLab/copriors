import datasets.transforms as custom_transforms
from datasets.dataset_utils import make_wrappers, get_loader, add_spurious
from utils.accuracy_meter import AccuracyMeter
from robustness.train import make_optimizer_and_schedule
import torch
import os
import dill
from robustness.tools import constants as consts
from robustness.tools import folder
import torch.nn.functional as F
import copy
import numpy as np
if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm


def make_ds_wrapper(ds_list, perform_aug, additional_transform_name, ds_stats):
    if len(ds_stats) == 3:
        mean, std, size = ds_stats
    else: # legacy
        if len(ds_stats) == 2:
            mean, std = ds_stats
        else:
            mean, std = None
        size = 96
    if perform_aug:
        primary_transform = custom_transforms.TRAIN_TRANSFORMS_DEFAULT(size)
    else:
        primary_transform = custom_transforms.TEST_TRANSFORMS_DEFAULT(size)
    additional_transform = custom_transforms.ADDITIONAL_TRANSFORMS[additional_transform_name]()
    return make_wrappers(ds_list, primary_transform, additional_transform)

class BaseTrainer:
    def __init__(self, train_dataset, unlabelled_dataset, val_dataset,
                 writer, store, log_hook, spurious=None):
        if spurious:
            train_dataset = add_spurious(train_dataset, mode=spurious)
        self.train_dataset = train_dataset
        self.unlabelled_dataset = unlabelled_dataset # can be None
        self.val_dataset = val_dataset
        self.writer = writer
        self.store = store
        self.log_hook = log_hook

    def initialize_model(self, arg_wrapper, model, checkpoint=None):
        opt, schedule = make_optimizer_and_schedule(arg_wrapper, model, checkpoint, None)
        # Put the model into parallel mode
        assert not hasattr(model, "module"), "model is already in DataParallel." 
        model = torch.nn.DataParallel(model).cuda()
        return model, opt, schedule

    def train_model(self, arg_wrapper, train_loader, val_loader, model, val_iters=15, checkpoint_iters=15, save_best=True, checkpoint=None):
        num_epochs = arg_wrapper.epochs
        # Reformat and read arguments
        model, opt, schedule = self.initialize_model(arg_wrapper, model, checkpoint)
        best_prec1, start_epoch = (0, 0)
        if checkpoint:
            start_epoch = checkpoint['epoch']

        for epoch in range(start_epoch, num_epochs):
            # train for one epoch
            train_prec1, train_prec5, train_loss = self.model_loop(arg_wrapper=arg_wrapper, loop_name='train', loader=train_loader,
                                                       model=model, opt=opt, epoch=epoch, backward=True)
            if (epoch % val_iters == 0) or (epoch == (num_epochs - 1)): # eval
                with torch.no_grad():
                    test_prec1, test_prec5, test_loss = self.model_loop(arg_wrapper=arg_wrapper, loop_name='val', loader=val_loader,
                                                            model=model, opt=opt, epoch=epoch, backward=False)
            
            if (epoch % checkpoint_iters == 0) or (epoch == (num_epochs - 1)):
                self._save_checkpoint(arg_wrapper, consts.CKPT_NAME_LATEST, model, opt, schedule, epoch)
                if test_prec1 > best_prec1:
                    best_prec1 = test_prec1
                    if save_best:
                        self._save_checkpoint(arg_wrapper, consts.CKPT_NAME_BEST, model, opt, schedule, epoch)
           
            if schedule: 
                schedule.step()

        return model, best_prec1

    def _save_checkpoint(self, arg_wrapper, filename, model, opt, schedule, epoch):
        sd_info = {
            'model':model.state_dict(),
            'optimizer':opt.state_dict(),
            'schedule':(schedule and schedule.state_dict()),
            'epoch': epoch+1,
        }
        ckpt_save_path = os.path.join(arg_wrapper.out_dir, filename)
        torch.save(sd_info, ckpt_save_path, pickle_module=dill)


    def _train_backward(self, arg_wrapper, model, loss, opt):
        reg_term = 0.0
        if arg_wrapper.regularizer is not None:
            reg_term =  arg_wrapper.regularizer(model, inp, target)
        loss = loss + reg_term
        # compute gradient and do SGD step
        opt.zero_grad()
        loss.backward()
        opt.step()

    def model_loop(self, arg_wrapper, loop_name, loader, model, opt, epoch, backward):
        writer = self.writer
        accuracy_meter = AccuracyMeter()
        model = model.train() if backward else model.eval()
        train_criterion = torch.nn.CrossEntropyLoss()
        
        iterator = tqdm(enumerate(loader), total=len(loader))
        for i, (inp, target, meta) in iterator:
            target = target.cuda(non_blocking=True)
            inp = inp.cuda()
            output = model(inp)

            loss = train_criterion(output, target)
            if len(loss.shape) > 0: loss = loss.mean()
            model_logits = output[0] if (type(output) is tuple) else output
            accuracy_meter.update(inp.size(0), model_logits, target, loss, meta)

            if backward:
                self._train_backward(arg_wrapper, model, loss, opt)

            curr_lr = 0 if opt is None else [param_group['lr'] for param_group in opt.param_groups][0]
           
            # USER-DEFINED HOOK
            self.log_hook(model, i, loop_name, inp, target, output, meta, epoch, writer, classes=arg_wrapper.classes , meter_dict=accuracy_meter.stats())

            iterator.set_description(f"{loop_name} Epoch {epoch} | {accuracy_meter.get_desc()} | LR: {curr_lr}")
            iterator.refresh()

        stats = accuracy_meter.stats()
        if writer is not None:
            for d, v in stats.items():
                writer.add_scalar('_'.join([loop_name, d]), v, epoch)
        store = self.store
        if store:
            store[loop_name].append_row({'loss': stats['loss'],
                                         'acc': stats['prec1'],
                                         'epoch': epoch})

        return stats['prec1'], stats['prec5'], stats['loss']

    def run(self, arg_wrapper, model, val_iters, checkpoint_iters, checkpoint=None):
        train_wrap = make_ds_wrapper([self.train_dataset], perform_aug=True, additional_transform_name=arg_wrapper.additional_transform, ds_stats=arg_wrapper.ds_stats)
        train_loader = get_loader(train_wrap, arg_wrapper)
        val_wrap = make_ds_wrapper([self.val_dataset], perform_aug=False, additional_transform_name=arg_wrapper.additional_transform, ds_stats=arg_wrapper.ds_stats)
        val_loader = get_loader(val_wrap, arg_wrapper)
        return self.train_model(arg_wrapper=arg_wrapper, train_loader=train_loader,
                         val_loader=val_loader, model=model, val_iters=val_iters, checkpoint_iters=checkpoint_iters, save_best=True, checkpoint=checkpoint)


class CoTrainDatasetTrainer(BaseTrainer):
    VALID_STRATEGIES = [
        'STANDARD_CONSTANT'
    ]

    def __init__(self, start_fraction, train_dataset, unlabelled_dataset,
                 val_dataset, writer, store, log_hook, out_dir, strategy='STANDARD_CONSTANT',
                 num_classes=10, pure=False, use_gt=False, spurious='NONE'):
        super().__init__(train_dataset, unlabelled_dataset, val_dataset, writer,
                         store, log_hook, spurious=spurious)
        self.start_fraction = start_fraction
        self.strategy = strategy
        self.pure_cotrain = pure
        self.use_gt = use_gt
        assert self.strategy in CoTrainDatasetTrainer.VALID_STRATEGIES, f'{self.strategy}' # atm, only have standard_constant
        self.out_dir = out_dir
        self.num_classes = num_classes

    def label_dataset_with_model(self, loader, model):
        model.eval()
        ims, ys, confs, gts, preds = [], [], [], [], []
        it = tqdm(loader)
        
        for im, gt, meta in it:
            with torch.no_grad():
                pred = model(im.cuda())
            y = pred.cpu().argmax(axis=1)
            p = F.softmax(pred, dim=1)
            conf = p.cpu().max(axis=1)[0]
            
            ims.append(meta['original_img'])
            ys.append(y)
            confs.append(conf)
            gts.append(gt)
            preds.append(p)

        ims = torch.cat(ims, 0)
        ys = torch.cat(ys, 0)
        confs = torch.cat(confs, 0)
        gts = torch.cat(gts, 0)
        preds = torch.cat(preds, 0)

        obj = {'ims': ims, 'ys': ys, 'gts': gts, 'confs': confs, 'preds': preds, 'idxs': torch.arange(len(gts))}
        return obj

    def _filter_obj(self, obj_, indices_):
        return {k: v[indices_] for k,v in obj_.items()} 

    def get_accuracy(self, obj):
        gts, ys = obj['gts'], obj['ys']
        return 0 if len(gts) == 0 else len(gts[gts == ys]) / len(gts)

    def take_top_perc_from_each_class(self, obj, per_class):
        ys = obj['ys']
        compute_acc = (obj['gts'] >= 0).all().item()
        num_classes = ys.max().item() + 1
        all_objs = []
        for i in range(num_classes):
            idx_class = torch.where(ys == i)[0]
            obj_class = self._filter_obj(obj, idx_class)
            sort_class = torch.argsort(obj_class['confs'], descending=True)
            thresh_idx = min(len(obj_class['confs']), per_class)
            sort_class = sort_class[:thresh_idx]
            obj_class = self._filter_obj(obj_class, sort_class)
            if compute_acc:
                print("Filtered class unlabelled accuracy", self.get_accuracy(obj_class))

            all_objs.append(obj_class)
        concat_obj = {k: torch.cat([indiv_obj[k] for indiv_obj in all_objs]) for k in obj.keys()}
        return concat_obj

    def get_fraction(self, era):
        return min(self.start_fraction*(era+1), 1)

    def _run_indiv_labelling(self, model_dicts, era):
        unfiltered_objs = []
        filtered_objs = []
        all_idx = None
        for i, model_dict in enumerate(model_dicts):
            wrapper = make_ds_wrapper([self.unlabelled_dataset], perform_aug=True,
                                      additional_transform_name=model_dict['model_arg'].additional_transform,
                                      ds_stats=model_dict['model_arg'].ds_stats)
            loader = get_loader(wrapper, model_dict['model_arg'], shuffle=False)
            all_obj = self.label_dataset_with_model(loader, model_dict['model'])
            unfiltered_acc = self.get_accuracy(all_obj)
            print("Unfiltered class accuracy", unfiltered_acc)
            self.writer.add_scalar(f'{i}_unlabelled_acc', unfiltered_acc, era)
            unfiltered_objs.append(all_obj)
            full_length = len(all_obj['ys'])

            fraction = self.get_fraction(era)
            per_class = round(fraction * full_length / self.num_classes)
            print("Fraction", fraction)
            obj = self.take_top_perc_from_each_class(all_obj, per_class=per_class)
        
            filtered_acc = self.get_accuracy(obj)
            print("Filtered class accuracy", filtered_acc)
            self.writer.add_scalar(f'{i}_filtered_unlabelled_acc', filtered_acc, era)
            self.store['unlabeled'].append_row({'model': i,
                                                'era': era,
                                                'acc': unfiltered_acc,
                                                'selected_acc': filtered_acc,
                                                'num_points': len(obj['ys'])})
            idx = obj['idxs']
            all_idx = torch.cat([all_idx, idx]) if all_idx is not None else idx
            filtered_objs.append(obj)
        return unfiltered_objs, filtered_objs, all_idx

    def _collapse_object_dicts(self, object_dicts):
        def fill_master_dict(num_examples_, master_dict_, class_idx_, objs_):
            examples = set()
            sorted_objs_by_class = []
            for obj in objs_:
                obj_mask = obj['ys'] == class_idx_
                obj_by_class = {k:v[obj_mask] for k,v in obj.items()} # filter by class
                confidence = torch.argsort(obj_by_class['confs'], descending=True)
                sorted_obj_by_class = {k:v[confidence] for k,v in obj_by_class.items()}
                sorted_objs_by_class.append(sorted_obj_by_class)
            for i in range(num_examples_):
                for obj in sorted_objs_by_class:
                    if i >= len(obj['idxs']):
                        continue
                    idx = obj['idxs'][i].item()
                    if idx not in master_dict_:
                        master_dict_[idx] = {'im': obj['ims'][i], 'gt': obj['gts'][i], 'ys': [], 'confs': [], 'preds': []}
                    master_dict_[idx]['ys'].append(obj['ys'][i])
                    master_dict_[idx]['confs'].append(obj['confs'][i])
                    master_dict_[idx]['preds'].append(obj['preds'][i])
                    examples.add(idx)
                    if len(examples) == num_examples_: break
                if len(examples) == num_examples_: break

        master_dict = {}
        amount_needed_per_class = int(len(object_dicts[0]['gts'])/self.num_classes)
        for c in range(self.num_classes):
            fill_master_dict(amount_needed_per_class, master_dict, c, object_dicts)
        return master_dict

    def _symmetric_get_cotrained_dataset_helper(self, model_dicts, era):
        unfiltered_objs, filtered_objs, all_idx = self._run_indiv_labelling(model_dicts, era)
        # collapse
        master_dict = self._collapse_object_dicts(filtered_objs)
        all_ims, all_gts, all_ys, all_indices = [], [], [], []
        for idx in master_dict.keys():
            info = master_dict[idx]
            # take everything
            num_to_take = len(info['ys'])
            for i in range(num_to_take):
                all_ims.append(info['im'])
                all_gts.append(info['gt'])
                all_ys.append(info['ys'][i])
                all_indices.append(torch.tensor(idx))
        
        ims = torch.stack(all_ims, 0)
        ys = torch.stack(all_ys, 0)
        idxs = torch.stack(all_indices, 0)
        gts = torch.stack(all_gts, 0)
        return {'idxs': idxs, 'ys': ys, 'gts': gts, 'ims': ims}

    def get_cotrained_dataset(self, model_dicts, era, save_log=True):
        log = self._symmetric_get_cotrained_dataset_helper(model_dicts, era)
        if self.use_gt:
            ds = [self.train_dataset, folder.TensorDataset(log['ims'], log['gts'])]
        else:
            ds = [self.train_dataset, folder.TensorDataset(log['ims'], log['ys'])]

        if save_log:
            gts, ys, idx = log['gts'], log['ys'], log['idxs']
            acc = len(gts[ys == gts])/len(gts)
            print('FULL_DS', 'COUNTS', {c: len(ys[ys == c]) for c in range(self.num_classes)}, 'Accuracy', acc)
            self.writer.add_scalar(f'cotraining_acc', acc, era)
            self.store['unlabeled'].append_row({'model': len(model_dicts),
                                                'era': era,
                                                'acc': acc,
                                                'selected_acc': acc,
                                                'num_points': len(ys)})
            torch.save(idx, os.path.join(self.out_dir, f'indices_era_{era}.pt'))
            torch.save(gts, os.path.join(self.out_dir, f'gts_era_{era}.pt'))
            torch.save(ys, os.path.join(self.out_dir, f'ys_era_{era}.pt'))
        return ds
        

    def co_train_models(self, input_model_args, input_models, co_training_eras, co_training_epochs_per_era, 
                        final_model, final_model_args, val_iters=15, checkpoint_iters=30):
        '''
            model_args: an arg wrapper for each input_model
            input_models: a list of pytorch input_models
            co_training_eras: number of co training steps
            co_training_epochs_per_era: number of forward passes per training step
            final_model: an untrained final pytorch model
            final_model_args: the arg wrapper for the final model
        '''

        
        model_dicts = []
        num_models = len(input_model_args)
        ### MODEL SETUP ####
        
        for i in range(num_models):
            if input_models is not None:
                model, opt, schedule  = self.initialize_model(input_model_args[i], input_models[i], checkpoint=None)
                model_dicts.append({'model_arg': input_model_args[i], 'model': model, 'opt': opt, 'schedule': schedule})
            else:
                model_dicts.append({'model_arg': input_model_args[i]})

        ### COTRAINING ####
        if self.pure_cotrain:
            assert num_models == 2
        for era in range(co_training_eras):
            if self.pure_cotrain:
                full_cotrain_ds = self.get_cotrained_dataset(model_dicts, era)
                cotrain_datasets = [self.get_cotrained_dataset([model_dicts[1]], era, save_log=False), self.get_cotrained_dataset([model_dicts[0]], era, save_log=False)]
            else:
                cotrain_ds = self.get_cotrained_dataset(model_dicts, era)
                full_cotrain_ds = cotrain_ds
                cotrain_datasets = [cotrain_ds for _ in range(num_models)]
            for i in range(num_models):
                model_dict = model_dicts[i]
                model_arg =  model_dict['model_arg']
                cotrain_ds = cotrain_datasets[i]
                cotrain_wrap = make_ds_wrapper(cotrain_ds, perform_aug=True, additional_transform_name=model_arg.additional_transform, ds_stats=model_arg.ds_stats)
                self.writer.add_scalar(f'cotrain_ds_size', len(cotrain_wrap), era)
                cotrain_loader = get_loader(cotrain_wrap, model_arg)
                val_wrap = make_ds_wrapper([self.val_dataset], perform_aug=False, additional_transform_name=model_arg.additional_transform, ds_stats=model_arg.ds_stats)
                val_loader = get_loader(val_wrap, model_arg)

                smoothing_period = 25
                losses = []
                prev_loss = np.infty
                for epoch in range(co_training_epochs_per_era):
                    epoch_num = era*co_training_epochs_per_era + epoch
                    train_prec1, train_prec5, train_loss = self.model_loop(arg_wrapper=model_arg, loop_name=f'{i}_cotrain_train', 
                                                                           loader=cotrain_loader, model=model_dict['model'], opt=model_dict['opt'],
                                                                           epoch=epoch_num, backward=True)
                    losses.append(train_loss)
                    if len(losses) >= smoothing_period:
                        avg = np.median(np.array(losses))
                        print(avg, prev_loss, train_prec1.item())
                        if avg > prev_loss and np.abs(100.00 - train_prec1.item()) < 0.5:
                            break
                        prev_loss = avg
                        losses = []
                    model_dict['schedule'].step()
                opt, schedule = make_optimizer_and_schedule(model_arg, model_dict['model'], None, None)
                model_dict['schedule'] = schedule # reset scheduler
                model_dict['opt'] = opt # reset optimizer

                with torch.no_grad():
                    self.model_loop(arg_wrapper=model_arg, loop_name=f'{i}_cotrain_val', 
                                        loader=val_loader, model=model_dict['model'], opt=model_dict['opt'],
                                        epoch=epoch_num, backward=False)
                self._save_checkpoint(model_dict['model_arg'], f'{i}_{era}.ckpt', model_dict['model'], model_dict['opt'], model_dict['schedule'], epoch=0)

        cotrain_ds = self.get_cotrained_dataset(model_dicts,co_training_eras)
        cotrain_wrap = make_ds_wrapper(cotrain_ds, perform_aug=True, additional_transform_name=final_model_args.additional_transform, ds_stats=final_model_args.ds_stats)
        cotrain_loader = get_loader(cotrain_wrap, final_model_args)
        val_wrap = make_ds_wrapper([self.val_dataset], perform_aug=False, additional_transform_name=final_model_args.additional_transform, ds_stats=final_model_args.ds_stats)
        val_loader = get_loader(val_wrap, final_model_args)
        return self.train_model(arg_wrapper=final_model_args, train_loader=cotrain_loader,
                                val_loader=val_loader, model=final_model, val_iters=val_iters, checkpoint_iters=checkpoint_iters, save_best=True, checkpoint=None)

class SingleStep(CoTrainDatasetTrainer):
    def run(self, input_model_args, input_models, epochs, final_model, final_model_args, val_iters=15, checkpoint_iters=30):
        return self.co_train_models(input_model_args=input_model_args, input_models=input_models, 
                             co_training_eras=0, co_training_epochs_per_era=0,
                             final_model=final_model, final_model_args=final_model_args, val_iters=val_iters, checkpoint_iters=checkpoint_iters)

class GTSingleStep(CoTrainDatasetTrainer):
    def __init__(self, indices, labels=None, **kwargs):
        self.indices = indices
        self.labels = labels
        print(kwargs)
        super().__init__(**kwargs)

    def get_gt_dataset(self, loader, fraction, indices=None, labels=None):
        ims, ys, confs = [], [], []
        it = tqdm(loader)
        for im, y, meta in it:
            y = y.cpu()    
            ims.append(meta['original_img'])
            ys.append(y)
        ims = torch.cat(ims, 0)
        ys = torch.cat(ys, 0)
        assert (ys >= 0).all().item()
        num_classes = ys.max().item() + 1

        if indices is None:
            indices = []
            for i in range(num_classes):
                per_class = round(fraction * len(ys) / num_classes)
                idx_class = torch.where(ys == i)[0]
                per_class = min(len(idx_class), per_class)
                indices.append(idx_class[:per_class])
            indices = torch.cat(indices)
                
        ims = ims[indices]
        ys = ys[indices]
        if labels is not None:
            acc = ys == labels
            acc = len(ys[acc])/len(ys)
            print("ACCURACY", acc)
            ys = labels

        for i in range(num_classes):
            print(i, len(ys[ys == i]))
        dataset = folder.TensorDataset(ims, ys)
        return dataset


    def get_cotrained_dataset(self, model_dicts, era):
        # model_dicts should contain one arg
        datasets = [self.train_dataset]
        assert len(model_dicts) == 1
        dummy_model_dict = model_dicts[0]

        fraction = min(self.start_fraction*(era+1), 1)
        wrapper = make_ds_wrapper([self.unlabelled_dataset], perform_aug=True, additional_transform_name='NONE', ds_stats=dummy_model_dict['model_arg'].ds_stats)
        loader = get_loader(wrapper, dummy_model_dict['model_arg'], shuffle=False)
        datasets.append(self.get_gt_dataset(loader, fraction, self.indices, self.labels))
        return datasets

    def run(self, epochs, model, model_args, out_dir, val_iters=15, checkpoint_iters=30):
        return self.co_train_models(input_model_args=[model_args], input_models=None, 
                             co_training_eras=0, co_training_epochs_per_era=0,
                             final_model=model,
                             final_model_args=model_args, val_iters=val_iters,
                             checkpoint_iters=checkpoint_iters)
