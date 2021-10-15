from robustness.tools.helpers import AverageMeter
from robustness.tools import helpers

class AccuracyMeter:
    def __init__(self, maxk=5):
        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.agree_meter = AverageMeter()
        self.disagree_meter = AverageMeter()
        self.maxk = maxk
        self.report_agree_meter = False

    def update(self, batch_size, model_logits, target, loss_mean, meta):
        maxk = min(model_logits.shape[-1], self.maxk)
        prec1, prec5 = helpers.accuracy(model_logits, target, topk=(1, maxk))
        prec1, prec5 = prec1[0], prec5[0]
        self.losses.update(loss_mean.item(), batch_size)
        self.top1.update(prec1, batch_size)
        self.top5.update(prec5, batch_size)
        if 'agrees' in meta:
            self.report_agree_meter = True
            agree = helpers.accuracy(model_logits[meta['agrees'] == 1], target[meta['agrees'] == 1])
            disagree = helpers.accuracy(model_logits[meta['agrees'] == 0], target[meta['agrees'] == 0])
            self.agree_meter.update(agree[0].item(), len(model_logits[meta['agrees'] == 1]))
            self.disagree_meter.update(disagree[0].item(), len(model_logits[meta['agrees'] == 0]))

    def stats(self):
        d = {
            'loss': self.losses.avg,
            'prec1': self.top1.avg,
            'prec5': self.top5.avg,
        }
        if self.report_agree_meter:
            d['agree'] = self.agree_meter.avg
            d['disagree'] = self.disagree_meter.avg
        return d

    def get_desc(self):
        stats = self.stats()
        desc = ('Loss {loss:.4f} | Top1 {top1_acc:.3f} | Top5 {top5_acc:.3f}').format(
            loss=stats['loss'], top1_acc=stats['prec1'], top5_acc=stats['prec5'])
        if self.report_agree_meter:
            desc += (' | Agree {agree:.3f} | Disagree {disagree:.3f}').format(agree=stats['agree'], disagree=stats['disagree'])
        return desc