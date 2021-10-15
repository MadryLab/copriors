from utils.common import to_numpy, rec_fn_apply
import torch
import cv2
import torchvision
import os
import csv
from PIL import Image
from torchvision import transforms


MAX_LOG_IMAGES = 4
import numpy as np
def log_images_hook(model, iteration, loop_type, inp, target, output, meta, epoch, writer, classes=None, meter_dict=None):
    if iteration != 0:
        return
    perm = torch.randperm(inp.size(0))[:MAX_LOG_IMAGES]
    orig_img = meta['original_img']
    for idx in perm:
        fig = compose_logged_image(x=inp[idx], y=target[idx], yh=output[idx], orig_img=orig_img[idx], classes=classes)
        writer.add_image(f'images/{loop_type}', fig, epoch)

def compose_logged_image(x, y, yh, orig_img, classes):
    gt = y.detach().cpu().item()
    pred = yh.argmax().detach().cpu().item()
    if classes is not None:
        gt = classes[gt]
        pred = classes[pred]
    pred_box = compose_prediction_box(x.shape[1:], gt, pred)
    fig = torchvision.utils.make_grid([orig_img.detach().cpu(), x.detach().cpu(), pred_box])
    return fig

def compose_prediction_box(img_shape, gt, pred):
    h,w = img_shape
    dy = int(20 * h/96.0)
    fontscale = 0.5*h/96.0
    text_image = np.ones((h, w, 3))*1
    for i, word in enumerate(['gt', str(gt), 'pred', str(pred)]):
        cv2.putText(text_image, word, org=(0,(i+1)*dy), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontscale, color=(0, 0, 0))
    text_image = text_image.transpose(2, 0, 1)
    return torch.FloatTensor(text_image)
