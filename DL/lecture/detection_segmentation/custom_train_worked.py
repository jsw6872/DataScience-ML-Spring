import time
import logging
import os, sys, math
import argparse
from collections import deque
import datetime
import copy

import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
# from tensorboardX import SummaryWriter
# from easydict import EasyDict as edict

from dataset import *
# from cfg import Cfg
from model import Yolov4
# from tool.darknet2pytorch import Darknet

from tool.tv_reference.utils import collate_fn as val_collate
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from tool.tv_reference.coco_utils import convert_to_coco_api
from tool.tv_reference.coco_eval import CocoEvaluator


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, GIoU=False, DIoU=False, CIoU=False):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    https://github.com/ultralytics/yolov3/blob/eca5b9c1d36e4f73bf2f94e141d864f1c2739e23/utils/utils.py#L262-L282
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        # intersection top left
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # intersection bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
        con_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, 0] + bboxes_a[:, None, 2]) - (bboxes_b[:, 0] + bboxes_b[:, 2])) ** 2 / 4 + (
                (bboxes_a[:, None, 1] + bboxes_a[:, None, 3]) - (bboxes_b[:, 1] + bboxes_b[:, 3])) ** 2 / 4

        w1 = bboxes_a[:, 2] - bboxes_a[:, 0]
        h1 = bboxes_a[:, 3] - bboxes_a[:, 1]
        w2 = bboxes_b[:, 2] - bboxes_b[:, 0]
        h2 = bboxes_b[:, 3] - bboxes_b[:, 1]

        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # intersection top left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # intersection bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        con_br = torch.max((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, :2] - bboxes_b[:, :2]) ** 2 / 4).sum(dim=-1)

        w1 = bboxes_a[:, 2]
        h1 = bboxes_a[:, 3]
        w2 = bboxes_b[:, 2]
        h2 = bboxes_b[:, 3]

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u

    if GIoU or DIoU or CIoU:
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            area_c = torch.prod(con_br - con_tl, 2)  # convex area
            return iou - (area_c - area_u) / area_c  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou


class Yolo_loss(nn.Module):
    def __init__(self, n_classes=80, n_anchors=3, device=None, batch=2):
        super(Yolo_loss, self).__init__()
        self.device = device
        self.strides = [8, 16, 32]
        image_size = 608
        self.n_classes = n_classes
        self.n_anchors = n_anchors

        self.anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5

        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        for i in range(3):
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32)
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)
            # calculate pred - xywh obj cls
            fsize = image_size // self.strides[i]
            grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).to(device)
            grid_y = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).permute(0, 1, 3, 2).to(device)
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)
            anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id):
        # target assignment
        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes).to(device=self.device)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(device=self.device)
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(self.device)
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(self.device)

        # labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (self.strides[output_id] * 2)
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (self.strides[output_id] * 2)
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]
        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4).to(self.device)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors[output_id], CIoU=True)

            # temp = bbox_iou(truth_box.cpu(), self.ref_anchors[output_id])

            best_n_all = anchor_ious_all.argmax(dim=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anch_masks[output_id][0]) |
                           (best_n_all == self.anch_masks[output_id][1]) |
                           (best_n_all == self.anch_masks[output_id][2]))

            if sum(best_n_mask) == 0:
                continue

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou(pred[b].reshape(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~ pred_best_iou

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin, labels=None):
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        for output_id, output in enumerate(xin):
            batchsize = output.shape[0]
            fsize = output.shape[2]
            n_ch = 5 + self.n_classes

            output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

            # logistic activation for xy, obj, cls
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

            pred = output[..., :4].clone()
            pred[..., 0] += self.grid_x[output_id]
            pred[..., 1] += self.grid_y[output_id]
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]

            obj_mask, tgt_mask, tgt_scale, target = self.build_target(pred, labels, batchsize, fsize, n_ch, output_id)

            # loss calculation
            output[..., 4] *= obj_mask
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            output[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            loss_xy += F.binary_cross_entropy(input=output[..., :2], target=target[..., :2],
                                              weight=tgt_scale * tgt_scale, reduction='sum')
            loss_wh += F.mse_loss(input=output[..., 2:4], target=target[..., 2:4], reduction='sum') / 2
            loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], reduction='sum')
            loss_cls += F.binary_cross_entropy(input=output[..., 5:], target=target[..., 5:], reduction='sum')
            loss_l2 += F.mse_loss(input=output, target=target, reduction='sum')

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


def collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append([img])
        bboxes.append([box])
    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images).div(255.0)
    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)
    return images, bboxes


def od_collate_fn(batch):
    imgs = []
    bboxes = []
    for img, box in batch:
        img = cv2.resize(img, (512, 512))
        imgs.append([img])
        # imgs.append(torch.from_numpy(img))  # 이미지
        bboxes.append(torch.cat((box[0].squeeze(),box[1]), dim = 0))  # sample[1] 어노테이션 정보
    imgs = np.concatenate(imgs, axis=0)
    imgs = imgs.transpose(0, 3, 1, 2)
    imgs = torch.from_numpy(imgs).div(255.0) # (batch_size, channel, width, height)
    # imgs = torch.stack(imgs)

    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)
    return imgs, bboxes


def train(model, learning_rate, batch_size, epochs=30, save_cp=True):
    train_dataset = Yolo_dataset(data_path, train=True)
    val_dataset = Yolo_dataset(data_path, train=False)

    # 원래 : drop_last=True, config.batch // config.subdivisions
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,collate_fn=collate)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # global_step = 0

    # learning rate setup
    # def burnin_schedule(i):
    #     if i < config.burn_in:
    #         factor = pow(i / config.burn_in, 4)
    #     elif i < config.steps[0]:
    #         factor = 1.0
    #     elif i < config.steps[1]:
    #         factor = 0.1
    #     else:
    #         factor = 0.01
    #     return factor

    optimizer = optim.Adam(model.parameters(),lr=learning_rate,betas=(0.9, 0.999),eps=1e-08)
    # optimizer = optim.SGD(params=model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=0.001)

    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    criterion = Yolo_loss(device=device, batch=batch_size, n_classes=80)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience=6, min_lr=1e-7)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, 0.001, 1e-6, 20)
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Optimizer:       {optimizer}
    ''') #Subdivisions:    {config.subdivisions}

    save_prefix = 'Yolov4_epoch'
    saved_models = deque()
    model.train()
    for epoch in range(epochs):
        since = time.time() 
        # model.train()
        train_loss = 0
        best_model_wts = copy.deepcopy(model.state_dict()) 
        # epoch_step = 0
        # with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', ncols=50) as pbar:
        for i, batch in enumerate(train_loader):
            # global_step += 1
            # epoch_step += 1
            images = batch[0]
            bboxes = batch[1]

            images = images.to(device=device, dtype=torch.float32)
            bboxes = bboxes.to(device=device)

            # optimizer.zero_grad() # 내가 추가한 코드

            bboxes_pred = model(images)
            loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(bboxes_pred, bboxes)
            # loss = loss / config.subdivisions
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            # train_loss = evaluate(model, train_loader, criterion)
            # val_loss = evaluate(model, val_loader, criterion)
            
            # if best_val_loss > val_loss:
            #     best_val_loss = val_loss
            #     best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since 
        print('-------------- epoch {} ----------------'.format(epoch+1))
        print('train Loss: {:.4f}'.format(train_loss))   
        # print('val Loss: {:.4f}'.format(val_loss))
        print('Completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)
    return model


def evaluate(model, data_loader, criterion):
    model.eval()
    eval_loss = 0
    # with torch.no_grad():
    for i, batch in enumerate(data_loader):
        images = batch[0]
        bboxes = batch[1]

        images = images.to(device, dtype = torch.float32)
        bboxes = bboxes.to(device)

        bboxes_pred = model(images)
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(bboxes_pred, bboxes)
        eval_loss += loss.item()
    return eval_loss / len(data_loader)
@torch.no_grad()
def evaluate(model, data_loader, cfg, device, logger=None, **kwargs):
    """ finished, tested
    """
    # cpu_device = torch.device("cpu")
    model.eval()
    # header = 'Test:'

    coco = convert_to_coco_api(data_loader.dataset, bbox_fmt='coco')
    coco_evaluator = CocoEvaluator(coco, iou_types = ["bbox"], bbox_fmt='coco')

    for images, targets in data_loader:
        model_input = [[cv2.resize(img, (cfg.w, cfg.h))] for img in images]
        model_input = np.concatenate(model_input, axis=0)
        model_input = model_input.transpose(0, 3, 1, 2)
        model_input = torch.from_numpy(model_input).div(255.0)
        model_input = model_input.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(model_input)

        # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        # outputs = outputs.cpu().detach().numpy()
        res = {}
        # for img, target, output in zip(images, targets, outputs):
        for img, target, boxes, confs in zip(images, targets, outputs[0], outputs[1]):
            img_height, img_width = img.shape[:2]
            # boxes = output[...,:4].copy()  # output boxes in yolo format
            boxes = boxes.squeeze(2).cpu().detach().numpy()
            boxes[...,2:] = boxes[...,2:] - boxes[...,:2] # Transform [x1, y1, x2, y2] to [x1, y1, w, h]
            boxes[...,0] = boxes[...,0]*img_width
            boxes[...,1] = boxes[...,1]*img_height
            boxes[...,2] = boxes[...,2]*img_width
            boxes[...,3] = boxes[...,3]*img_height
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # confs = output[...,4:].copy()
            confs = confs.cpu().detach().numpy()
            labels = np.argmax(confs, axis=1).flatten()
            labels = torch.as_tensor(labels, dtype=torch.int64)
            scores = np.max(confs, axis=1).flatten()
            scores = torch.as_tensor(scores, dtype=torch.float32)
            res[target["image_id"].item()] = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator


def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


def _get_date_str():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M')


if __name__ == "__main__":
    data_path = '/home/seongwoo/workspace/DataScience_ML-DL/DL/lecture/detection_segmentation/data'
    # train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

    # voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
    #            'bottle', 'bus', 'car', 'cat', 'chair',
    #            'cow', 'diningtable', 'dog', 'horse',
    #            'motorbike', 'person', 'pottedplant',
    #            'sheep', 'sofa', 'train', 'tvmonitor']

    # color_mean = (104, 117, 123)  # (BGR) 평균값
    # input_size = 300  # 이미지 input size

    train_dataset = Yolo_dataset(data_path, train=True)
    val_dataset = Yolo_dataset(data_path, train=False)

    batch_size = 4
    learning_rate = 0.05
    epochs = 5
    # 원래 : drop_last=True, config.batch // config.subdivisions
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6,collate_fn=collate)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6, collate_fn=collate)






    logging = init_logger(log_dir='log')
    # cfg = get_args(**Cfg)
    # os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda')
    logging.info(f'Using device {device}')

    # if cfg.use_darknet_cfg:
    #     model = Darknet(cfg.cfgfile)
    model = Yolov4(n_classes=80, inference=False)
    # model = torch.hub.load("VCasecnikovs/Yet-Another-YOLOv4-Pytorch", "yolov4", pretrained=True)
    model.to(device)

    try:
        saved_model = train(model=model, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size)
        torch.save(saved_model,'./model/test.pt')

    except KeyboardInterrupt:
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), 'INTERRUPTED.pth')
        else:
            torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

# @torch.no_grad()
# def coco_evaluate(model, data_loader, logger=None, **kwargs):
#     """ finished, tested
#     """
#     # cpu_device = torch.device("cpu")
#     model.eval()
#     # header = 'Test:'

#     coco = convert_to_coco_api(data_loader.dataset, bbox_fmt='coco')
#     coco_evaluator = CocoEvaluator(coco, iou_types = ["bbox"], bbox_fmt='coco')

#     for images, targets in data_loader:
#         model_input = [[cv2.resize(img, (cfg.w, cfg.h))] for img in images]
#         model_input = np.concatenate(model_input, axis=0)
#         model_input = model_input.transpose(0, 3, 1, 2)
#         model_input = torch.from_numpy(model_input).div(255.0)
#         model_input = model_input.to(device)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#         model_time = time.time()
#         outputs = model(model_input)

#         # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
#         model_time = time.time() - model_time

#         # outputs = outputs.cpu().detach().numpy()
#         res = {}
#         # for img, target, output in zip(images, targets, outputs):
#         for img, target, boxes, confs in zip(images, targets, outputs[0], outputs[1]):
#             img_height, img_width = img.shape[:2]
#             # boxes = output[...,:4].copy()  # output boxes in yolo format
#             boxes = boxes.squeeze(2).cpu().detach().numpy()
#             boxes[...,2:] = boxes[...,2:] - boxes[...,:2] # Transform [x1, y1, x2, y2] to [x1, y1, w, h]
#             boxes[...,0] = boxes[...,0]*img_width
#             boxes[...,1] = boxes[...,1]*img_height
#             boxes[...,2] = boxes[...,2]*img_width
#             boxes[...,3] = boxes[...,3]*img_height
#             boxes = torch.as_tensor(boxes, dtype=torch.float32)
#             # confs = output[...,4:].copy()
#             confs = confs.cpu().detach().numpy()
#             labels = np.argmax(confs, axis=1).flatten()
#             labels = torch.as_tensor(labels, dtype=torch.int64)
#             scores = np.max(confs, axis=1).flatten()
#             scores = torch.as_tensor(scores, dtype=torch.float32)
#             res[target["image_id"].item()] = {
#                 "boxes": boxes,
#                 "scores": scores,
#                 "labels": labels,
#             }
#         evaluator_time = time.time()
#         coco_evaluator.update(res)
#         evaluator_time = time.time() - evaluator_time

#     # gather the stats from all processes
#     coco_evaluator.synchronize_between_processes()

#     # accumulate predictions from all images
#     coco_evaluator.accumulate()
#     coco_evaluator.summarize()

#     return coco_evaluator


# def convert_output_to_boxes(outputs):
#     # Extract the bounding box coordinates, labels, and confidences
#     boxes = []
#     labels = []
#     confidences = []

#     # Iterate through the output and extract the bounding boxes and labels
#     for i, output in enumerate(outputs):
#         # Extract the bounding box coordinates and confidences
#         box = output[:, :4]
#         confidence = output[:, 4]

#         # Extract the labels
#         label = output[:, 5:]

#         # Append the bounding boxes and labels to the list
#         boxes.append(box)
#         confidences.append(confidence)
#         labels.append(label)

#     return boxes, labels, confidences



# def visualize_detections(model, image, device):
#     model.eval()
#     with torch.no_grad():
#         image = image.to(device)
#         outputs = model(image)
        
#         # Convert the output to bounding boxes
#         boxes, labels, confidences = convert_output_to_boxes(outputs)

#         # Draw the bounding boxes on the image
#         for box, label, confidence in zip(boxes, labels, confidences):
#             xmin, ymin, xmax, ymax = box
#             cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#             cv2.putText(image, f"{label}: {confidence:.2f}", (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Convert the image to BGR and display it
#     image = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2BGR)
#     cv2.imshow("Image with detections", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# Example usage
# image = cv2.imread("path/to/image.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = torch.from_numpy(image)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# visualize_detections(model, image, device)






# def get_args(**kwargs):
#     cfg = kwargs
#     parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     # parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
#     #                     help='Batch size', dest='batchsize')
#     parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
#                         help='Learning rate', dest='learning_rate')
#     parser.add_argument('-f', '--load', dest='load', type=str, default=None,
#                         help='Load model from a .pth file')
#     parser.add_argument('-g', '--gpu', metavar='G', type=str, default='-1',
#                         help='GPU', dest='gpu')
#     parser.add_argument('-dir', '--data-dir', type=str, default=None,
#                         help='dataset dir', dest='dataset_dir')
#     parser.add_argument('-pretrained', type=str, default=None, help='pretrained yolov4.conv.137')
#     parser.add_argument('-classes', type=int, default=80, help='dataset classes')
#     parser.add_argument('-train_label_path', dest='train_label', type=str, default='train.txt', help="train label path")
#     parser.add_argument(
#         '-optimizer', type=str, default='adam',
#         help='training optimizer',
#         dest='TRAIN_OPTIMIZER')
#     parser.add_argument(
#         '-iou-type', type=str, default='iou',
#         help='iou type (iou, giou, diou, ciou)',
#         dest='iou_type')
#     parser.add_argument(
#         '-keep-checkpoint-max', type=int, default=10,
#         help='maximum number of checkpoints to keep. If set 0, all checkpoints will be kept',
#         dest='keep_checkpoint_max')
#     args = vars(parser.parse_args())

#     # for k in args.keys():
#     #     cfg[k] = args.get(k)
#     cfg.update(args)

#     return edict(cfg)

