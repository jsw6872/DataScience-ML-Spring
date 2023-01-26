from math import sqrt
from itertools import product

import pandas as pd
import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def make_vgg():
    layers = []
    in_channels = 3  

    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256,
           256, 'MC', 512, 512, 512, 'M', 512, 512, 512]

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'MC':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU()]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(), conv7, nn.ReLU()]
    return nn.ModuleList(layers)


def make_extras():
    layers = []
    in_channels = 1024  

    cfg = [256, 512, 128, 256, 128, 256, 128, 256]

    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=1)]
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=1)]
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=1)]
    layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size=3)]
    layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size=1)]
    layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size=3)]
    return nn.ModuleList(layers)


def make_loc_conf(num_classes=21, bbox_aspect_num=[4, 6, 6, 6, 4, 4]):

    loc_layers = []
    conf_layers = []

    # source1의 합성곱 층
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[0]*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[0]* num_classes, kernel_size=3, padding=1)]

    # source2의 합성곱 층
    loc_layers += [nn.Conv2d(1024, bbox_aspect_num[1]*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, bbox_aspect_num[1]*num_classes, kernel_size=3, padding=1)]

    # source3의 합성곱 층
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[2]*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[2]*num_classes, kernel_size=3, padding=1)]

    # source4의 합성곱 층
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[3]*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[3]*num_classes, kernel_size=3, padding=1)]

    # source5의 합성곱 층
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[4]*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[4]*num_classes, kernel_size=3, padding=1)]

    # source6의 합성곱 층
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[5]*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[5]*num_classes, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)


#
class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()  
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale  
        self.reset_parameters()  
        self.eps = 1e-10

    def reset_parameters(self):
        init.constant_(self.weight, self.scale) # weight값이 전부 self.sclae의 값이 2개가 됨

    # 각 채널의 38*38 특징량의 채널방향 제곱합을 계산
    def forward(self, x):
        # pow : n승
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps # norm tensor size = torch.size([batch_num, 1, 38, 38])
        x = torch.div(x, norm) # x를 norm으로 나눔

        # 채널마다 1개의 계수를 가짐
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) # self.weight = torch.size([512])
        out = weights * x # weight = torch.size([batch_num, 512, 38, 38])
        return out


class DBox(object):
    def __init__(self, cfg):
        super(DBox, self).__init__()

        self.image_size = cfg['input_size']  # 화상크기 300*300
        self.feature_maps = cfg['feature_maps']
        self.num_priors = len(cfg["feature_maps"])  # source의 개수 : 6
        self.steps = cfg['steps']  # [8, 16, …] DBox의 픽셀 크기
        
        self.min_sizes = cfg['min_sizes'] #  [30, 60, …] 작은 정사각형의 DBox 픽셀
        
        self.max_sizes = cfg['max_sizes'] # [60, 111, …] 큰 정사각형의 DBox 픽셀
        
        self.aspect_ratios = cfg['aspect_ratios']  # 정사각형의 DBox 화면비(종횡비)

    def make_dbox_list(self):
        mean = []
        # 'feature_maps': [38, 19, 10, 5, 3, 1] source1~6
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):  # f까지의 수로 조합을 만들어냄
                # 특징량의 화상크기
                # 300 / 'steps': [8, 16, 32, 64, 100, 300],
                f_k = self.image_size / self.steps[k]

                # DBox의 중심좌표 x, y 0~1로 정규화되어있음
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # 화면비 1의 작은 DBox [cx,cy, width, height]
                # 'min_sizes': [30, 60, 111, 162, 213, 264]
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # 화면비 1의 큰 DBox [cx,cy, width, height]
                # 'max_sizes': [60, 111, 162, 213, 264, 315],
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 그 외 화면비의 defBox [cx,cy, width, height]
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        # DBox size : torch.Size([8732, 4])
        output = torch.Tensor(mean).view(-1, 4)

        # DBox가 이미지 밖으로 돌출되는 것을 막기 위해 크기를 0~1로 조정
        output.clamp_(max=1, min=0)

        return output


class SSD(nn.Module):

    def __init__(self, phase, cfg):
        super(SSD, self).__init__()

        self.phase = phase  # train or inference
        self.num_classes = cfg["num_classes"]  # 클래스 수 : 21

        # SSD 네트워크
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(cfg["num_classes"], cfg["bbox_aspect_num"])

        # DBox 작성
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

        # if phase == 'inference':
        #     self.detect = Detect()


def decode(loc, dbox_list):
    # 오프셋 정보(loc)로 DBox를 BBox로 변환
    # loc : [8732, 4], dbox_list : [8732, 4]
    
    # loc [Δcx, Δcy, Δwidth, Δheight]
    # DBox [cx, cy, width, height]

    # 오프셋정보로 BBox 구함
    boxes = torch.cat((
        dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, 2:],
        dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), dim=1)
    # boxes torch.Size([8732, 4])

    # BBox의 좌표 정보를 [cx, cy, width, height]에서 [xmin, ymin, xmax, ymax]로 변경
    boxes[:, :2] -= boxes[:, 2:] / 2  # 좌표 (xmin,ymin)로 변환
    boxes[:, 2:] += boxes[:, :2]  # 좌표 (xmax,ymax)로 변환

    return boxes


def nm_suppression(boxes, scores, overlap=0.45, top_k=200):
    """
    boxes중 overlap 이상의 BBox 삭제

    Parameters
    ----------
    boxes : [신뢰도 임계값 0.01을 넘은 BBox 수,4]
        BBox 정보
    scores :[신뢰도 임계값 0.01을 넘은 BBox 수]
        conf 정보

    Returns
    -------
    keep : 리스트
        conf의 내림차순으로 nms를 통과한 index 저장
    count：int
        nms를 통과한 BBox 수
    """

    # return
    count = 0
    keep = scores.new(scores.size(0)).zero_().long() # keep：torch.Size([신뢰도 임계값을 넘은 BBox 수]), item은 전부 0

    # 각 Bbox의 면적 area 계산
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    # boxes 복사. 나중에 BBox 중복도(IOU) 계산 시 모형으로 준비
    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    # socre 오름차순으로 나열
    v, idx = scores.sort(0)

    # 상위 k개 Bbox index 꺼냄(k개보다 적게 존재할 수도 있음)
    idx = idx[-top_k:]

    # idx의 item수가 0이 아닌 한 loop
    while idx.numel() > 0:
        i = idx[-1]  # 현재 conf 최대 index를 i로

        # keep 끝에 conf 최대 index 저장
        # 이 index의 BBox와 크게 겹치는 BBox삭제
        keep[count] = i
        count += 1

        # 마지막 BBox는 루프를 빠져나온다
        if idx.size(0) == 1:
            break

        # 현재 conf 최대의 index를 keep에 저장했으므로 idx를 하나 감소시킴
        idx = idx[:-1]

        # -------------------
        # 지금부터 keep에 저장한 BBox와 크게 겹치는 BBox 추출하여 삭제
        # -------------------
        # 하나 감소시킨 idx까지의 BBox를 out으로 지정한 변수로 작성
        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)

        # 
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

        # w, h의 텐서 크기를 index 하나 줄인 것으로 함
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        # clamp한 상태에서 BBox의 폭과 높이를 구함
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        # 폭이나 높이가 음수인 것은 0으로 맞춤
        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        # clamp된 상태의 면적을 구함
        inter = tmp_w*tmp_h

        # IoU = intersect / (area(a) + area(b) - intersect)
        rem_areas = torch.index_select(area, 0, idx)  # 각 BBox의 원래 면적
        union = (rem_areas - inter) + area[i]  # 2영역의 합 면적
        IoU = inter/union

        # IoU가 overlap보다 작은 idx만 남김
        idx = idx[IoU.le(overlap)]  # le:Less than or Equal
        # IoU가 overlap보더 큰 idx는 처음 선택한 keep에 저장한 idx와 동일한 물체에 BBox를 둘러싸고 있어 삭제
    return keep, count

# torch.autograd.Function 상속, SSD추론 시 conf와 loc의 출력에서 중복을 제거한 BBox출력
class Detect(Function):
    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1)  # conf를 softmax로 정규화하기 위해
        self.conf_thresh = conf_thresh  
        self.top_k = top_k  # conf(confidence)가 높은 top_k개를 nms_supression으로 게산에 사용하는 top_k
        self.nms_thresh = nms_thresh  # nm_supression으로 IOU가 nms_thresh보다 크면 동일한 물체의 BBox로 간주

    def forward(self, loc_data, conf_data, dbox_list):
        """
        Parameters
        ----------
        loc_data:  [batch_num,8732,4]
            오프셋정보
        conf_data: [batch_num, 8732,num_classes]
            confidence
        dbox_list: [8732,4]
            DBox 정보

        Returns
        -------
        output : torch.Size([batch_num, 21, 200, 5])
            （batch_num、class、conf의top200、BBox 정보）
        """

        # 각 크기 취득
        num_batch = loc_data.size(0)  # batch 크기
        num_dbox = loc_data.size(1)  # DBox 수 = 8732
        num_classes = conf_data.size(2)  # class수 = 21

        # conf 소프트맥스 정규화
        conf_data = self.softmax(conf_data)

        # [batch, 21, 200, 5]
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)

        # cof_data를[batch_num,8732,num_classes]에서[batch_num, num_classes,8732]으로 변경
        conf_preds = conf_data.transpose(2, 1)

        for i in range(num_batch):

            # 1. loc와DBox로 수정한 BBox [xmin, ymin, xmax, ymax] 구함
            decoded_boxes = decode(loc_data[i], dbox_list)

            # conf의 복사본 작성
            conf_scores = conf_preds[i].clone()

            # 이미지 class별로 계산(배경인 0은 계산 X)
            for cl in range(1, num_classes):

                # 2.conf의 임계값을 넘은 BBox를 꺼냄
                # conf의 임계값을 넘고 있늕지 마스크를 작성하여 임계값을 넘은 conf의 인덱스를 c_mask로 얻음
                c_mask = conf_scores[cl].gt(self.conf_thresh) # Greater than 의미 gt로 임계값이 넘으면 1, 이하는 0
                # conf_scores:torch.Size([21, 8732])
                # c_mask:torch.Size([8732])

                # scores torch.Size([임계값을 넘은 BBox 수])
                scores = conf_scores[cl][c_mask]

                if scores.nelement() == 0:  # nelement: scores의 합계를 구함
                    continue

                # decoded_boxes에 적용가능하도록 크기 변경
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # l_mask:torch.Size([8732, 4])

                # l_mask를 decoded_boxes로 적용
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # decoded_boxes[l_mask]로 1차원이 되기 때문에 view에서 (임계값 넘은 BBox 수, 4)로 크기 바꿈

                # 3. Non-Maximum Suppression 실행하여 중복되는 BBox 제거
                ids, count = nm_suppression(boxes, scores, self.nms_thresh, self.top_k)
                # ids：conf의 내림차로 Non-Maximum Suppression을 통과한 index 저장
                # count：Non-Maximum Suppression 통과한 BBox 수

                # output에 Non-Maximum Suppression을 뺀 결과
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)

        return output  # torch.Size([1, 21, 200, 5])



class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()

        self.phase = phase  # train or inference 지정
        self.num_classes = cfg["num_classes"]  # class 수 21

        # SSDのネットワークを作る
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(cfg["num_classes"], cfg["bbox_aspect_num"])

        # DBox
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

        # 추론 시 detect
        if phase == 'inference':
            self.detect = Detect()

    def forward(self, x):
        sources = list()  # source1～6 저장
        loc = list()  
        conf = list() 

        # vgg의 conv4_3까지 계산
        for k in range(23):
            x = self.vgg[k](x)

        # conv4_3 출력을 L2norm에 입력하고 source1을 작성하여 sources에 추가
        source1 = self.L2Norm(x)
        sources.append(source1)

        # vgg를 마지막까지 계산하여 source2를 작성하고 sources에 추가
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        sources.append(x)

        # extras의 conv와 ReLU 계산 source3～6을 sources에 추가
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:  # conv→ReLU→cov→ReLU하여 sources에 추가
                sources.append(x)

        # source1～6에 해당하는 conv 1회씩 적용
        for (x, l, c) in zip(sources, self.loc, self.conf): # sources엔 1~6개의 source있음
            loc.append(l(x).permute(0, 2, 3, 1).contiguous()) # contiguous : 메모리 상에 연속적으로 요소를 배치하는 명령
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            # l(x), c(x)의 출력 크기[batch_num, 4*화면비 종류 수, featuremap 높이, featuremap 폭]
            # source에 따라 화면비 종류가 달라 순서 바꾸어 조정
            # [minibatch, featuremap수, featuremap수,4*화면비의 종류 수]
            # view를 수행하므로 대상의 변수가 메모리 상에 연속적으로 배치되어야함

        # loc torch.Size([batch_num, 34928])
        # conf torch.Size([batch_num, 183372])になる
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # さらにlocとconfの形を整える
        # loc torch.Size([batch_num, 8732, 4])
        # conf torch.Size([batch_num, 8732, 21])
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        output = (loc, conf, self.dbox_list)

        if self.phase == "inference":
            # output torch.Size([batch_num, 21, 200, 5])
            return self.detect(output[0], output[1], output[2])

        else:
            return output
            # output은 (loc, conf, dbox_list)



if __name__ == '__main__':
    vgg_test = make_vgg()
    print(vgg_test)

    extras_test = make_extras()
    print(extras_test)

    loc_test, conf_test = make_loc_conf()
    print(loc_test)
    print(conf_test)

    # SSD300 config
    ssd_cfg = {
        'num_classes': 21,  
        'input_size': 300,
        'bbox_aspect_num': [4, 6, 6, 6, 4, 4], 
        'feature_maps': [38, 19, 10, 5, 3, 1], 
        'steps': [8, 16, 32, 64, 100, 300],  
        'min_sizes': [30, 60, 111, 162, 213, 264],
        'max_sizes': [60, 111, 162, 213, 264, 315],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    }

    # DBox 작성
    dbox = DBox(ssd_cfg)
    dbox_list = dbox.make_dbox_list()

    # DBox 출력확인
    print(pd.DataFrame(dbox_list.numpy()))