import torch
import torch.nn as nn
import torch.nn.functional as F

from match import match

class MultiBoxLoss(nn.Module):
    def __init__(self, jaccard_thresh=0.5, neg_pos=3, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_thresh = jaccard_thresh  # 0.5 임계치(iou)
        self.negpos_ratio = neg_pos  # 3:1 Hard Negative Mining 비율
        self.device = device

    def forward(self, predictions, targets):
        """
        Parameters
        ----------
        predictions : SSD net output(tuple)
            (loc=torch.Size([num_batch, 8732, 4]), conf=torch.Size([num_batch, 8732, 21]), dbox_list=torch.Size [8732,4])

        targets : [num_batch, num_objs, 5]
            5는 정답 annotation [xmin, ymin, xmax, ymax, label_ind]

        Returns
        -------
        loss_l : tensor
            loc의 손실값
        loss_c : tensor
            conf 손실값
        """

        # SSD 모델의 출력이 튜플로 되어 있어 개별적으로 분리한다.
        loc_data, conf_data, dbox_list = predictions

        # item 수 파악
        num_batch = loc_data.size(0)  # batch size
        num_dbox = loc_data.size(1)  # DBox = 8732
        num_classes = conf_data.size(2)  # class 수 = 21

        # conf_t_label：각 DBox에 가장 가까운 정답 BBox의 라벨 저장
        # loc_t: 각 DBox에 가장 가까운 정답 BBox의 위치 정보 저장
        conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self.device)
        loc_t = torch.Tensor(num_batch, num_dbox, 4).to(self.device)

        # DBox와 정답 어노테이션 targets를 match한 결과 덮어쓰기
        for idx in range(num_batch):
            # 정답 annotation BBox와 label 취득
            truths = targets[idx][:, :-1].to(self.device)  # BBox
            # 라벨 
            labels = targets[idx][:, -1].to(self.device)
            # 디폴트 박스를 새로운 변수로 준비
            dbox = dbox_list.to(self.device)

            # loc_t:각 DBox에 가장 가까운 정답 BBox위치 정보
            # conf_t_label：각 DBox에 가장 가까운 BBox의 라벨 정보
            # 가장 가까운 BBox와의 IOUrk 0.5보다 작은 경우 정답 BBox의 라벨 conf_t_label은 배경 클래스 0으로함
            variance = [0.1, 0.2] # 보정 계산 시 사용
            match(self.jaccard_thresh, truths, dbox, variance, labels, loc_t, conf_t_label, idx)

        # ----------
        # 위치손실：loss_l
        # Smooth L1으로 손실 계산
        # ----------
        # 물체 감지한 BBox 꺼내는 마스크
        pos_mask = conf_t_label > 0  # torch.Size([num_batch, 8732])

        # pos_mask를 loc_data 크기로 변형
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)

        # Positive DBox의 loc_data와 train시키는 loc_t
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

        # 물체를 발견한 positive DBox의 오프셋 정보 loc_t의 손실 계산
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # ----------
        # 클래스 예측의 손실：loss_c 계산
        # 물체발견DBox 및 배경 클래스의 DBox 비율을 1:3이 되게함(배경class가 압도적으로 많으므로)
        # ----------
        batch_conf = conf_data.view(-1, num_classes)

        # 클래스 예측의 손실함수 계산(reduction을 none으로 하여 합을 취하지않고 차원 보존)
        loss_c = F.cross_entropy(batch_conf, conf_t_label.view(-1), reduction='none')

        # -----------------
        # Negative DBox 중 Hard Negative Mining으로 추출하는 것을 구하는 마스크
        # -----------------
        # 물체 발견한 Positive Dbox의 손실 0으로 함, 라벨이 0은 배경이므로 제외
        num_pos = pos_mask.long().sum(1, keepdim=True)  # 배치별 물체 클래스의 예측 수
        loss_c = loss_c.view(num_batch, -1)  # torch.Size([num_batch, 8732])
        loss_c[pos_mask] = 0  # 물체 발견한 디폴트박스는 손실 0

        # Hard Negative Mining, 각 DBox손실의 크기 loss_c 순위 idx_rank 구함
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        # 배경 DBox의 수 num_neg를 구함, 물체발견 DBox의 3배로 함
        num_neg = torch.clamp(num_pos*self.negpos_ratio, max=num_dbox)

        
        # 배경 DBox의 수 num_neg보다 손실이 큰 DBox를 취하는 마스크
        # torch.Size([num_batch, 8732])
        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        # -----------------
        # Negative DBox 중 Hard Negative Mining으로 추출할 것을 구하는 마스크
        # -----------------

        # 마스크모양을 고쳐 conf_data에 맞춤
        # pos_idx_mask : Positive DBox의 conf를 꺼내는 마스크
        # neg_idx_mask : Hard Negative Mining으로 Negative DBox의 conf를 꺼내는 마스크
        # pos_mask：torch.Size([num_batch, 8732])→pos_idx_mask：torch.Size([num_batch, 8732, 21])
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        # conf_data에서 pos와 neg만 꺼내서 conf_hnm으로함 torch.Size([num_pos+num_neg, 21])
        conf_hnm = conf_data[(pos_idx_mask+neg_idx_mask).gt(0)].view(-1, num_classes)

        conf_t_label_hnm = conf_t_label[(pos_mask+neg_mask).gt(0)] # torch.Size([pos+neg])

        # confidence의 손실함수 계산
        loss_c = F.cross_entropy(conf_hnm, conf_t_label_hnm, reduction='sum')

        N = num_pos.sum() # 물체를 발견한 바운딩박스의 수
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c