# Semantic Segmentation
## Semantic Segmentation란 무엇인가
이미지의 픽셀마다 분류해서 어떤 라벨에 속하는지 분류
> 자율주행 등에 활용

### convolutionalization
- Deconvolution
  - feature map이 줄어드는 것이 아니라 늘어난다
  - 엄밀히 말하면 역Conv는 아니지만 이렇게 생각하면 쉬움


# Detection
바운딩 박스로 찾는 것
## R-CNN
이미지 안에서 아주 많은 region을 뽑고 똑같은 크기에 맞춰서 SVM으로 분류를 한다 -> 그 후 CNN을 통해 feature extraction
> 많은 region을 전부 CNN을 돌려줘야하는 문제점이 있음
## SPPNet
이미지에서 바운딩 박스를 뽑고 CNN을 1번 돌려 나온 feature map의 텐서만 들고 와서 비교
## Fast R-CNN
SPPNet과 유사  
여러 바운딩 박스를 만든 뒤 feature map에 대응한 뒤에 뉴럴넷을 통과시킨다
![R-CNN](../../img/fast_r_cnn.png)

## Faster R-CNN
- Region Proposal Network + Fase R-CNN

### Region Proposal Network
- 물체가 무엇인지보단 있는지 없는지 찾음
- 미리 정해놓은 바운딩박스(대충 어떤 크기의 물체들이 있을지 예상)를 k개 만들어놓고 찾는다
![RPN](../../img/rpn.png)

## YOLO
- Region Proposal 과정이 없기 때문에 좀 더 빠름
- 동시에 작용
