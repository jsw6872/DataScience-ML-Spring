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
## object detection 방식
![detection](../../img/detection_1.png)
### 2-stage Detector
물체의 위치를 찾는 문제(localization)와 분류 문제(classification)를 순차적으로 해결
- region proposals : 이미지 안에서 사물이 존재할 것 같은 곳에 나열
- 각각의 위치에 대해 feature extractor
![2-stage](../../img/2_stage.png)
### 1-Stage Detector
물체의 위치를 찾는 문제(localization)와 분류 문제(classification)를 한번에 해결
- 정확도가 다소 위 방식보다 낮을 수 있음
---
### R-CNN
이미지 안에서 아주 많은 region을 뽑고 똑같은 크기에 맞춰서 SVM으로 분류를 한다 -> 그 후 CNN을 통해 feature extraction
> 많은 region을 전부 CNN을 돌려줘야하는 문제점이 있음

### Fast R-CNN
SPPNet과 유사  
여러 바운딩 박스를 만든 뒤 feature map에 대응한 뒤에 뉴럴넷을 통과시킨다
![R-CNN](../../img/fast_r_cnn.png)

### Faster R-CNN
- RPN을 통해 기존 CPU에서 진행되던 selective Search의 문제 해결(GPU상에서 돌림)
  - feature map을 통해 물체가 있을 법한 곳에 예측할 수 있도록 함
- Region Proposal Network + Fase R-CNN
![based r_cnn](../../img/r_cnn.png)

#### Region Proposal Network
- 물체가 무엇인지보단 있는지 없는지 찾음
- 미리 정해놓은 바운딩박스(대충 어떤 크기의 물체들이 있을지 예상)를 k개 만들어놓고 찾는다
![RPN](../../img/rpn.png)

### YOLO
- Region Proposal 과정이 없기 때문에 좀 더 빠름
- 동시에 작용
