# cross validation ( 교차검증 )
- 모델의 일반화 능력을 더 합리적으로 안전하게 추정할 수 있는 방법
## train test split step
1. 데이터 분할
   1. 데이터 중 p% 를 무작위로 추출해 training data로 선정
   2. 나머지 (1-p)%의 데이터를 testing data 로 선정
2. 모델 학습
   1. training data를 이용하여 모든 비교하고자 하는 모델들을 수행
      - ex) 1차 회귀, 2차 회귀, 3차 회귀,,..
3. 모델 검증
   1. testing data 를 이용하여 학습된 각각의 모댈들의 testing 정확도를 측정
4. 모델 선정
   1. 모델들 중 가장 높은 테스팅 정확도를 보인 모델을 최적 모델로 선택

### 문제점
- 데이터 분할 시 나뉘어지는 방식에 따라 모델 검증 결과에 큰 편차 존재할 수 있음
- 운에 맡겨진다
---
## 교차검증이란
- train test split의 단점을 보완하기 위해 나온 방법
- 한번의 시도로 데이터를 나누다보면 우연에 의해 치우칠 수 있으니 데이터를 여러번 train test split함
### cross validation step
1. 데이터를 무작위로 섞고 같은 수를 가진 k개의 집단으로 나눔
2. 첫번째 그룹을 testing data로 선정, 나머지 training data
3. 두번째 그룹을 testing data 로 선정 나머지 training data
4. ,,,
5. 이 모든게 한 모델에 대해 이루어짐 -> n개의 모델이 이 과정 반복

## LOCV (leave one out of cross validation)
- k-fold cross validation의 극단적 형태
- k = 데이터의 개수
- 1개만 train 나머지는 test를 n번 반복
