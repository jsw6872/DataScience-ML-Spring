# ensenble model
- 하나의 모델이 아니라 여러개 모델의 투표로 Y값을 예측
- Regression 문제에서는 평균값으로 예측
- meta - classifier
  - 하나의 classifier가 다른 것들과 연계되어 만들어지는
- stacking (meta - ensemble) 등으로 발전
  - ensemble + ensemble
- 학습은 오래 걸리나 성능이 매우 좋음
![Ensemble](../img/ensemble.png)
![ensemble](../img/ensemble_2.png)

### key words
- vallila ensemble (voting)
- Boosting (sampling을 통해)
- Bagging (Boosting을 aggregation)
- Adaptive boosting(AdaBoost)
- XGBoost
- Light GBM
---
## Voting classifier
- 가장 기본적인 ensemble classifier
- 여러 개의 Model의 투표를 통해 최종 선택 실시
- Majority voting or Vallila ensemble 모델이라고 불림

### sklearn.esnsemble.VotingClassifier
- estimators : list형태로 여러 model들이 들어간다
- voting
  - hard : sum해서 다수
  - soft : 확률로 변환하여 각 feature의 weight값이 가장 높은 친구 선택
- weights : 각각의 모델이나 클래스에 weight를 준다
- n_jobs : cpu를 얼마나 쓸 것인가
---
## Sampling
- 단순히 같은 데이터 셋으로 만들면 ensemble의 효과를 볼 수 없음
- 다양한 sampling dataset으로 다양한 classifier를 만들어야함 

### Bootstrapping - Bagging
![bootstrapping](../img/bootstrapping.png)
- 데이터를 외부 추가 없이 추출하는 것

#### 0.632 bootstrap
![632](../img/632.png)

## bootstrap을 이용해 data ensemble을 만드는 것을 Bagging(Bootstrap Aggregatin)
- Bootstrap의 subset smaple로 모델 N개를 학습 (앙상블)
- Vallila ensemble과 달리 하나의 모델에 다양한 데이터를 넣는다
- High variance(over fitting이 심한) 모델에 적합
- Regressor(평균 or median), Classifier 모두 존재
  - DT의 경우 prunning 할 필요가 없는데 Bootstrap은 overfitting된걸 합쳐서 성능을 내기 때문에 overfitting이 되어도 그냥 놔둔다

### out of bag error
- OOB (out of bag error)인 sub sampling을 하고 남은 데이터로 모델 성능을 측정
- validation set과 유사

### sklearn.ensemble.BaggingClassifier/BaggingRegressor
- base_estimator : 하나의 모델을 넣는 것이기 때문에 model name
- n_estimators : bootstrap을 몇 개 할 것인가
- max_samples : 하나의 데이터를 얼마나 넣을 것인가
- max_features : 몇 개만 사용해서 학습을 할 것인가
- bootstrap : 
- bootstrap_features : 
- oob_score : bagging의 성능 측정을 위한 Validation
- warm_start : 이전의 학습된 데이터를 넣을 것인지
- n_jobs : cpu 몇 개