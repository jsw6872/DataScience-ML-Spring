# 성능은 어떻게 올릴까?
## ensenble model
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

