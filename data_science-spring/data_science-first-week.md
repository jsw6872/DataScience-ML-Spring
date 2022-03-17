# 데이터 전처리
## 전처리의 필요성
- 모델 input 시 문제 발생
  - 누락 데이터
    - NaN -> 누락된 값을 포함한 행 drop, 누락된 값을 변환
    ```python
    df.isnull().sum() # 각 칼럼 별로 NaN 값이 있는 합이 나옴
    df.dropna(axis = 0) # NaN 포함한 행 없애기
    df.dropna(axis = 1) # NaN 포함한 열 없애기
    df.fillna(value) # NaN에 값 고정해서 넣기
    df['~'] = df['~'].fillna(df['~'].mean()) 
    df['~'] = df['~'].fillna(df.mean()) # 각 칼럼에 해당하는 평균 값 자동으로 할당 
    ```
  - 결측치, 잘못된 값
  - 중복되는 값
    ```python
    df.drop_duplicates(keep='first, last')
    ```
# Model
## 모형의 분류
- 예측해야하는 값이 실수인가
  - Y : 회귀
  - N : 분류
- 정답을 알고 있는 데이터인가
  - Y : 지도학습
  - N : 비지도학습
- patameter로 모델 표현이 가능한가
  - Y : parametric method
  - N : Non-parametric Method
- --
## 비용
- 비용 : 현재 parameter로 주어진 데이터를 얼마나 잘 표현했는가를 측정한 함수(비교)
  - y : 데이터의 실제 레이블 값
  - $\hat{y}$ : 현재 회귀식을 통해 예측한 값
  - $cost = f(y,\hat(y))=\sum_{i=1}^{n} (y^i - \hat{y}^i)^2$
  - cost를 통해 주어진 데이터로 방정식을 구하고 $\theta_0$, $\theta_1$에 대해 각각 미분한 뒤 연립 -> 최적의 parameter
- --
## 기존 선형회귀 모형의 가정
- 서로 다른 변수 간에 독립적 (독립성)
- 독립변수의 증가에 따라 종속변수 비례하여 바뀜 (선형성)
## 선형회귀의 확장
- 독립성을 제거하여 feature 개수를 줄인다
  - feature 개수가 줄면 학습효과가 더 명확, over-fitting이 덜 일어남
- 선형성을 제거해 다차식 관계로 더 유연한 회귀선을 나타낼 수 있다.
---
## 범주형 변수의 처리
- 독립변수 2개
  > ex) 남 : 1, 여 : 0 과 같이 표현하는 방법이 있다
- 3개일 떄 ?? (one-hot enoding)