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
  - 