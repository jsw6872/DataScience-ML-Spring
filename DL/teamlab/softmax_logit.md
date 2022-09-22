# softmax & logit & cross entropy
## logit
- 확률이 0일 땐 음의 무한대, 1일 땐 양의 무한대로 가는 함수 (sigmoid와 역함수관계)
- $f(x) = log\frac{x}{1-x}$
- 이것을 이용하여 각 값들을 0~1 사이의 확률값으로 변환이 가능함
- 분류를 하는 딥러닝에서 logit은 확률화되지 않은 예측 결과를 일컫는다(softmax의 input값)

## softmax
- 다중 분류 문제에서 logit을 통해 나온 값을 전체 확률이 1이 되게 확률값으로 변환
- k를 2로 두면 sigmoid로 환원 (sigmoid의 일반형으로 생각)
![cross_entropy](../img/softmax.png)

## corss entropy
- multinomial classification에서의 loss function
- q는 softmax를 거쳐서 나온 확률값, p는 실제 y값 -> 이 둘이 모두 쓰여서 cross entropy
![cross_entropy](../img/cross_ent.png)

---
- 코드에서 사용될때는 서로 다른 용도(sigmoid는 activation에, softmax는 classification에)로 사용