# **(week2) ML LifeCycle 회고록**

## 1. 주간 키워드

### (1) course 
* Linear regression
* logistic regression
* activation fucntion (sigmoid, softmax, tanh, relu)
* Neural Network
* backpropagation
* RNN model
* attention
* transformer

### (2) Paper
* Generative Adversarial Nets review

## 2. 학습 회고

### (1) 학습 진행사항
* 12일: 1~3강 / GAN paper review
* 13일: GAN paper review / 4~6강
* 14일: 6~10강
* 15일: 기본과제 1~3
* 16일: 복습 및 학습정리 수행

### (2) 학습 태도
* 잘했던것, 좋았던것, 계속할것
    * 수식을 이해하면서 논문 리뷰 진행
    * github.io 블로그 개설
    * 저녁 7~8시 사이에 주기적인 운동(수영)

* 잘못했던것, 아쉬운것, 부족한것
    * 강의 및 과제 스케줄을 타이트하게 잡아서 목요일에서 금요일 넘어갈 때 밤샘...
    * Backpropagation 및 Transformer에 대한 지식이 부족함

* 도전할것, 시도할것
    * 멘토님께서 지도해주신 포트폴리오와 CV 제작
    * 과제 및 강의를 통해 다양한 activation function에 대한 코드 짜보기
    * 효율적인 블로그 정리 방법 체득

* 키워드
    * torch는 소중하구나

## 3. 피어세션 회고

* 12일: Attenion is all you need paper review (발표자: 이시하)
* 13일: Generative Adversarial Nets paper review (발표자: 김명철)
* 14일: ImageNet Classification with Deep Convolutional Neural Networks paper review (발표자: 박민영)
* 15일: 휴일
* 16일: 팀 회고

## 4. 멘토링 회고

* 포트폴리오 구성: 각자 만들어서 DM으로 보내서 피드백 받기
* 캐글 추천
    * 데이터 전처리 연습 (pandas, numpy, DF)
    * 정형: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
    * NLP: https://www.kaggle.com/competitions/nlp-getting-started
* 논문 읽는 법, 속도
    * 실험 결과 자세하게 보지 않아도 괜찮다.
    * 실험: ablation study → 분석적 결과
    * 논문 제시하는 method, 제일 열심히 읽어야 하는 것은 intro
    * 발표자료: intro(motivation, background), method(방법론), 실험(분석 위주로 보고 디스커션 해보고), 비슷한 방법론이 다른 연구,모델에 적용된 것 찾아서 디스커션 해보기 (e.g., residual learning 이 사용된 다른 연구나  모델이 뭐가 있는지)
* 취업 목표 키워야 할 역량
    * AI가 처음이면 우선 부캠 강의, 프로젝트 열심히
    * CV 공부하고, 나중에 NLP
* NLP 기초
    * 텍스트 전처리 (영어, 한국어), tokenizer
    * Model: RNN, LSTM, Seq2Seq, attention, Transformer
    * Pre-training: Masked language model, auto-regressive language model, LLM, VLM
    * Tool: huggingface
    * 강의: https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4
    * Lec 9 ~ 12, transformer 까지 알고 있다는 전제하에
* 모델 경량화
    * https://jaeho-lee.github.io/docs/teaching/spring24/
* 블로그
    * 꾸준히 정리

## 5. 주요 코드

### (1) cost fuction


```python
import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    제곱 오차를 계산하는 코드
    """
    squared_errors = (y_true - y_pred)**2

    """
    평균 제곱 오차를 계산하는 코드
    """
    mse = np.mean(squared_errors)

    return mse

def r_squared(y_true, y_pred):
    """
    실제 값의 평균을 계산하는 코드
    """
    mean_y_true = np.mean(y_true)

    """
    총 제곱합(TSS)을 계산하는 코드
    """
    tss = np.sum((y_true - mean_y_true) ** 2)

    """
    잔차 제곱합(RSS)을 계산하는 코드
    """
    rss = np.sum((y_true - y_pred) ** 2)

    """
    결정 계수를 계산하는 코드
    """
    r2 = 1 - (rss / tss)

    return r2
```

### (2) TwoLayerNN with numpy(forward, backpropagation)


```python
class TwoLayerNN:
    """ a neural network with 2 layers """

    def __init__(self, input_dim, num_hiddens, num_classes):
        """
        이 코드는 수정하실 필요가 없습니다.
        """
        self.input_dim = input_dim
        self.num_hiddens = num_hiddens
        self.num_classes = num_classes
        self.params = self.initialize_parameters(input_dim, num_hiddens, num_classes)

    def initialize_parameters(self, input_dim, num_hiddens, num_classes):
        """
        Question (a)

        Xavier Initialization를 이용하여 파라미터를 초기화 하십시오.

        - refer to https://paperswithcode.com/method/xavier-initialization for Xavier initialization

        Inputs
        - input_dim
        - num_hiddens: hidden units의 수
        - num_classes: 클래스의 수
        Returns
        - params: 초기화된 파라미터들의 딕셔너리
        """
        params = {}

        params['w1'] = np.random.randn(input_dim, num_hiddens) / np.sqrt(input_dim)
        params['b1'] = np.zeros((1, num_hiddens))

        params['w2'] = np.random.randn(num_hiddens, num_classes) / np.sqrt(num_hiddens)
        params['b2'] = np.zeros((1, num_classes))

        return params
        

    def forward(self, X):
        """
        Question (b)

        2-layer NN의 forward pass을 정의하고 구현하십시오.
        - ff_dict는 다음 문제에서 backpropagation을 실행하는데 사용됩니다.

        주어진 네트워크의 구조는 다음과 같습니다.

          y = softmax(sigmoid(X W1 + b1) W2 + b2)

        - X: input 행렬 (N, D)
        - y: 실제 클래스 분포 행렬 (N, C)
        - N: (전체 데이터 셋 혹은 미니 배치) 예제의 수
        - D: feature dimensionality
        - C: 클래스의 수

        Inputs
        - X: input 행렬 (N, D)

        Returns
        - y: 모델의 output
        - ff_dict: 모든 fully connected units과 activations의 딕셔너리
        """
        ff_dict = {}

        # sigmoid activation function
        w1_b1 = X.dot(self.params['w1']) + self.params['b1']
        sigmoid_act = 1 / (1 + np.exp(-w1_b1))

        ff_dict['w1_b1'] = w1_b1
        ff_dict['sigmoid_act'] = sigmoid_act
        
        # softmax activation function
        w2_b2 = sigmoid_act.dot(self.params['w2']) + self.params['b2']

        exp_scores = np.exp(w2_b2)
        softmax_act = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        ff_dict['w2_b2'] = w2_b2
        ff_dict['softmax_act'] = softmax_act

        return softmax_act, ff_dict


    def backward(self, X, Y, ff_dict):
        """
        Question (c)

        2-layer NN의 backpropagation을 구현하고 모든 모델 파라미터들의 gradients 딕셔너리를 반환 하십시오.

        Inputs:
         - X:input 행렬 (B, D)
             B는 미니 배치 예제의 수, D는 feature dimensionality

         - Y: 실제 클래스 분포 행렬 (B, C)
              B는 미니 배치 예지의 수, C는 클래스의 수
         - ff_dict: 모든 fully connected units과 activations의 딕셔너리

        Returns:
         - grads: weight과 bia에 대응하는 gradients들을 포함하는 딕셔너리
        """

        grads = {}
        m = X.shape[0]

        # 2번째 layer 역전파
        grad_y_pred = ff_dict['softmax_act'] - Y
        grads['b2'] = np.sum(grad_y_pred, axis=0, keepdims=True) / m
        grads['w2'] = ff_dict['sigmoid_act'].T.dot(grad_y_pred) / m

        # 1번째 layer 역전파
        grad_sigmoid = grad_y_pred.dot(self.params['w2'].T) * (ff_dict['sigmoid_act'] * (1 - ff_dict['sigmoid_act']))

        grads['b1'] = np.sum(grad_sigmoid, axis=0, keepdims=True) / m
        grads['w1'] = X.T.dot(grad_sigmoid) / m

        return grads


    def compute_loss(self, Y, Y_hat):
        """
        엔트로피 손실을 계산하는 과정입니다.
        이 코드는 수정하실 필요가 없습니다.

        Inputs
            Y: 실제 라벨
            Y_hat: 예측된 라벨
        Returns
            loss:
        """
        loss = -(1/Y.shape[0]) * np.sum(np.multiply(Y, np.log(Y_hat)))
        return loss


    def train(self, X, Y, X_val, Y_val, lr, n_epochs, batch_size, log_interval=1):
        """
        미니 배치 gradient descent(경사 하강법)을 실행하는 과정입니다.
        이 코드는 수정하실 필요가 없습니다.

        Inputs
        - X
        - Y
        - X_val: val 데이터
        - Y_Val: val 라벨
        - lr: learning rate
        - n_epochs: 실행할 에포크의 수
        - batch_size
        - log_interval: 로그를 남길 에포크 간격
        """
        for epoch in range(n_epochs):
            for X_batch, Y_batch in load_batch(X, Y, batch_size):
                self.train_step(X_batch, Y_batch, batch_size, lr)
            if epoch % log_interval==0:
                Y_hat, ff_dict = self.forward(X)
                train_loss = self.compute_loss(Y, Y_hat)
                train_acc = self.evaluate(Y, Y_hat)
                Y_hat, ff_dict = self.forward(X_val)
                valid_loss = self.compute_loss(Y_val, Y_hat)
                valid_acc = self.evaluate(Y_val, Y_hat)
                print('epoch {:02} - train loss/acc: {:.3f} {:.3f}, valid loss/acc: {:.3f} {:.3f}'.\
                      format(epoch, train_loss, train_acc, valid_loss, valid_acc))


    def train_step(self, X_batch, Y_batch, batch_size, lr):
        """
        Question (d)

        gradient descent를 사용하여 파라미터를 업데이트 하십시오.

        Inputs
        - X_batch
        - Y_batch
        - batch_size
        - lr: learning rate
        """

        Y_prediction, ff_dict = self.forward(X_batch)

        grads = self.backward(X_batch, Y_batch, ff_dict)

        self.params['w1'] -= lr * grads['w1']
        self.params['b1'] -= lr * grads['b1']
        self.params['w2'] -= lr * grads['w2']
        self.params['b2'] -= lr * grads['b2']


    def evaluate(self, Y, Y_hat):
        """
        Question (e)

        classification의 accuracy를 계산하십시오.

        Inputs
        - Y: (N, C) 형태의 배열(원-핫 인코딩)
             C는 클래스의 수
        - Y_hat: (N, C)형태의 배열(소프트 맥스)
             C는 클래스의 수

        Returns
            accuracy: float 형태의 classification accuracy
        """

        predictions = np.argmax(Y_hat, axis=1)
        true_classes = np.argmax(Y, axis=1)

        accuracy = np.mean(predictions == true_classes)
    
        return accuracy

        
```

### (3) Multi-head self attention


```python
import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
      def __init__(self, num_heads, hidden_size):
        super().__init__()
        self.num_heads = num_heads
        self.attn_head_size = int(hidden_size / num_heads)
        self.head_size = self.num_heads * self.attn_head_size

        self.Q = nn.Linear(hidden_size, self.head_size)
        self.K = nn.Linear(hidden_size, self.head_size)
        self.V = nn.Linear(hidden_size, self.head_size)
        # TODO: 최종적인 Z를 얻어내기 위해 필요한 아래 코드의 빈 칸을 채워주세요.
        # ==================================================
        self.dense = nn.Linear(hidden_size, self.head_size)
        # ==================================================

      def tp_attn(self, x):
        x_shape = x.size()[:-1] + (self.num_heads, self.attn_head_size)
        x = x.view(*x_shape)
        return x.permute(0, 2, 1, 3)

      def forward(self, hidden_states):
        Q, K, V = self.Q(hidden_states), self.K(hidden_states), self.V(hidden_states)
        Q_layer, K_layer, V_layer = self.tp_attn(Q), self.tp_attn(K), self.tp_attn(V)

        # TODO: forward() 메서드 내에서 10강에서 배웠던 Self-attention 수식에 따라, 어느 부분에 Q_layer, K_layer, V_layer가 들어가야 할지 빈칸을 채우시오.
        # ==================================================
        attn = torch.matmul(Q_layer, K_layer.transpose(-1, -2)) / math.sqrt(self.attn_head_size)
        attn = nn.Softmax(dim=-1)(attn)
        output = torch.matmul(attn, V_layer)
        # ==================================================

        # TODO: forward() 메서드 내에서 10강에서 배웠던 Multi-head Self-attention의 특징인 concatenate 하고, original input size로 linear transformation하는 부분의 빈 칸을 채워주세요.
        # ==================================================
        output = output.permute(0, 2, 1, 3).contiguous()
        output_shape = output.size()[:-2] + (self.head_size,)
        output = output.view(*output_shape)

        Z = self.dense(output)
        # ==================================================

        return Z
```
