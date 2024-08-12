## **Logistic Regression 과제 답안**
---

### **1) 과제 개요**

- pytorch를 이용하여 Cars - Purchase Decision 데이터에 대해 차량 구매 여부를 예측하는 로지스틱 회귀 분석을 시행합니다.

### **2) 실습 진행 목적 및 배경**

- pytorch를 이용하여 로지스틱 회귀 분석을 위한 모델을 구축하고, 로지스틱 회귀 분석을 실시하는 방법을 공부합니다.

- 데이터에 맞는 데이터셋 클래스를 구축함으로써 재사용이 수월한 코드를 작성하는 방법을 공부합니다.

### **3) 실습 수행으로 얻어갈 수 있는 역량**

- 로지스틱 회귀 분석에 필요한 pytorch 구성 요소들을 구현할 수 있다.

- 실제 데이터에 대해 로지스틱 회귀 분석을 실시할 수 있다.

- 주어진 데이터셋에 맞는 Dataset 클래스를 작성할 수 있다.

### **4) 과제 핵심 내용**
- Cars - Purchase Decision 데이터에 맞는 pytorch Dataset 클래스를 작성합니다.

- 로지스틱 회귀 모델을 pytorch로 구현하고, 이를 학습시킵니다.

- 머신러닝을 통해 얻은 모델과 통계적인 방법으로 얻은 모델의 성능을 비교합니다.

### **5) 데이터셋 개요 및 저작권 정보**

- 사용 데이터셋: [Cars - Purchase Decision Dataset](https://www.kaggle.com/datasets/gabrielsantello/cars-purchase-decision-dataset)
  - 차 구매를 고민하고 있는 1000명의 고객에 대한 성별, 나이, 연봉, 그리고 구매 여부가 담긴 데이터
- 저작권 정보: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)

### **6) Required Package**

```python
scikit-learn >= 1.4.2
torch >= 2.3.0
pandas >= 2.2.2
numpy >= 1.26.4
```

---
### **과제 진행 방법**
`[CODE START]`와 `[CODE END]` 사이의 코드들을 완성하는 것을 목표로 합니다.

---
## **Cars - Purchase Decision Dataset을 위한 pytorch Dataset 클래스 작성하기**

#### 문제 1. 데이터셋 개요에 첨부된 링크에서 데이터를 다운받고, 해당 데이터셋을 다루는 Dataset 클래스를 작성하세요.
1.1 `__init__` 함수를 완성하세요.

- file_path에 해당하는 파일을 읽어 변수 `data`에 저장하세요. (힌트: pandas 라이브러리의 read_csv 함수를 사용하세요.)

- 주어진 데이터에서 *User ID*와 *Purchased* 열을 제외한 데이터를 변수 `X`에 저장하고, *Purchased* 열을 변수 `y`에 저장하세요.

- 랜덤하게 학습용과 평가용으로 나눈 데이터 중에서 `mode` 인자로 들어온 값에 따라 Float Tensor 타입으로 변환 후에 각각 인스턴스 변수 `X`와 `y`에 저장하세요.

1.2 `__len__` 함수를 완성하세요.

- 해당 함수가 데이터의 개수를 반환하도록 작성하세요.

1.3 `__getitem__` 함수를 완성하세요.

- 주어진 인덱스에 맞는 예측변수와 종속변수 값을 tuple 형태로 반환하도록 작성하세요.

- 편의를 위해 종속변수의 차원이 0차원이 아닌, 1차원이 되도록 작성하세요.


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    


```python
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class CarsPurchaseDataset(Dataset):
    def __init__(self, file_path="/content/drive/MyDrive/NAVER_BoostCamp/과제/car_data.csv", mode="train"):
        # [CODE START]
        data = pd.read_csv(file_path, sep = ",", header = 0)
        # [CODE END]

        # [CODE START]
        X = pd.DataFrame(data.drop(['User ID', 'Purchased'], axis=1))
        y = data.Purchased
        # [CODE END]

        X['Gender'] = X.Gender.apply(lambda x: 0 if x == "Male" else 1)
        scaler = StandardScaler()
        X[['Age','AnnualSalary']] = scaler.fit_transform(X[['Age', 'AnnualSalary']])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # [CODE START]
        if mode == "train":
            self.X = torch.tensor(X_train.values, dtype=torch.float32).to(device)
            self.y = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
        else:
            self.X = torch.tensor(X_test.values, dtype=torch.float32).to(device)
            self.y = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)
        # [CODE END]

    def __len__(self):
        # [CODE START]
        return len(self.X)
        # [CODE END]

    def __getitem__(self, idx):
        # [CODE START]
        return self.X[idx], self.y[idx]
        # [CODE END]
```

올바르게 작성했을 경우, 학습용 데이터셋와 평가용 데이터셋에 대한 길이가 각각 800, 200으로 반환되어 아래 코드가 에러 없이 잘 수행됩니다.


```python
train_data = CarsPurchaseDataset()
test_data = CarsPurchaseDataset(mode="test")

assert len(train_data) == 800
assert len(test_data) == 200
```

또한 학습용과 평가용 데이터셋 객체 모두 반환하는 데이터가 튜플 형태로 주어질 것이며, 각각의 모양은 (3,)과 (1,)로 나타나야 합니다.

즉, 아래 코드가 에러 없이 잘 수행되어야 합니다.


```python
X, y = next(iter(train_data))
assert X.shape == (3,)
assert y.shape == (1,)

print("Data is loaded correctly!")
print(X)
```

    Data is loaded correctly!
    tensor([ 1.0000, -0.1968,  2.1703], device='cuda:0')
    

### 문제 2. 생성한 데이터셋 객체를 바탕으로 DataLoader 객체를 생성하세요.
- 학습용 데이터로더 객체의 경우, 배치 사이즈는 32로 하며, 랜덤으로 셔플된 데이터가 반환되도록 생성하세요.
- 평가용 데이터로더 객체의 경우, 배치 사이즈는 64로 하여 생성하세요.
- 데이터로더 객체를 저장할 변수는 각각 `train_loader`와 `test_loader`입니다.


```python
from torch.utils.data import DataLoader
# [CODE START]
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
# [CODE END]
```

올바르게 작성했다면, 학습용 데이터로더와 평가용 데이터로더 모두 아래와 같이 올바른 모양의 텐서를 반환할 것입니다.


```python
batched_X, batched_y = next(iter(train_loader))
assert batched_X.shape == (32, 3)
assert batched_y.shape == (32, 1)

batched_X, batched_y = next(iter(test_loader))
assert batched_X.shape == (64, 3)
assert batched_y.shape == (64, 1)

print("Shape of batch is correct!")
```

    Shape of batch is correct!
    

---
## **로지스틱 회귀 모형 작성하고 학습하기**

이제, 각자 환경에 맞는 device를 설정합니다.


```python
device = "cuda"
```

### 문제 3. 로지스틱 회귀 모델을 위한 클래스를 작성하세요.

3.1 `__init__` 함수 완성하기

- 인스턴스 변수 `linear`에 `input_size`에 저장된 차원의 데이터를 받아 1차원의 데이터를 반환하는 선형 layer를 저장하세요.

3.2 `forward` 함수 완성하기

- 로지스틱 회귀 모델에 맞는 출력값을 반환하도록 작성하세요.


```python
import torch.nn as nn

class LogisticRegressionNN(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegressionNN, self).__init__()
        # [START CODE]
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
        # [END CODE]

    def forward(self, x):
        # [START CODE]
        output = self.linear(x)
        output_sig = self.sigmoid(output)
        return output_sig
        # [END CODE]
```

주어진 데이터의 예측변수의 차원은 3이므로 이에 맞게 모델을 초기화합니다.


```python
input_size = 3
model = LogisticRegressionNN(input_size)
```

### 문제 4. 로지스틱 회귀 모델을 학습하기 위한 손실함수 객체를 알맞게 생성하여 변수 `criterion`에 저장하세요.


```python
# [START CODE]
criterion = nn.BCELoss()
# [END CODE]
```

### 문제 5. 로지스틱 회귀 모델을 학습하기 위한 옵티마이저 객체를 생성하여 변수 optimizer에 저장하세요.
- 옵티마이저의 종류는 수강생분께서 선택하시면 됩니다.
- 여러 옵티마이저를 가지고 학습해보시기를 권장합니다.


```python
import torch.optim as optim
# [START CODE]
optimizer = optim.SGD(model.parameters(), lr=0.01)
# [END CODE]
```

### 문제 6. 모델을 학습하는 함수 `train`을 완성하세요.

함수가 받는 인자는 순서대로 모델 객체, 손실함수 객체, 옵티마이저 객체, 데이터로더 객체, 디바이스 종류, 그리고 epoch 횟수입니다.

아래 요구하는 여섯가지 동작은 각 배치마다 수행하는 것으로, 이에 대한 코드를 순서대로 구현하여 완성하세요.

6.1 옵티마이저에 저장된 그래디언트 정보를 초기화합니다.

6.2 모델의 출력값을 계산하여 변수 `outputs`에 저장합니다.

6.3 모델의 출력값과 실제 label값을 바탕으로 손실함수를 계산합니다.

6.4 손실함수에 대해 역전파 방식으로 그래디언트를 계산합니다.

6.5 옵티마이저의 최적화 기법을 수행합니다.

6.6 변수 `epoch_loss`에 계산된 손실함수 값을 저장합니다.

<font color='red'><b>*주의: 모델과 데이터가 같은 디바이스에 있어야 에러없이 잘 동작합니다.*</font>


```python
def train(model, criterion, optimizer, dataloader, device, num_epoch=100):
    model.train()
    model.to(device)

    for epoch in range(num_epoch):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            # [START CODE]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # [END CODE]
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}")
```

학습 코드를 올바르게 작성했다면, 아래 코드가 잘 실행되어 로지스틱 회귀 모델이 잘 수행될 것입니다.


```python
train(model, criterion, optimizer, train_loader, device, 100)
```

    Epoch 1, Loss: 18.01440942287445
    Epoch 2, Loss: 17.158704578876495
    Epoch 3, Loss: 16.412197172641754
    Epoch 4, Loss: 15.761390924453735
    Epoch 5, Loss: 15.191308081150055
    Epoch 6, Loss: 14.692788571119308
    Epoch 7, Loss: 14.252231776714325
    Epoch 8, Loss: 13.863539576530457
    Epoch 9, Loss: 13.518257349729538
    Epoch 10, Loss: 13.210152328014374
    Epoch 11, Loss: 12.935680657625198
    Epoch 12, Loss: 12.687653094530106
    Epoch 13, Loss: 12.46413403749466
    Epoch 14, Loss: 12.26170039176941
    Epoch 15, Loss: 12.077678263187408
    Epoch 16, Loss: 11.909357458353043
    Epoch 17, Loss: 11.756370216608047
    Epoch 18, Loss: 11.614862233400345
    Epoch 19, Loss: 11.485519140958786
    Epoch 20, Loss: 11.365313977003098
    Epoch 21, Loss: 11.254873633384705
    Epoch 22, Loss: 11.152529150247574
    Epoch 23, Loss: 11.057174116373062
    Epoch 24, Loss: 10.968585699796677
    Epoch 25, Loss: 10.886023610830307
    Epoch 26, Loss: 10.808537125587463
    Epoch 27, Loss: 10.736326426267624
    Epoch 28, Loss: 10.668383091688156
    Epoch 29, Loss: 10.604500502347946
    Epoch 30, Loss: 10.5444954931736
    Epoch 31, Loss: 10.488022953271866
    Epoch 32, Loss: 10.434814780950546
    Epoch 33, Loss: 10.383774191141129
    Epoch 34, Loss: 10.336514234542847
    Epoch 35, Loss: 10.291091859340668
    Epoch 36, Loss: 10.248650968074799
    Epoch 37, Loss: 10.208366215229034
    Epoch 38, Loss: 10.169976711273193
    Epoch 39, Loss: 10.13301095366478
    Epoch 40, Loss: 10.098521828651428
    Epoch 41, Loss: 10.065489798784256
    Epoch 42, Loss: 10.033073127269745
    Epoch 43, Loss: 10.003007054328918
    Epoch 44, Loss: 9.97438895702362
    Epoch 45, Loss: 9.947263434529305
    Epoch 46, Loss: 9.920402765274048
    Epoch 47, Loss: 9.89519253373146
    Epoch 48, Loss: 9.871281117200851
    Epoch 49, Loss: 9.848024845123291
    Epoch 50, Loss: 9.825808882713318
    Epoch 51, Loss: 9.804468035697937
    Epoch 52, Loss: 9.784089356660843
    Epoch 53, Loss: 9.764285206794739
    Epoch 54, Loss: 9.745845079421997
    Epoch 55, Loss: 9.72753831744194
    Epoch 56, Loss: 9.710094302892685
    Epoch 57, Loss: 9.692815512418747
    Epoch 58, Loss: 9.676629900932312
    Epoch 59, Loss: 9.661699563264847
    Epoch 60, Loss: 9.646061271429062
    Epoch 61, Loss: 9.632367849349976
    Epoch 62, Loss: 9.618114084005356
    Epoch 63, Loss: 9.604225605726242
    Epoch 64, Loss: 9.591615229845047
    Epoch 65, Loss: 9.578996270895004
    Epoch 66, Loss: 9.566552996635437
    Epoch 67, Loss: 9.554883599281311
    Epoch 68, Loss: 9.543127834796906
    Epoch 69, Loss: 9.532555341720581
    Epoch 70, Loss: 9.52185994386673
    Epoch 71, Loss: 9.510801315307617
    Epoch 72, Loss: 9.501300036907196
    Epoch 73, Loss: 9.491938292980194
    Epoch 74, Loss: 9.482217997312546
    Epoch 75, Loss: 9.473188817501068
    Epoch 76, Loss: 9.464608937501907
    Epoch 77, Loss: 9.455835849046707
    Epoch 78, Loss: 9.44742202758789
    Epoch 79, Loss: 9.439649939537048
    Epoch 80, Loss: 9.431150436401367
    Epoch 81, Loss: 9.4246826171875
    Epoch 82, Loss: 9.416797965765
    Epoch 83, Loss: 9.409613892436028
    Epoch 84, Loss: 9.402928829193115
    Epoch 85, Loss: 9.395883947610855
    Epoch 86, Loss: 9.389215648174286
    Epoch 87, Loss: 9.382949382066727
    Epoch 88, Loss: 9.37701365351677
    Epoch 89, Loss: 9.370909169316292
    Epoch 90, Loss: 9.364820331335068
    Epoch 91, Loss: 9.359213054180145
    Epoch 92, Loss: 9.35332177579403
    Epoch 93, Loss: 9.348099574446678
    Epoch 94, Loss: 9.342719584703445
    Epoch 95, Loss: 9.337729275226593
    Epoch 96, Loss: 9.332984209060669
    Epoch 97, Loss: 9.327614665031433
    Epoch 98, Loss: 9.323302939534187
    Epoch 99, Loss: 9.318579077720642
    Epoch 100, Loss: 9.313909322023392
    

### 문제 7. 모델을 평가하는 함수 `test`를 완성하세요.
함수가 받는 인자는 순서대로 모델 객체와 데이터로더 객체입니다.

7.1 모델의 출력값을 계산하여 변수 `y_pred`에 저장하세요.

7.2 `y_pred`에 저장된 값 각각에 대해 0.5보다 크면 레이블을 1로, 0.5보다 작으면 0으로 레이블을 예측하도록 하여 예측 레이블을 변수 `y_pred_class`에 저장하세요.

7.3 예측한 레이블과 실제 레이블이 일치하는 개수를 세어 변수 `correct`에 더해주세요.

7.4 배치의 개수를 세어 변수 `n_data`에 더해주세요.


```python
def test(model, dataloader):
    model.eval()
    correct = 0 # 예측한 레이블이 정답과 일치하는 개수를 저장하기 위한 변수
    n_data = 0 # 전체 데이터의 개수를 저장하기 위한 변수
    for inputs, targets in test_loader:
        # [START CODE]
        y_pred = model(inputs)
        y_pred_class = (y_pred > 0.5).float()
        correct += (targets == y_pred_class).sum().item()
        n_data += targets.size(0)
        # [END CODE]
    print(f"accuracy: {correct/n_data}")
```

평가 코드를 올바르게 작성하였다면, 아래와 같이 평가용 데이터에 대한 정확도가 출력됩니다.


```python
test(model, test_loader)
```

    accuracy: 0.81
    

## **Scikit-Learn 결과와의 비교**

이제, 머신러닝을 통해 계산한 로지스틱 회귀 모형과 통계적인 방법으로 적합한 로지스틱 회귀 모형을 비교하고자 합니다.

scikit-learn 라이브러리에서 `LogisticRegression`클래스를 이용하여 통계적인 방법으로 로지스틱 회귀 모형을 적합할 수 있습니다.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

아래와 같이 로지스틱 회귀 모형 객체를 생성하고, 학습 데이터를 바탕으로 적합할 수 있습니다.


```python
# 로지스틱 회귀 모델 초기화 및 학습
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(train_data.X.cpu().numpy(), train_data.y.cpu().numpy())
```

    /usr/local/lib/python3.10/dist-packages/sklearn/utils/validation.py:1183: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>



이제 평가용 데이터를 바탕으로 성능을 확인해보면, 두 방법 모두 비슷하게 적합됨을 확인할 수 있습니다.


```python
y_pred = logistic_regression_model.predict(test_data.X.cpu().numpy())
accuracy = accuracy_score(test_data.y.cpu().numpy(), y_pred)
print(f"Test Accuracy: {accuracy}")
```

    Test Accuracy: 0.79
    

## 콘텐츠 라이선스

<hr style="height:5px;border:none;color:#5F71F7;background-color:#5F71F7">

<font color='red'><b>WARNING</font> : 본 교육 콘텐츠의 지식재산권은 재단법인 네이버커넥트에 귀속됩니다. 본 콘텐츠를 어떠한 경로로든 외부로 유출 및 수정하는 행위를 엄격히 금합니다. 다만, 비영리적 교육 및 연구활동에 한정되어 사용할 수 있으나 재단의 허락을 받아야 합니다. 이를 위반하는 경우, 관련 법률에 따라 책임을 질 수 있습니다. </b>
