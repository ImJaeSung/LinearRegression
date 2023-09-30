# House Prices Prediction task using Neural Network LinearRegression 

##House Prices Prediction
=======================
### 0. Data
- With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.
- Data fields : https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
#### Shape of data
- Shape of Train data : 1460 x 81
- Shape of Test data : 1459 x 80
- The number of numeric variable : 37 (including SalePrice)
- The number of category variable : 43

#### Missing Value
- LotFrontage     259
Alley           1369
\vdots$
PoolQC           1453
Fence            1179
MiscFeature      1406
=>Total : 6965
#### Distribution
- numeric
![numeric.png](https://www.dropbox.com/scl/fi/h7vyynnxnimcl6mycy9og/numeric.png?rlkey=7zapxuvicxsrrorokmsebc4na&dl=0&raw=1)
- categorical
- ![category.png](https://www.dropbox.com/scl/fi/ztp9upk525e491x7xlqiq/category.png?rlkey=qrk9800h96ih318fq5u0mtsvp&dl=0&raw=1)
 
### 1. Data-Preprocessing
#### Drop columns 
- ID
#### Missing Value 
- numeric feature fill zero 
- categorical feature : get_dummies option
#### Scaling
- target : skew -> log transformation
![target.png](https://www.dropbox.com/scl/fi/p87eqvson3bclew53n4z1/target.png?rlkey=jc10y9alvozpufi4yqgoawu5b&dl=0&raw=1)
- numeric columns scaling (sklearn.standardscaler)
### outlier detection
![boxplot.png](https://www.dropbox.com/scl/fi/3iugsjgfhzjc5tc6ndkja/boxplot.png?rlkey=246bubzaqh18nxqrvdopdru2n&dl=0&raw=1)


### 2. Modeling
- the number of hidden layers = 4
- activation function = ReLU
- dropout with prob = 0.2
![Alt Text](https://thebook.io/img/080263/135_1.jpg)

#### hidden layer1
- $h1 : \mathbb{R}^{288}$$ $$\mapsto$$ $$\mathbb{R}^{512}$
- $H^{(1)} = ReLU(XW^{(1)}+B^{(1)})$
- $W^{(1)} \in {R}^{288\times512}$
- $B^{(1)})\in {R}^{512}$
- $ReLU = Max(0, x)$

```
self.hidden_layer1 = nn.Sequential(
            nn.Linear(288, 512),
            nn.ReLU(inplace = True)
)
```

#### hidden layer2
- $h2 : \mathbb{R}^{512} \mapsto$$ $$\mathbb{R}^{1024}$
- $H^{(2)} = ReLU(H^{(1)}W^{(2)}+B^{(2)})$
- $W^{(2)} \in {R}^{512\times1024}$
- $B^{(2)})\in {R}^{1024}$
```
self.hidden_layer2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace = True)
)
```
#### hidden layer3
- $h3 : \mathbb{R}^{1024} \mapsto$$ $$\mathbb{R}^{1024}$
- $H^{(3)} = ReLU(H^{(2)}W^{(3)}+B^{(3)})$
- $W^{(3)} \in {R}^{1024\times1024}$
- $B^{(3)})\in {R}^{1024}$

```
self.hidden_layer3 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace = True)
)
```

#### hidden layer4
- $h4 : \mathbb{R}^{1024}\mapsto\mathbb{R}^{512}$
- $H^{(4)} = ReLU(H^{(3)}W^{(4)}+B^{(4)})$
- $W^{(4)} \in {R}^{1024\times512}$
- $B^{(4)})\in {R}^{512}$

```
self.hidden_layer4 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace = True)
)
```

#### output layer
- $o : \mathbb{R}^{1024} \mapsto \mathbb{R}^{1}$
- $O = ReLU(H^{(4)}W^{(out)}+B^{(out)})$
- $W^{(out)} \in {R}^{512\times1}$
- $B^{(out)}\in {R}^{1}$

```
self.output_layer = nn.Sequential(
            nn.Linear(512, 1),
            nn.ReLU(inplace = True)
)
```
#### Activation function
- The ReLU function is defined as: For x > 0 the output is x, 
- i.e. f(x) = max(0,x)
 
#### Dropout
![Alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbv1pFt%2Fbtq1ZMTLNSD%2FfpYC8x7Q5kSCNKdyXk8JfK%2Fimg.png)

### 3. Training
- epoch = 3000
- learning rate = 0.001
- optimizer = Adam
- Loss = MSE
- learning rate decay : ReduceLROnPlateau
- Early stopping -> 1204 epoch

#### K-Fold Cross-Validation
- K = 5
#### Activation function
- So for the derivative f '(x) it's actually:
- if x < 0, output is 0. if x > 0, output is 1.
- The derivative f '(0) is not defined. So it's usually set to 0 or you modify the activation function to be f(x) = max(e,x) for a small e

#### Optimizer
- Adam
- Momentum + RMSProp
$m_{n} = \beta_{1}m_{n-1}+(1- \beta_{1}) \bigtriangledownf(x_{n})$
$v_{n}$$ = \beta_{2}v_{n-1}+(1- \beta_{1}) \bigtriangledownf(x_{n}) \bigodot \bigtriangledownf(x_{n})$
$\hat{m_{n}} =  \frac{m_{n}}{1- \beta_{1}^(n+1)}$
$\hat{v_{n}} =  \frac{v_{n}}{1- \beta_{2}^(n+1)}$
$x_{n+1} = x_{n} - learningrate \frac{1}{\sqrt(\hat{v_{n}})}\bigodot \hat{m_{n}}$

#### Learning rate decay
```
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', min_lr = 0.00001, 
                                                       threshold = 0.01, factor = 0.1, patience = 2)
```
#### Early stopping rule
- patience : 3

#### Loss function
- L = $$\frac{1}{2}(target_{1}-O_{1})^2$$ +$$\frac{1}{2}(target_{2}-O_{2})^2$$ + $$\cdots$$ + $$\frac{1}{2}(target_{1168}-O_{1168})^2$$
- $$\frac{\partial L}{\partial O_{1}}$$ = 2 $$\times\frac{1}{2}(target_{1}-O_{1})^1\times(-1)+0$$
- $$\frac{\partial L}{\partial O_{2}}$$ = 2 $$\times\frac{1}{2}(target_{2}-O_{2})^1\times(-1)+0$$
- $$\vdots$$
- $$\frac{\partial L}{\partial O_{1168}}$$ = 2 $$\times\frac{1}{2}(target_{1168}-O_{1168})^1\times(-1)+0$$

#### Output layer
- $$\frac{\partial L}{\partial O}$$ = $$\begin{bmatrix}
\frac{\partial L}{\partial O_{1}}\\
\frac{\partial L}{\partial O_{2}}\\
\vdots \\
\frac{\partial L}{\partial O_{1168}}
\end{bmatrix}$$ 

- $$\frac{\partial L}{\partial H_{4}}$$ = $$\frac{\partial L}{\partial O}$$ $$W_{out}^T$$ =$$\begin{bmatrix}
\frac{\partial L}{\partial O_{1}}\\
\frac{\partial L}{\partial O_{2}}\\
\vdots \\
\frac{\partial L}{\partial O_{1168}}
\end{bmatrix}$$$$\begin{bmatrix}
w_{1,1} \\
w_{2,1} \\
\vdots \\
w_{512,1} \\
\end{bmatrix}^T$$

- $$\frac{\partial L}{\partial W_{out}}$$ = $$H_{4}^T$$ $$\frac{\partial L}{\partial O}$$ = $$\begin{bmatrix} 
h_{1,1} & h_{1,2} & \cdots & h_{1,512} \\
h_{2,1} & h_{2,2} & \cdots & h_{1,512} \\
h_{3,1} & h_{3,2} & \cdots & h_{3,512} \\
\vdots & \vdots & \ddots & \vdots\\
h_{1168,1} & h_{1168,2} & \cdots & h_{1168,512} \\
\end{bmatrix}^T$$ $$\begin{bmatrix}
\frac{\partial L}{\partial O_{1}}\\
\frac{\partial L}{\partial O_{2}}\\
\vdots \\
\frac{\partial L}{\partial O_{1168}}
\end{bmatrix}$$

- $$\frac{\partial L}{\partial B_{out}}$$ = $$\sum_{i =1}^{1168}$$ $$\frac{\partial L}{\partial O_{i}}$$

#### hidden layer4
- $$\frac{\partial L}{\partial H_{4}}$$ = $$\frac{\partial L}{\partial O}$$ $$W_{out}^T$$
- $$\frac{\partial L}{\partial H_{3}}$$ = $$\frac{\partial L}{\partial H_{4}}$$ $$W_{4}^T$$ = $$\frac{\partial L}{\partial O}$$ $$W_{out}^T$$ $$W_{4}^T$$
- $$\frac{\partial L}{\partial W_{4}}$$ = $$H_{3}^T$$ $$\frac{\partial L}{\partial H_{4}}$$ = $$H_{3}^T$$ $$\frac{\partial L}{\partial O}$$ $$W_{out}^T$$
- $$\frac{\partial L}{\partial B_{4}}$$ = $$\frac{\partial L}{\partial H_{4}}$$

#### hidden layer3
- $$\frac{\partial L}{\partial H_{3}}$$
- $$\frac{\partial L}{\partial H_{2}}$$ = $$\frac{\partial L}{\partial H_{3}}$$ $$W_{3}^T$$
- $$\frac{\partial L}{\partial W_{3}}$$ = $$H_{2}^T$$ $$\frac{\partial L}{\partial H_{3}}$$
- $$\frac{\partial L}{\partial B_{out}}$$ = $$\frac{\partial L}{\partial H_{3}}$$

#### hidden layer2
- $$\frac{\partial L}{\partial H_{2}}$$
- $$\frac{\partial L}{\partial H_{1}}$$ = $$\frac{\partial L}{\partial H_{2}}$$ $$W_{2}^T$$
- $$\frac{\partial L}{\partial W_{2}}$$ = $$H_{1}^T$$ $$\frac{\partial L}{\partial H_{2}}$$
- $$\frac{\partial L}{\partial B_{2}}$$ = $$\frac{\partial L}{\partial H_{2}}$$

#### hidden layer1
- $$\frac{\partial L}{\partial H_{1}}$$
- $$\frac{\partial L}{\partial X}$$ = $$\frac{\partial L}{\partial H_{1}}$$ $$W_{1}^T$$
- $$\frac{\partial L}{\partial W_{1}}$$ = $$X^T$$ $$\frac{\partial L}{\partial H_{1}}$$
- $$\frac{\partial L}{\partial B_{1}}$$ = $$\frac{\partial L}{\partial H_{3}}$$

#### Mean of Validation score
![valid score.png](https://www.dropbox.com/scl/fi/4u1d45ms9ohkq3rucz7lg/valid-score.png?rlkey=18atsecvk7w3ydk3e11730b4i&dl=0&raw=1)

### 4. Prediction
- Average of best model predictions for each fold
- validation set score : 0.0146
- report score from kaggle : 0.13443
- Evaluation metric : RMSE


##### Regression using neural network
- Deep learning requires calculating complex, high-dimensional composite functions created through a neural network. 
- It will probably be very difficult.
- Also, even if calculation is possible, it will require a relatively large amount of calculation. 
- Gradient descent is a method designed to find the minimum value of high-dimensional functions handled in deep learning.
