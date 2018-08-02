
# Multiple Linear Regression

Similar to the Simple Linear Regression but with multiple variables

<img src="img/formula.png" width="300" height="100">

__Assumptions of a Linear Regression:__
- Linearity
- Homoscedasticity
- Multivariate normality
- Independence of errors
- Lack of multicollinearity

More information [here](https://www.youtube.com/watch?v=but2n_zBPpU)

Before starting to build a Linear Regression model you need to check if these assumptions are true

First let's import our dataset


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>R&amp;D Spend</th>
      <th>Administration</th>
      <th>Marketing Spend</th>
      <th>State</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>165349.20</td>
      <td>136897.80</td>
      <td>471784.10</td>
      <td>New York</td>
      <td>192261.83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>162597.70</td>
      <td>151377.59</td>
      <td>443898.53</td>
      <td>California</td>
      <td>191792.06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>153441.51</td>
      <td>101145.55</td>
      <td>407934.54</td>
      <td>Florida</td>
      <td>191050.39</td>
    </tr>
    <tr>
      <th>3</th>
      <td>144372.41</td>
      <td>118671.85</td>
      <td>383199.62</td>
      <td>New York</td>
      <td>182901.99</td>
    </tr>
    <tr>
      <th>4</th>
      <td>142107.34</td>
      <td>91391.77</td>
      <td>366168.42</td>
      <td>Florida</td>
      <td>166187.94</td>
    </tr>
  </tbody>
</table>
</div>



In our dataset, we see that every columns contain numerical values except State which is a __categorical data__

So we can't put this feature un our multiple linear regression equation.

So we need to create __dummy variables__.

<img src="img/dummy_variables.png" width="600" height="300">


So in our equation we will use the New York column instead of the State column (for California the value if 1 if a value in New York column is 0, so we don't need to use this column)

<img src="img/dummy_variables_2.png" width="600" height="300">

#### __Dummy Variable Trap__

If you are adding both Dummy (New York -> D1 & California -> D2) variables you will duplicate a variable (Because D2 = 1 - D1).  

The phenomenom when one or several independent variables in a linear regression predict another is called multicollinearity.

Because of that the model can't distinguish the effect of D1 from the effect of D2 and therefor it is not going to work proprely that is called the __Dummy Variable Trap__ (So we can't have the constant and Dummy variables at the same time).

So one rule , __always omit one variable__ in the equation.

<img src="img/dummy_variables_3.png" width="600" height="300">


Before going further, you need to know what is a __p_value__ : P value is a statistical measure that helps scientists determine whether or not their hypotheses are correct

For detailed information, have a look [here](https://www.wikihow.com/Calculate-P-Value)

#### __Building a model__

Have a look to the [Step by step guide](https://www.superdatascience.com/wp-content/uploads/2017/02/Step-by-step-Blueprints-For-Building-Models.pdf)

## Example

Let's begin : We want to predict the profit __(Dependent variable vector Y )__ using our spend columns __(Independent variables matric X )__

First create the matrix and vector


```python
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

X,y
```




    (array([[165349.2, 136897.8, 471784.1, 'New York'],
            [162597.7, 151377.59, 443898.53, 'California'],
            [153441.51, 101145.55, 407934.54, 'Florida'],
            [144372.41, 118671.85, 383199.62, 'New York'],
            [142107.34, 91391.77, 366168.42, 'Florida'],
            [131876.9, 99814.71, 362861.36, 'New York'],
            [134615.46, 147198.87, 127716.82, 'California'],
            [130298.13, 145530.06, 323876.68, 'Florida'],
            [120542.52, 148718.95, 311613.29, 'New York'],
            [123334.88, 108679.17, 304981.62, 'California'],
            [101913.08, 110594.11, 229160.95, 'Florida'],
            [100671.96, 91790.61, 249744.55, 'California'],
            [93863.75, 127320.38, 249839.44, 'Florida'],
            [91992.39, 135495.07, 252664.93, 'California'],
            [119943.24, 156547.42, 256512.92, 'Florida'],
            [114523.61, 122616.84, 261776.23, 'New York'],
            [78013.11, 121597.55, 264346.06, 'California'],
            [94657.16, 145077.58, 282574.31, 'New York'],
            [91749.16, 114175.79, 294919.57, 'Florida'],
            [86419.7, 153514.11, 0.0, 'New York'],
            [76253.86, 113867.3, 298664.47, 'California'],
            [78389.47, 153773.43, 299737.29, 'New York'],
            [73994.56, 122782.75, 303319.26, 'Florida'],
            [67532.53, 105751.03, 304768.73, 'Florida'],
            [77044.01, 99281.34, 140574.81, 'New York'],
            [64664.71, 139553.16, 137962.62, 'California'],
            [75328.87, 144135.98, 134050.07, 'Florida'],
            [72107.6, 127864.55, 353183.81, 'New York'],
            [66051.52, 182645.56, 118148.2, 'Florida'],
            [65605.48, 153032.06, 107138.38, 'New York'],
            [61994.48, 115641.28, 91131.24, 'Florida'],
            [61136.38, 152701.92, 88218.23, 'New York'],
            [63408.86, 129219.61, 46085.25, 'California'],
            [55493.95, 103057.49, 214634.81, 'Florida'],
            [46426.07, 157693.92, 210797.67, 'California'],
            [46014.02, 85047.44, 205517.64, 'New York'],
            [28663.76, 127056.21, 201126.82, 'Florida'],
            [44069.95, 51283.14, 197029.42, 'California'],
            [20229.59, 65947.93, 185265.1, 'New York'],
            [38558.51, 82982.09, 174999.3, 'California'],
            [28754.33, 118546.05, 172795.67, 'California'],
            [27892.92, 84710.77, 164470.71, 'Florida'],
            [23640.93, 96189.63, 148001.11, 'California'],
            [15505.73, 127382.3, 35534.17, 'New York'],
            [22177.74, 154806.14, 28334.72, 'California'],
            [1000.23, 124153.04, 1903.93, 'New York'],
            [1315.46, 115816.21, 297114.46, 'Florida'],
            [0.0, 135426.92, 0.0, 'California'],
            [542.05, 51743.15, 0.0, 'New York'],
            [0.0, 116983.8, 45173.06, 'California']], dtype=object),
     array([ 192261.83,  191792.06,  191050.39,  182901.99,  166187.94,
             156991.12,  156122.51,  155752.6 ,  152211.77,  149759.96,
             146121.95,  144259.4 ,  141585.52,  134307.35,  132602.65,
             129917.04,  126992.93,  125370.37,  124266.9 ,  122776.86,
             118474.03,  111313.02,  110352.25,  108733.99,  108552.04,
             107404.34,  105733.54,  105008.31,  103282.38,  101004.64,
              99937.59,   97483.56,   97427.84,   96778.92,   96712.8 ,
              96479.51,   90708.19,   89949.14,   81229.06,   81005.76,
              78239.91,   77798.83,   71498.49,   69758.98,   65200.33,
              64926.08,   49490.75,   42559.73,   35673.41,   14681.4 ]))



Then We have to encode categorical data (State).

__WARNING__: Don't forget to remove one Dummy variable column. For our column ou are going to remove de first column


```python
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]


# Formatting numpy to only show integer and not float
np.set_printoptions(formatter={'float': '{: 0.0f}'.format})

X
```




    array([[ 0,  1,  165349,  136898,  471784],
           [ 0,  0,  162598,  151378,  443899],
           [ 1,  0,  153442,  101146,  407935],
           [ 0,  1,  144372,  118672,  383200],
           [ 1,  0,  142107,  91392,  366168],
           [ 0,  1,  131877,  99815,  362861],
           [ 0,  0,  134615,  147199,  127717],
           [ 1,  0,  130298,  145530,  323877],
           [ 0,  1,  120543,  148719,  311613],
           [ 0,  0,  123335,  108679,  304982],
           [ 1,  0,  101913,  110594,  229161],
           [ 0,  0,  100672,  91791,  249745],
           [ 1,  0,  93864,  127320,  249839],
           [ 0,  0,  91992,  135495,  252665],
           [ 1,  0,  119943,  156547,  256513],
           [ 0,  1,  114524,  122617,  261776],
           [ 0,  0,  78013,  121598,  264346],
           [ 0,  1,  94657,  145078,  282574],
           [ 1,  0,  91749,  114176,  294920],
           [ 0,  1,  86420,  153514,  0],
           [ 0,  0,  76254,  113867,  298664],
           [ 0,  1,  78389,  153773,  299737],
           [ 1,  0,  73995,  122783,  303319],
           [ 1,  0,  67533,  105751,  304769],
           [ 0,  1,  77044,  99281,  140575],
           [ 0,  0,  64665,  139553,  137963],
           [ 1,  0,  75329,  144136,  134050],
           [ 0,  1,  72108,  127865,  353184],
           [ 1,  0,  66052,  182646,  118148],
           [ 0,  1,  65605,  153032,  107138],
           [ 1,  0,  61994,  115641,  91131],
           [ 0,  1,  61136,  152702,  88218],
           [ 0,  0,  63409,  129220,  46085],
           [ 1,  0,  55494,  103057,  214635],
           [ 0,  0,  46426,  157694,  210798],
           [ 0,  1,  46014,  85047,  205518],
           [ 1,  0,  28664,  127056,  201127],
           [ 0,  0,  44070,  51283,  197029],
           [ 0,  1,  20230,  65948,  185265],
           [ 0,  0,  38559,  82982,  174999],
           [ 0,  0,  28754,  118546,  172796],
           [ 1,  0,  27893,  84711,  164471],
           [ 0,  0,  23641,  96190,  148001],
           [ 0,  1,  15506,  127382,  35534],
           [ 0,  0,  22178,  154806,  28335],
           [ 0,  1,  1000,  124153,  1904],
           [ 1,  0,  1315,  115816,  297114],
           [ 0,  0,  0,  135427,  0],
           [ 0,  1,  542,  51743,  0],
           [ 0,  0,  0,  116984,  45173]])



Now we are able to split our data into Train and test sets


```python
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

    c:\users\yanni iyeze\appdata\local\programs\python\python36-32\lib\site-packages\sklearn\cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)


We are now able to fit our Multiple Linear Regression model to the training set and predict using the test set that will create the vector of predictions.


```python
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
```


```python
plt.plot(y_test, label = 'real profit (y_test)')
plt.plot(y_pred, color='red', label='predicted profit')
plt.legend()
plt.show()
```


![png](img/output_10_0.png)


## Example Backward Elimination

To have better results by removing non-statistical variables.

We are going to use the statsmodels library
- 1: Choosing an "acceptable" value for p-value
- 2: Create an optimal matrix with the features you want use
- 3: Use the regressor to calculate p-values for your optimal matrix

If you have a feature with a p-value greater than you acceptable value remove its index from you opt matrix (2:) and start again

__WARNING__ : Be careful with the index of features when you are removing features.

You are done when your regressor have features with p-values lower than the acceptance value.


```python
# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm

# Step 1 - SL = 0.05 - Invert np.ones and X to have it at the beginning
X = np.append(arr = np.ones((50,1)).astype(int), values= X, axis=1)

# Step 2 - Choosing features

# Optimal Matrix of features
X_opt = X[:, [0, 1, 2, 3, 4, 5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

# Step 3
regressor_OLS.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.951</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.945</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   169.9</td>
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 14 May 2018</td> <th>  Prob (F-statistic):</th> <td>1.34e-27</td>
</tr>
<tr>
  <th>Time:</th>                 <td>22:12:47</td>     <th>  Log-Likelihood:    </th> <td> -525.38</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    50</td>      <th>  AIC:               </th> <td>   1063.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    44</td>      <th>  BIC:               </th> <td>   1074.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     5</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td> 5.013e+04</td> <td> 6884.820</td> <td>    7.281</td> <td> 0.000</td> <td> 3.62e+04</td> <td>  6.4e+04</td>
</tr>
<tr>
  <th>x1</th>    <td>  198.7888</td> <td> 3371.007</td> <td>    0.059</td> <td> 0.953</td> <td>-6595.030</td> <td> 6992.607</td>
</tr>
<tr>
  <th>x2</th>    <td>  -41.8870</td> <td> 3256.039</td> <td>   -0.013</td> <td> 0.990</td> <td>-6604.003</td> <td> 6520.229</td>
</tr>
<tr>
  <th>x3</th>    <td>    0.8060</td> <td>    0.046</td> <td>   17.369</td> <td> 0.000</td> <td>    0.712</td> <td>    0.900</td>
</tr>
<tr>
  <th>x4</th>    <td>   -0.0270</td> <td>    0.052</td> <td>   -0.517</td> <td> 0.608</td> <td>   -0.132</td> <td>    0.078</td>
</tr>
<tr>
  <th>x5</th>    <td>    0.0270</td> <td>    0.017</td> <td>    1.574</td> <td> 0.123</td> <td>   -0.008</td> <td>    0.062</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>14.782</td> <th>  Durbin-Watson:     </th> <td>   1.283</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.001</td> <th>  Jarque-Bera (JB):  </th> <td>  21.266</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.948</td> <th>  Prob(JB):          </th> <td>2.41e-05</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.572</td> <th>  Cond. No.          </th> <td>1.45e+06</td>
</tr>
</table>




```python
#Removing index with big p-value -> 2
#Optimal Matrix of features
X_opt_2 = X[:, [0, 1, 3, 4, 5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt_2).fit()

regressor_OLS.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.951</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.946</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   217.2</td>
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 14 May 2018</td> <th>  Prob (F-statistic):</th> <td>8.49e-29</td>
</tr>
<tr>
  <th>Time:</th>                 <td>22:12:47</td>     <th>  Log-Likelihood:    </th> <td> -525.38</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    50</td>      <th>  AIC:               </th> <td>   1061.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    45</td>      <th>  BIC:               </th> <td>   1070.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td> 5.011e+04</td> <td> 6647.870</td> <td>    7.537</td> <td> 0.000</td> <td> 3.67e+04</td> <td> 6.35e+04</td>
</tr>
<tr>
  <th>x1</th>    <td>  220.1585</td> <td> 2900.536</td> <td>    0.076</td> <td> 0.940</td> <td>-5621.821</td> <td> 6062.138</td>
</tr>
<tr>
  <th>x2</th>    <td>    0.8060</td> <td>    0.046</td> <td>   17.606</td> <td> 0.000</td> <td>    0.714</td> <td>    0.898</td>
</tr>
<tr>
  <th>x3</th>    <td>   -0.0270</td> <td>    0.052</td> <td>   -0.523</td> <td> 0.604</td> <td>   -0.131</td> <td>    0.077</td>
</tr>
<tr>
  <th>x4</th>    <td>    0.0270</td> <td>    0.017</td> <td>    1.592</td> <td> 0.118</td> <td>   -0.007</td> <td>    0.061</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>14.758</td> <th>  Durbin-Watson:     </th> <td>   1.282</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.001</td> <th>  Jarque-Bera (JB):  </th> <td>  21.172</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.948</td> <th>  Prob(JB):          </th> <td>2.53e-05</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.563</td> <th>  Cond. No.          </th> <td>1.40e+06</td>
</tr>
</table>




```python
#Removing index with big p-value -> 1
#Optimal Matrix of features
X_opt_3 = X[:, [0, 3, 4, 5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt_3).fit()

regressor_OLS.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.951</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.948</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   296.0</td>
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 14 May 2018</td> <th>  Prob (F-statistic):</th> <td>4.53e-30</td>
</tr>
<tr>
  <th>Time:</th>                 <td>22:12:47</td>     <th>  Log-Likelihood:    </th> <td> -525.39</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    50</td>      <th>  AIC:               </th> <td>   1059.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    46</td>      <th>  BIC:               </th> <td>   1066.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td> 5.012e+04</td> <td> 6572.353</td> <td>    7.626</td> <td> 0.000</td> <td> 3.69e+04</td> <td> 6.34e+04</td>
</tr>
<tr>
  <th>x1</th>    <td>    0.8057</td> <td>    0.045</td> <td>   17.846</td> <td> 0.000</td> <td>    0.715</td> <td>    0.897</td>
</tr>
<tr>
  <th>x2</th>    <td>   -0.0268</td> <td>    0.051</td> <td>   -0.526</td> <td> 0.602</td> <td>   -0.130</td> <td>    0.076</td>
</tr>
<tr>
  <th>x3</th>    <td>    0.0272</td> <td>    0.016</td> <td>    1.655</td> <td> 0.105</td> <td>   -0.006</td> <td>    0.060</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>14.838</td> <th>  Durbin-Watson:     </th> <td>   1.282</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.001</td> <th>  Jarque-Bera (JB):  </th> <td>  21.442</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.949</td> <th>  Prob(JB):          </th> <td>2.21e-05</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.586</td> <th>  Cond. No.          </th> <td>1.40e+06</td>
</tr>
</table>




```python
#Removing index with big p-value -> 4
#Optimal Matrix of features
X_opt_4 = X[:, [0, 3, 5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt_4).fit()

regressor_OLS.summary()

#Removing index with big p-value
#Optimal Matrix of features
X_opt_5 = X[:, [0, 3]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt_5).fit()

regressor_OLS.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.947</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.945</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   849.8</td>
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 14 May 2018</td> <th>  Prob (F-statistic):</th> <td>3.50e-32</td>
</tr>
<tr>
  <th>Time:</th>                 <td>22:12:47</td>     <th>  Log-Likelihood:    </th> <td> -527.44</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    50</td>      <th>  AIC:               </th> <td>   1059.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    48</td>      <th>  BIC:               </th> <td>   1063.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td> 4.903e+04</td> <td> 2537.897</td> <td>   19.320</td> <td> 0.000</td> <td> 4.39e+04</td> <td> 5.41e+04</td>
</tr>
<tr>
  <th>x1</th>    <td>    0.8543</td> <td>    0.029</td> <td>   29.151</td> <td> 0.000</td> <td>    0.795</td> <td>    0.913</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>13.727</td> <th>  Durbin-Watson:     </th> <td>   1.116</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.001</td> <th>  Jarque-Bera (JB):  </th> <td>  18.536</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.911</td> <th>  Prob(JB):          </th> <td>9.44e-05</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.361</td> <th>  Cond. No.          </th> <td>1.65e+05</td>
</tr>
</table>



## Automatic Backward Elimination
### Backward Elimination with p-values only


```python
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
```

### Backward Elimination with p-values and Adjusted R Squared


```python

import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.950
    Model:                            OLS   Adj. R-squared:                  0.948
    Method:                 Least Squares   F-statistic:                     450.8
    Date:                Mon, 14 May 2018   Prob (F-statistic):           2.16e-31
    Time:                        22:12:47   Log-Likelihood:                -525.54
    No. Observations:                  50   AIC:                             1057.
    Df Residuals:                      47   BIC:                             1063.
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const       4.698e+04   2689.933     17.464      0.000    4.16e+04    5.24e+04
    x1             0.7966      0.041     19.266      0.000       0.713       0.880
    x2             0.0299      0.016      1.927      0.060      -0.001       0.061
    ==============================================================================
    Omnibus:                       14.677   Durbin-Watson:                   1.257
    Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.161
    Skew:                          -0.939   Prob(JB):                     2.54e-05
    Kurtosis:                       5.575   Cond. No.                     5.32e+05
    ==============================================================================

    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 5.32e+05. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

plt.plot(y_test, label = 'real profit (y_test)')
plt.plot(y_pred, color='red', label='predicted profit')
plt.legend()
plt.show()
```


![png](img/output_20_0.png)
