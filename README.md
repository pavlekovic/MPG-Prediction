# MPG Prediction

```python
# Importing libraries and packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # for visualization purposes
import seaborn as sns # for visualization purposes
import statsmodels as sm # for intercept
from statsmodels.stats.outliers_influence import variance_inflation_factor  # we will need vif for checking multicollinearity
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV  # for data splitting and parameter tuning
from sklearn.pipeline import Pipeline # used when scaling data to avoid data leaks (used with LR)
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # linear regression models
from sklearn.preprocessing import StandardScaler # for creating scaled data for linear regression
from sklearn.metrics import r2_score,mean_squared_error   # for scoring the model and calculating the mean squared error
from sklearn.tree import DecisionTreeRegressor  # decision tree regression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor  # ensemble methods for regression
from sklearn.model_selection import cross_val_score # for cross-validation
from xgboost import XGBRegressor # XGB Regressor

import warnings
warnings.filterwarnings('ignore')
```

## Data import


```python
# Load data from the repository
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', sep="\\s+", header=None)

# Rename columns according to 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.names'
data.columns = ['mpg', 'cylinders', 'displacement', 'horsepower' , 'weight', 'acceleration', 'model year', 'origin', 'car']

# Initial glance over data
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model year</th>
      <th>origin</th>
      <th>car</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504.0</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693.0</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436.0</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150.0</td>
      <td>3433.0</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140.0</td>
      <td>3449.0</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
    </tr>
  </tbody>
</table>
</div>



## Data cleaning

### Checking for variable types and missing values


```python
# Information about variable types and null values
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 398 entries, 0 to 397
    Data columns (total 9 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   mpg           398 non-null    float64
     1   cylinders     398 non-null    int64  
     2   displacement  398 non-null    float64
     3   horsepower    398 non-null    object 
     4   weight        398 non-null    float64
     5   acceleration  398 non-null    float64
     6   model year    398 non-null    int64  
     7   origin        398 non-null    int64  
     8   car           398 non-null    object 
    dtypes: float64(4), int64(3), object(2)
    memory usage: 28.1+ KB



```python
# There must be a reason why column horsepower is object when in reality should be float64 (number)
# so we use .unique to display unique values and check for oddities
data.horsepower.unique()
```




    array(['130.0', '165.0', '150.0', '140.0', '198.0', '220.0', '215.0',
           '225.0', '190.0', '170.0', '160.0', '95.00', '97.00', '85.00',
           '88.00', '46.00', '87.00', '90.00', '113.0', '200.0', '210.0',
           '193.0', '?', '100.0', '105.0', '175.0', '153.0', '180.0', '110.0',
           '72.00', '86.00', '70.00', '76.00', '65.00', '69.00', '60.00',
           '80.00', '54.00', '208.0', '155.0', '112.0', '92.00', '145.0',
           '137.0', '158.0', '167.0', '94.00', '107.0', '230.0', '49.00',
           '75.00', '91.00', '122.0', '67.00', '83.00', '78.00', '52.00',
           '61.00', '93.00', '148.0', '129.0', '96.00', '71.00', '98.00',
           '115.0', '53.00', '81.00', '79.00', '120.0', '152.0', '102.0',
           '108.0', '68.00', '58.00', '149.0', '89.00', '63.00', '48.00',
           '66.00', '139.0', '103.0', '125.0', '133.0', '138.0', '135.0',
           '142.0', '77.00', '62.00', '132.0', '84.00', '64.00', '74.00',
           '116.0', '82.00'], dtype=object)




```python
# The odd value is "?" which represents a null value and we can substitute it for a mean horsepower
data.horsepower = data.horsepower.str.replace('?','NaN').astype(float) # this replaces '?' with nan
data.horsepower.fillna(data.horsepower.mean(), inplace=True) # this replaces all nan values with mean

# to simplify, make the column int type
data.horsepower = data.horsepower.astype(int)
```

### Checking for duplicate values


```python
# Information about duplicate values
print("Duplicate rows:", data.duplicated().sum())
```

    Duplicate rows: 0


### Checking for zero values


```python
print((data == 0).sum())
```

    mpg             0
    cylinders       0
    displacement    0
    horsepower      0
    weight          0
    acceleration    0
    model year      0
    origin          0
    car             0
    dtype: int64


### Checking for outliers and inconsistencies


```python
display(data.describe())
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model year</th>
      <th>origin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>398.000000</td>
      <td>398.000000</td>
      <td>398.000000</td>
      <td>398.000000</td>
      <td>398.000000</td>
      <td>398.000000</td>
      <td>398.000000</td>
      <td>398.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>23.514573</td>
      <td>5.454774</td>
      <td>193.425879</td>
      <td>104.462312</td>
      <td>2970.424623</td>
      <td>15.568090</td>
      <td>76.010050</td>
      <td>1.572864</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.815984</td>
      <td>1.701004</td>
      <td>104.269838</td>
      <td>38.199230</td>
      <td>846.841774</td>
      <td>2.757689</td>
      <td>3.697627</td>
      <td>0.802055</td>
    </tr>
    <tr>
      <th>min</th>
      <td>9.000000</td>
      <td>3.000000</td>
      <td>68.000000</td>
      <td>46.000000</td>
      <td>1613.000000</td>
      <td>8.000000</td>
      <td>70.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>17.500000</td>
      <td>4.000000</td>
      <td>104.250000</td>
      <td>76.000000</td>
      <td>2223.750000</td>
      <td>13.825000</td>
      <td>73.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>23.000000</td>
      <td>4.000000</td>
      <td>148.500000</td>
      <td>95.000000</td>
      <td>2803.500000</td>
      <td>15.500000</td>
      <td>76.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>29.000000</td>
      <td>8.000000</td>
      <td>262.000000</td>
      <td>125.000000</td>
      <td>3608.000000</td>
      <td>17.175000</td>
      <td>79.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>46.600000</td>
      <td>8.000000</td>
      <td>455.000000</td>
      <td>230.000000</td>
      <td>5140.000000</td>
      <td>24.800000</td>
      <td>82.000000</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
</div>


## Visual inspection


```python
# First drop variable car
data1 = data.drop(['car'], axis=1)

# Visually inspect variables
data1.hist(figsize=(12, 8), color='#298c8c')
plt.show()
```


    
![png](mpg_files/output_15_0.png)
    



```python
# Explore relationships between variables visually (correlation matrix)
plt.figure(figsize=(10,6))
sns.heatmap(data1.corr(),cmap=plt.cm.Blues,annot=True) # use seaborn
plt.show()
```


    
![png](mpg_files/output_16_0.png)
    



```python
# Cylinders, Displacement, Horsepower and Weight are all highly (negatively) correlated so we have multicollinearity issue
# Calculate variance inflation factor (VIF) for each variable to determine which one to keep
X1 = sm.tools.add_constant(data1) # add intercept to the dataset 
# Calculate VIF for the dataset
vif1 = pd.Series([variance_inflation_factor(X1.values,i)
                     for i in range(X1.shape[1])], index=X1.columns)
vif1
```




    const           780.811358
    mpg               5.583594
    cylinders        10.742336
    displacement     22.159830
    horsepower        9.056781
    weight           13.468785
    acceleration      2.515908
    model year        1.954947
    origin            1.853326
    dtype: float64




```python
# Highest VIF is Displacement so remove all variables with VIF above 10 and check again
data2 = data1.drop(['cylinders', 'displacement', 'weight'],axis=1)

# Calculate VIF again after removing Displacement
X2 = sm.tools.add_constant(data2)
vif2 = pd.Series([variance_inflation_factor(X2.values,i) 
                     for i in range(X2.shape[1])], index=X2.columns)
vif2
```




    const           715.683761
    mpg               3.982206
    horsepower        4.131289
    acceleration      2.029527
    model year        1.607080
    origin            1.542532
    dtype: float64



#### All looks good and none of the variables' VIF is above 5 so other variables are all included in the model

## Predictive modeling

### Split the dataset into training and testing data


```python
# Divide data into independent and dependent data (X, y)
X = data2.drop('mpg',axis=1)  # create a DF of independent variables (without a dependent variable)
y = data2.mpg   # create a series of the dependent variable

# split the data into training and testing data (80-20 split)
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=.2, random_state=42)
```

### Fit regression models


```python
# Linear Regression
model_1 = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])
model_1.fit(X_train, y_train)

# Checking if model 1 is over or underfitted
m1_train_pred = model_1.predict(X_train)
m1_test_pred = model_1.predict(X_test)
# R2 Scores
m1_train_r2 = r2_score(y_train, m1_train_pred)
m1_test_r2 = r2_score(y_test, m1_test_pred)
# MSE
m1_train_mse = mean_squared_error(y_train, m1_train_pred)
m1_test_mse = mean_squared_error(y_test, m1_test_pred)
# Cross-validation
cv_scores1 = cross_val_score(model_1, X_train, y_train, cv=5, scoring='r2')

# Print results
print('Model 1: Linear Regression')
print(' Train R2 : {:.4f}'.format(m1_train_r2))
print(' Test  R2 : {:.4f}'.format(m1_test_r2))
print(' Train MSE: {:.4f}'.format(m1_train_mse))
print(' Test  MSE: {:.4f}'.format(m1_test_mse))
print(' CV R2: (M ± SD): {:.4f} ± {:.4f} \n'.format(cv_scores1.mean(), cv_scores1.std()))


# Ridge
d2 = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])
params = {'ridge__alpha': [0, 0.1, 0.01, 0.001, 1]}
model_2 = GridSearchCV(d2, params, cv=5, n_jobs=-1)
model_2.fit(X_train,y_train)

# Checking if model 2 is over or underfitted
m2_train_pred = model_2.predict(X_train)
m2_test_pred = model_2.predict(X_test)
# R2 Scores
m2_train_r2 = r2_score(y_train, m2_train_pred)
m2_test_r2 = r2_score(y_test, m2_test_pred)
# MSE
m2_train_mse = mean_squared_error(y_train, m2_train_pred)
m2_test_mse = mean_squared_error(y_test, m2_test_pred)
# Cross-validation
cv_scores2 = cross_val_score(model_2, X_train, y_train, cv=5, scoring='r2')

# Print results
print('Model 2: Ridge')
print(' Train R2 : {:.4f}'.format(m2_train_r2))
print(' Test  R2 : {:.4f}'.format(m2_test_r2))
print(' Train MSE: {:.4f}'.format(m2_train_mse))
print(' Test  MSE: {:.4f}'.format(m2_test_mse))
print(' CV R2: (M ± SD): {:.4f} ± {:.4f} \n'.format(cv_scores2.mean(), cv_scores2.std()))


# Lasso
d3 = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso())])
params = {'lasso__alpha':[0.1,0.01,0.001,1],
          'lasso__max_iter':[1000,10000,100000,1000000]}
model_3 = GridSearchCV(d3, params, cv=5, n_jobs=-1)
model_3.fit(X_train,y_train)

# Checking if model 2 is over or underfitted
m3_train_pred = model_3.predict(X_train)
m3_test_pred = model_3.predict(X_test)
# R2 Scores
m3_train_r2 = r2_score(y_train, m3_train_pred)
m3_test_r2 = r2_score(y_test, m3_test_pred)
# MSE
m3_train_mse = mean_squared_error(y_train, m3_train_pred)
m3_test_mse = mean_squared_error(y_test, m3_test_pred)
# Cross-validation
cv_scores3 = cross_val_score(model_3, X_train, y_train, cv=5, scoring='r2')

# Print results
print('Model 3: Lasso')
print(' Train R2 : {:.4f}'.format(m3_train_r2))
print(' Test  R2 : {:.4f}'.format(m3_test_r2))
print(' Train MSE: {:.4f}'.format(m3_train_mse))
print(' Test  MSE: {:.4f}'.format(m3_test_mse))
print(' CV R2: (M ± SD): {:.4f} ± {:.4f} \n'.format(cv_scores3.mean(), cv_scores3.std()))


# DTR
dtr = DecisionTreeRegressor(random_state=42)
params = {'max_features':[None,'sqrt','log2'],
          'min_samples_split':range(2, 10),
          'min_samples_leaf':range(1, 10),
          'max_depth':range(2, 10)} 
model_4 = RandomizedSearchCV(dtr, params, cv=5, n_jobs=-1)
model_4.fit(X_train,y_train)

# Checking if model 4 is over or underfitted
m4_train_pred = model_4.predict(X_train)
m4_test_pred = model_4.predict(X_test)
# R2 Scores
m4_train_r2 = r2_score(y_train, m4_train_pred)
m4_test_r2 = r2_score(y_test, m4_test_pred)
# MSE
m4_train_mse = mean_squared_error(y_train, m4_train_pred)
m4_test_mse = mean_squared_error(y_test, m4_test_pred)
# Cross-validation
cv_scores4 = cross_val_score(model_4.best_estimator_, X_train, y_train, cv=5, scoring='r2')

# Print results
print('Model 4: Decision Tree Regressor')
print(' Train R2 : {:.4f}'.format(m4_train_r2))
print(' Test  R2 : {:.4f}'.format(m4_test_r2))
print(' Train MSE: {:.4f}'.format(m4_train_mse))
print(' Test  MSE: {:.4f}'.format(m4_test_mse))
print(' CV R2: (M ± SD): {:.4f} ± {:.4f} \n'.format(cv_scores4.mean(), cv_scores4.std()))


# RFR
rfr = RandomForestRegressor(random_state=42)
params = {
    'n_estimators': [100,200,300,400,500],
    'max_depth': range(2, 10),
    'min_samples_split': range(2, 10),
    'min_samples_leaf': range(1, 10),
    'max_features': ['sqrt', 'log2']
}
model_5 = RandomizedSearchCV(rfr, params, cv=5, n_jobs=-1, n_iter=50, random_state=42)
model_5.fit(X_train,y_train)

# Checking if model 5 is over or underfitted
m5_train_pred = model_5.predict(X_train)
m5_test_pred = model_5.predict(X_test)
# R2 Scores
m5_train_r2 = r2_score(y_train, m5_train_pred)
m5_test_r2 = r2_score(y_test, m5_test_pred)
# MSE
m5_train_mse = mean_squared_error(y_train, m5_train_pred)
m5_test_mse = mean_squared_error(y_test, m5_test_pred)
# Cross-validation
cv_scores5 = cross_val_score(model_5.best_estimator_, X_train, y_train, cv=5, scoring='r2')

# Print results
print('Model 5: Random Forest Regressor')
print(' Train R2 : {:.4f}'.format(m5_train_r2))
print(' Test  R2 : {:.4f}'.format(m5_test_r2))
print(' Train MSE: {:.4f}'.format(m5_train_mse))
print(' Test  MSE: {:.4f}'.format(m5_test_mse))
print(' CV R2: (M ± SD): {:.4f} ± {:.4f} \n'.format(cv_scores5.mean(), cv_scores5.std()))


# GBR
gbr = GradientBoostingRegressor(random_state=42)
params = {'n_estimators': list(range(100, 501, 100)),
          'max_depth': list(range(3, 10)),
          'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
          'min_samples_split': list(range(2, 10)),
          'min_samples_leaf': list(range(1, 10)),
          'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
          'max_features': ['sqrt', 'log2']}
model_6 = RandomizedSearchCV(gbr, params, cv=5, n_jobs=-1, n_iter=50, random_state=42) 
model_6.fit(X_train,y_train)

# Checking if model 6 is over or underfitted
m6_train_pred = model_6.predict(X_train)
m6_test_pred = model_6.predict(X_test)
# R2 Scores
m6_train_r2 = r2_score(y_train, m6_train_pred)
m6_test_r2 = r2_score(y_test, m6_test_pred)
# MSE
m6_train_mse = mean_squared_error(y_train, m6_train_pred)
m6_test_mse = mean_squared_error(y_test, m6_test_pred)
# Cross-validation
cv_scores6 = cross_val_score(model_6.best_estimator_, X_train, y_train, cv=5, scoring='r2')

# Print results
print('Model 6: Gradient Boosting Regressor')
print(' Train R2 : {:.4f}'.format(m6_train_r2))
print(' Test  R2 : {:.4f}'.format(m6_test_r2))
print(' Train MSE: {:.4f}'.format(m6_train_mse))
print(' Test  MSE: {:.4f}'.format(m6_test_mse))
print(' CV R2: (M ± SD): {:.4f} ± {:.4f} \n'.format(cv_scores6.mean(), cv_scores6.std()))


# XGB
xgb = XGBRegressor(random_state=42)
params = {'n_estimators': list(range(100, 501, 100)),
          'max_depth': list(range(3, 10)),
          'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
          'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
          'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
          'min_child_weight': [1, 3, 5, 7],
          'gamma': [0, 0.1, 0.3, 0.5, 1],
          'reg_alpha': [0, 0.1, 0.5, 1],
          'reg_lambda': [1, 1.5, 2]}
model_7 = RandomizedSearchCV(xgb, params, cv=5, n_jobs=-1, n_iter=50, random_state=42) 
model_7.fit(X_train,y_train)

# Checking if model 7 is over or underfitted
m7_train_pred = model_7.predict(X_train)
m7_test_pred = model_7.predict(X_test)
# R2 Scores
m7_train_r2 = r2_score(y_train, m7_train_pred)
m7_test_r2 = r2_score(y_test, m7_test_pred)
# MSE
m7_train_mse = mean_squared_error(y_train, m7_train_pred)
m7_test_mse = mean_squared_error(y_test, m7_test_pred)
# Cross-validation
cv_scores7 = cross_val_score(model_7.best_estimator_, X_train, y_train, cv=5, scoring='r2')

# Print results
print('Model 7: XGBoost Regressor')
print(' Train R2 : {:.4f}'.format(m7_train_r2))
print(' Test  R2 : {:.4f}'.format(m7_test_r2))
print(' Train MSE: {:.4f}'.format(m7_train_mse))
print(' Test  MSE: {:.4f}'.format(m7_test_mse))
print(' CV R2: (M ± SD): {:.4f} ± {:.4f} \n'.format(cv_scores7.mean(), cv_scores7.std()))
```

    Model 1: Linear Regression
     Train R2 : 0.7425
     Test  R2 : 0.7719
     Train MSE: 16.1446
     Test  MSE: 12.2617
     CV R2: (M ± SD): 0.7317 ± 0.0452 
    
    Model 2: Ridge
     Train R2 : 0.7425
     Test  R2 : 0.7720
     Train MSE: 16.1446
     Test  MSE: 12.2571
     CV R2: (M ± SD): 0.7317 ± 0.0453 
    
    Model 3: Lasso
     Train R2 : 0.7425
     Test  R2 : 0.7720
     Train MSE: 16.1446
     Test  MSE: 12.2572
     CV R2: (M ± SD): 0.7317 ± 0.0452 
    
    Model 4: Decision Tree Regressor
     Train R2 : 0.8775
     Test  R2 : 0.8397
     Train MSE: 7.6775
     Test  MSE: 8.6200
     CV R2: (M ± SD): 0.7947 ± 0.0205 
    
    Model 5: Random Forest Regressor
     Train R2 : 0.9106
     Test  R2 : 0.8832
     Train MSE: 5.6023
     Test  MSE: 6.2799
     CV R2: (M ± SD): 0.8245 ± 0.0353 
    
    Model 6: Gradient Boosting Regressor
     Train R2 : 0.9364
     Test  R2 : 0.8801
     Train MSE: 3.9858
     Test  MSE: 6.4462
     CV R2: (M ± SD): 0.8381 ± 0.0185 
    
    Model 7: XGBoost Regressor
     Train R2 : 0.9526
     Test  R2 : 0.8692
     Train MSE: 2.9697
     Test  MSE: 7.0312
     CV R2: (M ± SD): 0.8251 ± 0.0239 
    



```python
# Create the table with parameters from above for easy comparison
model_results = pd.DataFrame({
    'Model': [
        'Linear Regression',
        'Ridge',
        'Lasso',
        'Decision Tree Regressor',
        'Random Forest Regressor',
        'Gradient Boosting Regressor',
        'XGBoost Regressor'
    ],
    'Train_R2': [m1_train_r2, m2_train_r2, m3_train_r2, m4_train_r2, m5_train_r2, m6_train_r2, m7_train_r2],
    'Test_R2': [m1_test_r2, m2_test_r2, m3_test_r2, m4_test_r2, m5_test_r2, m6_test_r2, m7_test_r2],
    'Train_MSE': [m1_train_mse, m2_train_mse, m3_train_mse, m4_train_mse, m5_train_mse, m6_train_mse, m7_train_mse],
    'Test_MSE': [m1_test_mse, m2_test_mse, m3_test_mse, m4_test_mse, m5_test_mse, m6_test_mse, m7_test_mse],
    'CV_R2_Mean': [cv_scores1.mean(), cv_scores2.mean(), cv_scores3.mean(), cv_scores4.mean(),
                   cv_scores5.mean(), cv_scores6.mean(), cv_scores7.mean()],
    'CV_R2_SD': [cv_scores1.std(), cv_scores2.std(), cv_scores3.std(), cv_scores4.std(),
                 cv_scores5.std(), cv_scores6.std(), cv_scores7.std()]
})

# Round up to 3 decimal places
model_results = model_results.round(3)

# Display the table
model_results.sort_values(by=['Test_R2'], ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Train_R2</th>
      <th>Test_R2</th>
      <th>Train_MSE</th>
      <th>Test_MSE</th>
      <th>CV_R2_Mean</th>
      <th>CV_R2_SD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Random Forest Regressor</td>
      <td>0.911</td>
      <td>0.883</td>
      <td>5.602</td>
      <td>6.280</td>
      <td>0.824</td>
      <td>0.035</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gradient Boosting Regressor</td>
      <td>0.936</td>
      <td>0.880</td>
      <td>3.986</td>
      <td>6.446</td>
      <td>0.838</td>
      <td>0.019</td>
    </tr>
    <tr>
      <th>6</th>
      <td>XGBoost Regressor</td>
      <td>0.953</td>
      <td>0.869</td>
      <td>2.970</td>
      <td>7.031</td>
      <td>0.825</td>
      <td>0.024</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Decision Tree Regressor</td>
      <td>0.878</td>
      <td>0.840</td>
      <td>7.678</td>
      <td>8.620</td>
      <td>0.795</td>
      <td>0.020</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Linear Regression</td>
      <td>0.742</td>
      <td>0.772</td>
      <td>16.145</td>
      <td>12.262</td>
      <td>0.732</td>
      <td>0.045</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ridge</td>
      <td>0.742</td>
      <td>0.772</td>
      <td>16.145</td>
      <td>12.257</td>
      <td>0.732</td>
      <td>0.045</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lasso</td>
      <td>0.742</td>
      <td>0.772</td>
      <td>16.145</td>
      <td>12.257</td>
      <td>0.732</td>
      <td>0.045</td>
    </tr>
  </tbody>
</table>
</div>



## Best fitting model

The Decision Tree Regressor shows a strong test R² of 0.8536, but the relatively large gap between train MSE (7.91) and test MSE (7.87) suggests a close but potentially overfit model. Its cross-validated R² of 0.7621 with a standard deviation of 0.0229 indicates moderate generalizability, though not as high as some ensemble methods.

Comparing the Random Forest Regressor and the Gradient Boosting Regressor, both exhibit excellent performance. Their test R² values are very close (0.8832 for Random Forest vs. 0.8801 for Gradient Boosting), indicating strong and similar generalization. However, Gradient Boosting achieves lower train and test MSE (3.99 and 6.45 respectively) compared to Random Forest (5.60 and 6.28), suggesting a slightly better fit.

Moreover, Gradient Boosting has the highest cross-validated R² of 0.8381 and the smallest standard deviation (0.0185) among all models, indicating both excellent predictive power and exceptional stability across folds.

While XGBoost shows the highest Train R² (0.9526) and lowest Train MSE (2.97), its larger gap to Test R² (0.8692) and Test MSE (7.03) suggests a degree of overfitting. Its CV R² of 0.8251 is strong, but still slightly lower than that of Gradient Boosting.

For these reasons, Gradient Boosting Regressor is identified as the best-performing model for this dataset, offering a strong balance of accuracy, generalization, and stability.

## Vizualization of predicted vs. actual data


```python
# Create a scatterplot comparing actual and predicted mpg values
dataP = data2.drop('mpg',axis=1)  # create a new DataFrame of the feature variables

# make a DataFrame of the actual mpg and the predicted mpg 
dataGraph = pd.DataFrame({'Actual mpg':data2.mpg.values,
                      'Predicted mpg':model_6.predict(dataP.values)})

# make a scatter plot of the actual and the predicted mpg of a car
plt.figure(figsize=(12,8))
plt.scatter(dataGraph.index,dataGraph['Actual mpg'].values, color='#298c8c', label='Actual mpg')
plt.scatter(dataGraph.index,dataGraph['Predicted mpg'].values, color='#ea801c', label='Predicted mpg')
plt.title('Comapring the Actual mpg values to the Predicted mpg values\nModel accuracy = 88%', fontsize=16)
plt.xlabel('Car index')
plt.ylabel('Mile Per Gallon (mpg)')
plt.legend(loc='upper left')
plt.show()
```


    
![png](mpg_files/output_29_0.png)

