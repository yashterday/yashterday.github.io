# Housing Prices Regression Analysis

### Kaggle Competition: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview

### I used logistic regressions for my analysis with this dataset as I still need work with my more advanced machine learning techniques

## Data Handling

```python
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
%matplotlib inline
```

```python
X = pd.read_csv('house_train.csv')
X_test = pd.read_csv('house_test.csv')

y = X['SalePrice'].reset_index(drop=True)
y = np.log1p(y)
train_features = X.drop(['SalePrice'], axis=1)
features = pd.concat([train_features, X_test]).reset_index(drop=True)

#check for null values which may mess with our predictions later on
features.isnull().sum().sort_values(ascending = False).head(20)
```

    PoolQC          2909
    MiscFeature     2814
    Alley           2721
    Fence           2348
    FireplaceQu     1420
    LotFrontage      486
    GarageCond       159
    GarageQual       159
    GarageYrBlt      159
    GarageFinish     159
    GarageType       157
    BsmtCond          82
    BsmtExposure      82
    BsmtQual          81
    BsmtFinType2      80
    BsmtFinType1      79
    MasVnrType        24
    MasVnrArea        23
    MSZoning           4
    BsmtHalfBath       2
    dtype: int64

```python
#double check the data type of each variable
features.dtypes
```

    Id                 int64
    MSSubClass         int64
    MSZoning          object
    LotFrontage      float64
    LotArea            int64
                      ...   
    MiscVal            int64
    MoSold             int64
    YrSold             int64
    SaleType          object
    SaleCondition     object
    Length: 80, dtype: object

### We need to replace the NA values in the data set with either 0 for numeric or most common/"None" for categoric depending on how the variable is being treated

```python
for column in features:

    # populating with 0
    if column in ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF','GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'TotalBsmtSF','Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea']:
        features[column] = features[column].fillna(0)

    # populate with 'None'
    if column in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', "PoolQC", 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2', 'Neighborhood', 'BldgType', 'HouseStyle', 'MasVnrType', 'FireplaceQu', 'Fence', 'MiscFeature']:
        features[column] = features[column].fillna('None')

    # populate with most frequent value for categorical data
    if column in ['Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'RoofStyle', 'Electrical', 'Functional', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'RoofMatl', 'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'PavedDrive', 'SaleType', 'SaleCondition']:
        features[column] = features[column].fillna(features[column].mode()[0])
```

```python
features.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>

## Feature Engineering

### Now that we have the data processed its time to add additional independent variables as well to help the prediction model have a far more accurate prediction

```python
features['total_yrs'] = features['YearRemodAdd'] - features['YearBuilt']  
features['total_sqrft'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

features['total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] + features['1stFlrSF'] + features['2ndFlrSF'])

features['total_bath'] = (features['FullBath'] + (0.5 * features['HalfBath']) + features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))

features['total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] + features['EnclosedPorch'] + features['ScreenPorch'] + features['WoodDeckSF'])
```

```python
features['pool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
features['2nd_flr'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
features['garage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
features['bsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
features['fireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

#handling the Nulls not taken care of by the feature engineering
features['MSSubClass'] = features['MSSubClass'].apply(str)
features["MSSubClass"] = features["MSSubClass"].fillna("Unknown")

features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

features['LotArea'] = features['LotArea'].astype(np.int64)

features['Alley'] = features['Alley'].fillna('Pave')

features['MasVnrArea'] = features['MasVnrArea'].astype(np.int64)
```

```python
features.shape
features.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>total_yrs</th>
      <th>total_sqrft</th>
      <th>total_sqr_footage</th>
      <th>total_bath</th>
      <th>total_porch_sf</th>
      <th>pool</th>
      <th>2nd_flr</th>
      <th>garage</th>
      <th>bsmt</th>
      <th>fireplace</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>2566.0</td>
      <td>2416.0</td>
      <td>3.5</td>
      <td>61</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>2524.0</td>
      <td>2240.0</td>
      <td>2.5</td>
      <td>298</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>1</td>
      <td>2706.0</td>
      <td>2272.0</td>
      <td>3.5</td>
      <td>42</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>55</td>
      <td>2473.0</td>
      <td>1933.0</td>
      <td>2.0</td>
      <td>307</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>3343.0</td>
      <td>2853.0</td>
      <td>3.5</td>
      <td>276</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 90 columns</p>
</div>

```python
# double check for null
features.isnull().sum().sort_values(ascending = False).head(20)
```

    fireplace       0
    RoofMatl        0
    Exterior2nd     0
    MasVnrType      0
    MasVnrArea      0
    ExterQual       0
    ExterCond       0
    Foundation      0
    BsmtQual        0
    BsmtCond        0
    BsmtExposure    0
    BsmtFinType1    0
    BsmtFinSF1      0
    BsmtFinType2    0
    BsmtFinSF2      0
    BsmtUnfSF       0
    TotalBsmtSF     0
    Heating         0
    HeatingQC       0
    CentralAir      0
    dtype: int64

## Time for Model Building and Fitting

```python
#one hot encoding to make it easier and faster for the model
features_2 = pd.get_dummies(features).reset_index(drop=True)

#go back to X and X_test so we have our train and test split, use length of y to help seperate the two back
X = features_2.iloc[:len(y), :]
X_test = features_2.iloc[len(X):, :]
print('Dimensions for each df')
print('X', X.shape, 'y', y.shape, 'X_test', X_test.shape)
```

    Dimensions for each df
    X (1460, 327) y (1460,) X_test (1459, 327)

```python
X.dtypes
```

    Id                         int64
    LotFrontage              float64
    LotArea                    int64
    OverallQual                int64
    OverallCond                int64
                              ...   
    SaleCondition_AdjLand      uint8
    SaleCondition_Alloca       uint8
    SaleCondition_Family       uint8
    SaleCondition_Normal       uint8
    SaleCondition_Partial      uint8
    Length: 327, dtype: object

### For this current iteration of the Regression, we only used ridge regression with 10 k-folds, but I will start to stack different models as my skillset grows

```python
#import models
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
```

### Ridge Regression

```python
kfolds = KFold(n_splits = 10, random_state = 42, shuffle = True)
alphas_no = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_no, cv=kfolds))

ridge = ridge.fit(X,y)
rmse_cv(ridge).mean()
```

    0.13773568754637133

### LASSO Regression

```python
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=42))
rmse_cv(lasso).mean()
```

    0.1348839439429676

### We have a solid Root Mean Square error, but I'm confident that once I start using stacked models, this error will drop closer to 0
