# Heart Disease Decision Tree Classifier Model

#### This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The "target" field refers to the presence of heart disease in the patient.

# 1.  Get the Data Ready

```python
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
%matplotlib inline
```

```python
# the heart.csv file can be found in the github along with this notebook
df = pd.read_csv('heart.csv')
df.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>3</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>1</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
      <td>1</td>
      <td>1</td>
      <td>120</td>
      <td>236</td>
      <td>0</td>
      <td>1</td>
      <td>178</td>
      <td>0</td>
      <td>0.8</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>354</td>
      <td>0</td>
      <td>1</td>
      <td>163</td>
      <td>1</td>
      <td>0.6</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

```python
#double check to see if there's some balance to the amount of 1s and 0s for the targets
df['target'].value_counts()
```

    1    165
    0    138
    Name: target, dtype: int64

#### I converted the 1s and 0s into Yes and No as well as rename the column to ensure it would be easier for viewers to read later on.

```python
df['target'] = df['target'].replace(0, 'no');
df['target'] = df['target'].replace(1, 'yes');
df = df.rename(columns={'target': 'heart_disease'})
```

```python
# making sure that our values are in the correct form for our shift to a numpy array
df.dtypes
```

    age                int64
    sex                int64
    cp                 int64
    trestbps           int64
    chol               int64
    fbs                int64
    restecg            int64
    thalach            int64
    exang              int64
    oldpeak          float64
    slope              int64
    ca                 int64
    thal               int64
    heart_disease     object
    dtype: object

    age                int64
    sex                int64
    cp                 int64
    trestbps           int64
    chol               int64
    fbs                int64
    restecg            int64
    thalach            int64
    exang              int64
    oldpeak          float64
    slope              int64
    ca                 int64
    thal               int64
    heart_disease     object
    dtype: object

# 2.  Pre-Processing

#### Get your arrays ready for the decision tree model and create one array with the independent variables and one without to use for predictions in the future

```python
#note the one-hot encoding was already present in the data set so didn't need to be done
X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']].values
X[0:5]
```

    array([[ 63. ,   1. ,   3. , 145. , 233. ,   1. ,   0. , 150. ,   0. ,
              2.3,   0. ,   0. ,   1. ],
           [ 37. ,   1. ,   2. , 130. , 250. ,   0. ,   1. , 187. ,   0. ,
              3.5,   0. ,   0. ,   2. ],
           [ 41. ,   0. ,   1. , 130. , 204. ,   0. ,   0. , 172. ,   0. ,
              1.4,   2. ,   0. ,   2. ],
           [ 56. ,   1. ,   1. , 120. , 236. ,   0. ,   1. , 178. ,   0. ,
              0.8,   2. ,   0. ,   2. ],
           [ 57. ,   0. ,   0. , 120. , 354. ,   0. ,   1. , 163. ,   1. ,
              0.6,   2. ,   0. ,   2. ]])

```python
y = df[['heart_disease']]
y[0:5]
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
      <th>heart_disease</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>yes</td>
    </tr>
  </tbody>
</table>
</div>

```python
#### Now use the train_test split to get training and testing splits for the X and Y arrays
```

```python
from sklearn.model_selection import train_test_split
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 3)
#check size of each set
X_train.shape
y_train.shape
X_test.shape
y_test.shape
```

    (91, 1)

# 4. Prediction Time

#### We will be using a decision tree classifier model with an "entropy" criterion to have the tree pick columns by best information availability.

```python
heartTree = DecisionTreeClassifier(criterion="entropy")
heartTree
```

    DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort=False,
                           random_state=None, splitter='best')

```python
heartTree.fit(X_train,y_train)
```

    DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort=False,
                           random_state=None, splitter='best')

#### Using the model we just made with heartTree, we will now apply the tree to the testing sets to check how accurate our model really is

```python
predTree = heartTree.predict(X_test)
```

```python
print (predTree[0:5])
print (y_test[0:5])
```

    ['yes' 'yes' 'yes' 'yes' 'yes']
        heart_disease
    245            no
    162           yes
    10            yes
    161           yes
    73            yes

#### It seems like the model is not 100% accurate but lets check the accuracy score to get an idea of all the predictions

```python
from sklearn import metrics
from sklearn.metrics import mean_squared_error
x = pd.DataFrame(predTree)

import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))
```

    DecisionTrees's Accuracy:  0.8131868131868132

```python
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

enc = preprocessing.OneHotEncoder()

enc.fit(y_test)

onehotlabels = enc.transform(y_test).toarray()
onehotlabels.shape

enc.fit(x)

onehotlabels2 = enc.transform(x).toarray()
onehotlabels2.shape
```

    (91, 2)

```python
def rmse(y_actual, y_pred):
    return np.sqrt(mean_squared_error(y_actual, y_pred))

print('RMSE score on train data:')
print(rmse(onehotlabels, onehotlabels2))
```

    RMSE score on train data:
    0.4322189107537832

### Very high root mean squre error, I'll be coming back to improve this model!
