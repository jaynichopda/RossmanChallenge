import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


print ("Load data")
train = pd.read_csv("data/train.csv", low_memory=False)
test = pd.read_csv("data/test.csv", low_memory=False)
store = pd.read_csv("data/store.csv", low_memory=False)

train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

## Sort data by store and date
train = train.sort_values(by=['Store','Date'])
train.head()
test = test.sort_values(by=['Store','Date'])
test.head()

list(train)

## Column names for reference:
['Store',
 'DayOfWeek',
 'Date',
 'Sales',
 'Customers',
 'Open',
 'Promo',
 'StateHoliday',
 'SchoolHoliday',
 'StoreType',
 'Assortment',
 'CompetitionDistance',
 'CompetitionOpenSinceMonth',
 'CompetitionOpenSinceYear',
 'Promo2',
 'Promo2SinceWeek',
 'Promo2SinceYear',
 'PromoInterval']

## EDA
'''
print ("Show plots")
## Scatter Plot
sns.lmplot(x='CompetitionDistance', y='Sales', data = train)
sns.lmplot(x='Customers', y='Sales', data = train)
sns.lmplot(x='DayOfWeek', y='Sales', data = train)
sns.lmplot(x='Assortment', y='Sales', data = train, fit_reg=False)

# 1. Enlarge the plot
plt.figure(figsize=(20,20))

sns.lmplot(x='StoreType', y='Sales', data = train, fit_reg=False)
 
# Scatterplot arguments
sns.lmplot(x='CompetitionDistance', y='Sales', data=train,
           fit_reg=False, # No regression line
           hue='Assortment')   # Color by assortment

# Calculate correlations
corr = train.corr()
#corr
 
# Heatmap
sns.heatmap(corr) 

# Distribution Plot (Histogram)
sns.distplot(train.Sales)

# Count Plot (Bar Plot)
sns.countplot(x='Open', data= train)
'''

## Starting to build model
## Consider only open stores with non zero sales for training
train = train[train["Open"] != 0]
train = train[train["Sales"] != 0]

# Thanks to Chenglong Chen for providing this in the forum
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe

def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe


# Build some features
def all_features(features, data):
    # Given Features
    features.extend(['Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                     'CompetitionOpenSinceYear', 'Promo', 'Promo2', 'Promo2SinceWeek',
                     'Promo2SinceYear', 'DayOfWeek','SchoolHoliday'])

    
    #Custom Features
    features.append('month')
    features.append('year')
    features.append('DayBeforeHoliday')
    features.append('XMAS')
    
    data['year'] = data['Date'].apply(lambda x: int(x.split('-')[0]))
    data['month'] = data['Date'].apply(lambda x: int(x.split('-')[1]))
    data['Date_D'] = data['Date'].apply(lambda x: int(x.split('-')[2]))
    data['DayBeforeHoliday'] = np.where((data['SchoolHoliday'].shift(-1)==1) & (data['SchoolHoliday']==0), 1, 0)
    data['XMAS'] = np.where((data['month']==12)&((data['Date_D']==22)|(data['Date_D']==23)),1,0)
    
    
    for x in ['StoreType', 'Assortment', 'StateHoliday']:
        features.append(x)
        labels = data[x].unique()
        map_labels = dict(zip(labels, range(0,len(labels))))
        data[x] = data[x].map(map_labels)
    
    

features = []
all_features(features, train)
all_features([], test)
 
num_trees = 400

params = {"objective": "reg:linear",
          "eta": 0.3,
          "max_depth": 10,
          "subsample": 0.85,
          "colsample_bytree": 0.7,
          "silent": 1
          }


print("Train a XGBoost model")

X_train, X_test = train_test_split(train, test_size=10000, random_state=42)
dtrain = xgb.DMatrix(X_train[features], np.log(X_train["Sales"] + 1))
dvalid = xgb.DMatrix(X_test[features], np.log(X_test["Sales"] + 1))
dtest = xgb.DMatrix(test[features])

#Validation to watch performance 
watch_list = [(dvalid, 'eval'), (dtrain, 'train')]

gbm = xgb.train(params, dtrain, num_trees,
                evals=watch_list,
                early_stopping_rounds=100, 
                feval=rmspe_xg, verbose_eval=True)

print("Validating")
train_pred = gbm.predict(xgb.DMatrix(X_test[features]))

indices = train_pred < 0
train_pred[indices] = 0
error = rmspe(np.exp(train_pred) - 1, X_test['Sales'].values)
print('error', error)



## Test Data

test_pred = gbm.predict(xgb.DMatrix(test[features]))

indices = test_pred < 0
test_pred[indices] = 0
closed_idx = (test['Open']==0).values
test_pred[closed_idx] = 0
submission_test = pd.DataFrame({"Id": test["Id"], "Sales": np.exp(test_pred) - 1})
submission_test.to_csv("xgboost_submit.csv", index=False)
