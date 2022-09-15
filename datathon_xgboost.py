import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df_test = pd.read_csv('bike_test.csv', decimal=",")
df_test.drop(columns= 'dteday', inplace= True)


df = pd.read_csv('bike_train.csv', decimal=",")
df.drop(columns=['casual', 'registered', 'dteday'], inplace=True)
data = df.copy()

X, y = data.iloc[:,:-1],data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


'''
df = pd.read_csv('bike_train.csv', index_col='instant', decimal=",")
df.drop(columns=['casual', 'registered', 'dteday'], inplace=True)
data_train = df.iloc[:10000, :]
data_test = df.iloc[10000:, :]



df_test = pd.read_csv('bike_test.csv', index_col='instant', decimal=",")
df_test.drop(columns= 'dteday', inplace= True)

X_train = data_train.iloc[:, :-1]
y_train = data_train.iloc[:, -1:]

X_test = data_test.iloc[:, :-1]
y_test =data_test.iloc[:, -1:]
'''


# Various hyper-parameters to tune
xgb1 = XGBRegressor()
parameters = {
              'n_estimators': [800],
              'colsample_bytree': [0.8],
              'max_depth': [15],
              'reg_alpha': [1.1, 1.2, 1.3],
              'reg_lambda': [1.1, 1.2, 1.3],
              'subsample': [0.7, 0.8]
              }

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 5,
                        n_jobs = -1,
                        verbose=True)

xgb_grid.fit(X_train, y_train)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

preds = xgb_grid.predict(X_test)


# rmse test
mse = mean_squared_error(y_test, preds)
rmse = mse**.5

print(f' rmse test: {rmse}')

#rmse train
prediction_train = xgb_grid.predict(X_train)
mse = mean_squared_error(y_train, prediction_train)
rmse = mse**.5

print(f'rmse train: {rmse}')


prediction0 = xgb_grid.predict(df_test).round(2)
df_predict = pd.DataFrame(prediction0, columns=['pred'])


df_predict.to_csv('horaciogit.csv', index=False)


#params colab
'''
Fitting 10 folds for each of 18 candidates, totalling 180 fits
0.9579325347257572
{'colsample_bytree': 0.8, 'learning_rate': 0.02, 'max_depth': 15, 'min_child_weight': 5, 'n_estimators': 1000, 'nthread': 4, 'objective': 'reg:squarederror', 'silent': 1, 'subsample': 0.7}
'''



