import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('bike_train.csv', decimal=",")
df.drop(columns=['casual', 'registered', 'dteday'], inplace=True)


df_test = pd.read_csv('bike_test.csv',  decimal=",")
df_test.drop(columns= 'dteday', inplace= True)

X = df.iloc[:, :-1]
y = df.iloc[:, -1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



rf = RandomForestRegressor(n_estimators = 1200, max_features = 'auto', max_depth = 18, random_state = 18,
                           bootstrap=True, min_samples_leaf= 3, min_samples_split= 6).fit(X_train,y_train.values.ravel())

prediction = rf.predict(X_test)

mse = mean_squared_error(y_test, prediction)
rmse = mse**.5
print(f' mse antes gridsearch: {mse}')
print(f' rmse antes gridsearch: {rmse}')

#prediction train
prediction_train = rf.predict(X_train)
mse = mean_squared_error(y_train, prediction_train)
rmse = mse**.5
print(f' mse train: {mse}')
print(f'rmse train: {rmse}')



yplot=y_test.reset_index()
ypredplot=pd.DataFrame(prediction)
 
#Plot 
#
fig, ax = plt.subplots(figsize=(12, 3.5))
yplot.loc[100:200, 'cnt'].plot(ax=ax, linewidth=2, label='real')
ypredplot.loc[100:200, :].plot(linewidth=2, label='prediction', ax=ax)
ax.set_title('Prediction vs real ')
ax.legend();
plt.show()



#tuning grid search
parameters = {
    'bootstrap': [True],
    'n_estimators': [800,1000],
    'max_depth': [13,18],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10],
    
    
}
regr = RandomForestRegressor(random_state=18, criterion='squared_error', n_jobs=-1)

clf = GridSearchCV(regr, parameters)
clf.fit(X_train, y_train.values.ravel())




prediction = clf.predict(X_test)

mse = mean_squared_error(y_test, prediction)
rmse = mse**.5
print(f' mse despues--- gridsearch---: {mse}')
print(f' rmse despues--- gridsearch--: {rmse}')


ypredplot=pd.DataFrame(prediction)
 
#Plot 
#
fig, ax = plt.subplots(figsize=(12, 3.5))
yplot.loc[100:200, 'cnt'].plot(ax=ax, linewidth=2, label='real')
ypredplot.loc[100:200, :].plot(linewidth=2, label='prediction', ax=ax)
ax.set_title('Prediction vs real ')
ax.legend();
plt.show()


print (f' best params : {clf.best_params_}')



prediction0 = rf.predict(df_test).round(2)
df_predict= pd.DataFrame(prediction0, columns=['pred'])

#df_predict.to_csv('horaciogit.csv', index=False)


