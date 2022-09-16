import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# \Lectura dataset y preparacion conjunto entrenamiento y test 
#=================================================================0
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

bike_test= scaler.transform(df_test)


# Regresion arboles aleatorios
#===================================================================
rf = RandomForestRegressor(n_estimators = 1500, max_features = 'auto', max_depth = 15, random_state = 18,
                           bootstrap=True, min_samples_leaf= 3, min_samples_split= 6).fit(X_train,y_train.values.ravel())

prediction = rf.predict(X_test)

# rmse test
#===================================================
mse = mean_squared_error(y_test, prediction)
rmse = mse**.5
print(f' mse antes gridsearch: {mse}')
print(f' rmse antes gridsearch: {rmse}')

#prediction train, rmse train
#=====================================================
prediction_train = rf.predict(X_train)
mse = mean_squared_error(y_train, prediction_train)
rmse = mse**.5
print(f' mse train: {mse}')
print(f'rmse train: {rmse}')


# Grafica comparar real vs prediccion
#============================================================
yplot=y_test.reset_index()
ypredplot=pd.DataFrame(prediction)
 

fig, ax = plt.subplots(figsize=(12, 3.5))
yplot.loc[100:200, 'cnt'].plot(ax=ax, linewidth=2, label='real')
ypredplot.loc[100:200, :].plot(linewidth=2, label='prediction', ax=ax)
ax.set_title('Prediction vs real ')
ax.legend();
plt.show()



#tuning grid search
#=====================================================
parameters = {
    'bootstrap': [True],
    'n_estimators': [1000],
    'max_depth': [30],
#    'max_features': [2, 3],
    'min_samples_leaf': [2],
    'min_samples_split': [2],
    
    
}
regr = RandomForestRegressor(random_state=18)

clf = GridSearchCV(regr, 
                   parameters,
                   cv=5,
                   n_jobs=-1,
                   verbose=2)


clf.fit(X_train, y_train.values.ravel())

prediction = clf.predict(X_test)

# Evaluacion rmse y nuevo grafico
#===================================================0
mse = mean_squared_error(y_test, prediction)
rmse = mse**.5
print(f' mse despues--- gridsearch---: {mse}')
print(f' rmse despues--- gridsearch--: {rmse}')


ypredplot=pd.DataFrame(prediction)
 
# Nuevo Plot 
#
fig, ax = plt.subplots(figsize=(12, 3.5))
yplot.loc[100:200, 'cnt'].plot(ax=ax, linewidth=2, label='real')
ypredplot.loc[100:200, :].plot(linewidth=2, label='prediction', ax=ax)
ax.set_title('Prediction vs real ')
ax.legend();
plt.show()


print (f' best params : {clf.best_params_}')



prediction0 = rf.predict(bike_test).round(2)
df_predict= pd.DataFrame(prediction0, columns=['pred'])

#df_predict.to_csv('horaciogit.csv', index=False)


