%pip install seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("housing.csv")

df.head()

df.describe()

df.info()

df["total_bedrooms"].median()
df["total_bedrooms"]=df["total_bedrooms"].fillna(median)
df=pd.get_dummies(df, columns=['ocean_proximity'])

df.info()

print(df.isnull().sum())

sns.regplot(x="median_income", y="median_house_value", data=df)

X=df.drop("median_house_value",axis=1)
y=df["median_house_value"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaler=scaler.fit_transform(X_train)
X_test_scaler=scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
print("MSE:",mse)
print("RMSE:",rmse)
print("MAE:",mae)
print("R2",r2)
