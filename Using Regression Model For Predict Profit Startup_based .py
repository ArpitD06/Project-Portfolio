from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
df = pd.read_csv("C:/Users/dubey/OneDrive/Desktop/startups.csv")
print(df)
X = df.iloc[:,0:4].values
y = df.iloc[:,4].values

print(y)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
le = LabelEncoder()
X[:,3] = le.fit_transform(X[:,3] )
transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(),[3])],remainder='passthrough')
X = transformer.fit_transform(X)
X=X[:,1:]
print(X)

## how to replace missing values using sklearn
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy='mean')
X[:,0:3] = imputer.fit_transform(X[:,0:3])
print('After replacing missing values',X)

# Feature scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler
sc = StandardScaler()
X[:,2:] = sc.fit_transform(X[:,2:] )
print(X)


#  EDA :
import matplotlib.pyplot as plt
plt.scatter(df['R&D Spend'], df['Profit'])
plt.show()

#MODEL BUILDING :

# split the data into train and test test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# y = mx + c ==> Simple Linear Regression
# y = m1x1 + m2x2 + m3x3... + c
print("Coefficient value = ",regressor.coef_)
print("Intercept value = ",regressor.intercept_)

# testing the model - predict y for the hidden x (x_test)
y_pred_test = regressor.predict(X_test)
result_df = pd.DataFrame({'Actual':y_test,'Predicted':y_pred_test})
print(result_df)



'''
1. Evaluate
2. Test if the analysis itself is worth (Hypothesis
3. Is this the best solution possible by this algorithm?

'''
# Model Evaluation
from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, y_pred_test)
print("Mean Absolute Error = ", mae)
'''
3069.05             20186329.71                4492.919954
Mean Absolute Error Mean Squared Error Root Mean Squared Error

'''
mse = metrics.mean_squared_error(y_test, y_pred_test)
rmse = mse ** 0.5
print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", rmse)
print("R Square Score: ", metrics.r2_score(y_test, y_pred_test))

'''
All the X included:
Mean Absolute Error =  2969.0527677379227
Mean Squared Error:  17941201.05396587
Root Mean Squared Error:  4235.705496604535
R Square Score:  0.9901105113397771
-----------------------------------------------------
Only R&D Spend:
Mean Absolute Error =  4622.662727991248
Mean Squared Error:  29243263.17924025
Root Mean Squared Error:  5407.704058030566

R square talks about the model accuracy with regression line vs average line
'''


X = df.iloc[:, 1:2].values
y = df.iloc[:, 4].values

# MODEL BUILDING :

# split the data into train and test test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# y = mx + c ==> Simple Linear Regression
print("Coefficient value = ", regressor.coef_)
print("Intercept value = ", regressor.intercept_)

# testing the model - predict y for the hidden x (x_test)
y_pred_test = regressor.predict(X_test)
result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
print(result_df)

# Model Evaluation
from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, y_pred_test)
print("Mean Absolute Error = ", mae)

mse = metrics.mean_squared_error(y_test, y_pred_test)
rmse = mse ** 0.5
print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", rmse)
print("R Square Score: ", metrics.r2_score(y_test, y_pred_test))

'''
Only R&D Spend:
Mean Absolute Error =  4622.662727991248
Mean Squared Error:  29243263.17924025
Root Mean Squared Error:  5407.704058030566
-------------------------------------------------
Only Marketing Spend:
Mean Absolute Error =  23050.338819800003
Mean Squared Error:  778280710.8115087
Root Mean Squared Error:  27897.682893235215
--------------------------------------------------
Only Admin spend:
Mean Absolute Error =  32957.1972634327
Mean Squared Error:  1774858259.7633147
Root Mean Squared Error:  42129.06668516779
---------------------------------------------------
You will face this challenge:
    Efficiency vs Accuracy
'''


