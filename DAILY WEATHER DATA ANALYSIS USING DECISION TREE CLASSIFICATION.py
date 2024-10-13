#PROJECT 5: DAILY WEATHER DATA ANALYSIS USING DECISION TREE CLASSIFICATION -------
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
'''
Classification of Weather Data using scikit-learn----
'''
data = pd.read_csv("C:/Users/dubey/Downloads/data (1)/data/data_weather.csv")
print("Columns are:",data.columns)
print("Data:\n",data)
print("Null Data:\n",data[data.isnull().any(axis=1)])

'''
Let us now check all the columns in the data. 
 Each row in daily_weather.csv captures weather data for a separate day. 
 Sensor measurements from the weather station were captured at one-minute intervals. These 
measurements were then processed to generate values to describe daily weather. Since this dataset 
was created to classify low-humidity days vs. non-low-humidity days (that is, days with normal or 
high humidity), the variables included are weather measurements in the morning, with one 
measurement, namely relatively humidity, in the afternoon. The idea is to use the morning weather 
values to predict whether the day will be low-humidity or not based on the afternoon measurement 
of relative humidity. 
 Each row, or sample, consists of the following variables: 
o number: unique number for each row 
o air_pressure_9am: air pressure averaged over a period from 8:55am to 9:04am (Unit: 
hectopascals) 
o air_temp_9am: air temperature averaged over a period from 8:55am to 9:04am (Unit: 
degrees Fahrenheit) 
o air_wind_direction_9am: wind direction averaged over a period from 8:55am to 9:04am 
(Unit: degrees, with 0 means coming from the North, and increasing clockwise) 
o air_wind_speed_9am: wind speed averaged over a period from 8:55am to 9:04am (Unit: 
miles per hour) 
o max_wind_direction_9am:** wind gust direction averaged over a period from 8:55am to 
9:10am (Unit: degrees, with 0 being North and increasing clockwise*) 
o max_wind_speed_9am: wind gust speed averaged over a period from 8:55am to 9:04am 
(Unit: miles per hour) 
o rain_accumulation_9am: amount of rain accumulated in the 24 hours prior to 9am (Unit: 
millimeters) 
o rain_duration_9am: amount of time rain was recorded in the 24 hours prior to 9am (Unit: 
seconds) 
o relative_humidity_9am: relative humidity averaged over a period from 8:55am to 9:04am 
(Unit: percent) 
o relative_humidity_3pm
'''

#Data Cleaning Steps:
#We will not need the "number" column for each row so we can clean it.
del data['number']
#Let us drop null values using the pandas dropna function.
before_rows = data.shape[0]
print(before_rows)
data = data.dropna()
after_rows = data.shape[0]
print(after_rows)
#How many rows dropped due to cleaning?
print("Total rows dropped: ",before_rows - after_rows)

#Convert to a Classification Task
# Binarize the relative_humidity_3pm to 0 or 1.
clean_data = data.copy()
clean_data['high_humidity_label'] =(clean_data['relative_humidity_3pm'] > 24.99) * 1
print(clean_data['high_humidity_label'])

# Target is stored in 'y'.
y= clean_data[['high_humidity_label']].copy()
clean_data['relative_humidity_3pm'].head()
print("Y Data: \n", y.head())

#Use 9am Sensor Signals as Features to Predict Humidity at 3pm
morning_features = ['air_pressure_9am',	'air_temp_9am',	'avg_wind_direction_9am', 'avg_wind_speed_9am', 'max_wind_direction_9am',
'max_wind_speed_9am',	'rain_accumulation_9am',	'rain_duration_9am']
X = clean_data[morning_features].copy()
print("Columns in X:",X.columns)
print("Columns in Y:",y.columns)
'''
Perform Test and Train split----
 In the training phase, the learning algorithm uses the training data to adjust the model’s parameters 
to minimize errors. At the end of the training phase, we get the trained model. 
 In the testing phase, the trained model is applied to test data. Test data is separate from the training 
data, and is previously unseen by the model. The model is then evaluated on how it performs on the 
test data. The goal in building a classifier model is to have the model perform well on training as well 
as test data. 
'''
#Perform Test and Train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)
print ("X_train is as under:")
print(X_train.head())
print ("X_test is as under:")
print(X_test.head())
print ("y_train is as under:")
print(y_train.head())
print ("y_test is as under:")
print(y_test.head())
print ("Let us describe y_train:\n",y_train.describe())

#Fit on Train Set
humidity_classifier = DecisionTreeClassifier(max_leaf_nodes=10,
random_state=0)
humidity_classifier.fit(X_train, y_train)
type(humidity_classifier)

#Predict on Test Set-----
y_pred = humidity_classifier.predict(X_test)
print("Sample Predictions: \n",y_pred[:10])
print("Actual Data: \n",y_test['high_humidity_label'][:10])

from sklearn import metrics

#cm = metrics.confusion_matrix(y_test,y_pred)
#print("CONFUSION MATRIX :  DECISION TREE\n",cm)


#Calculate the Model Accuracy , MAE, MSE , RMSE and R SQUARE SCORE-----

print("Accuracy Score: \n",metrics.accuracy_score(y_test,  y_pred))

mae = metrics.mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error = ", mae)

mse = metrics.mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)

rmse = mse ** 0.5
print("Root Mean Squared Error: ", rmse)

print("R Square Score: ", metrics.r2_score(y_test, y_pred))

'''
Accuracy Score: 0.8153409090909091
Mean Absolute Error =  0.1846590909090909
Mean Squared Error:  0.1846590909090909
Root Mean Squared Error:  0.4297197818452054
R Square Score:  0.2613397901533495
'''