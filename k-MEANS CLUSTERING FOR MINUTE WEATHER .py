# PROJECT 7: k-MEANS CLUSTERING USING SCIKIT-LEARN IN PYTHON

#Dataset : Minute Weather---
'''
The minute weather dataset contains raw sensor measurements captured at one-minute
intervals.The data is in the file minute_weather.csv, which is a comma-separated file.
Each row in minute_weather.csv contains weather data captured for a one-minute interval. Each
row, or sample, consists of the following variables:
o rowID: unique number for each row (Unit: NA)
o hpwren_timestamp: timestamp of measure (Unit: year-month-day hour:minute:second)
o air_pressure: air pressure measured at the timestamp (Unit: hectopascals)
o air_temp: air temperature measure at the timestamp (Unit: degrees Fahrenheit)
o avg_wind_direction: wind direction averaged over the minute before the timestamp (Unit:
degrees, with 0 means coming from the North, and increasing clockwise)
o avg_wind_speed: wind speed averaged over the minute before the timestamp (Unit: meters
per second)
o max_wind_direction: highest wind direction in the minute before the timestamp (Unit:
degrees, with 0 being North and increasing clockwise)
o max_wind_speed: highest wind speed in the minute before the timestamp (Unit: meters per
second)
o min_wind_direction: smallest wind direction in the minute before the timestamp (Unit:
degrees, with 0 being North and inceasing clockwise)
o min_wind_speed: smallest wind speed in the minute before the timestamp (Unit: meters
per second)
o rain_accumulation: amount of accumulated rain measured at the timestamp (Unit:
millimeters)
o rain_duration: length of time rain has fallen as measured at the timestamp (Unit: seconds)
o relative_humidity: relative humidity measured at the timestamp (Unit: percent)
'''
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from itertools import cycle, islice
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
data = pd.read_csv("C:/Users/dubey/Downloads/data (1)/data/weather.zip", compression='zip')
print("Data Shape:",data.shape)
print("Columns of the Data:",data.columns)
print("Sample Data:\n",data.head())
#Lots of rows, so let us sample down by taking every 10th row.
sampled_df =data[(data['rowID'] % 10) == 0]
print("Sampled Data Shape:", sampled_df.shape)

#Perform Some Statistics--
print("Sampled Data Describe:", sampled_df.describe().transpose())
print("Rain Accumulation Data Shape", sampled_df[sampled_df["rain_accumulation"] == 0].shape)
print("Rain Duration Data Shape", sampled_df[sampled_df["rain_duration"]== 0].shape)

#Drop all the Rows with Empty rain_duration and rain_accumulation
del sampled_df['rain_accumulation']
del sampled_df['rain_duration']
rows_before = sampled_df.shape[0]
sampled_df = sampled_df.dropna()
rows_after = sampled_df.shape[0]
#How many rows did we drop ?
print("How many rows did we drop ? ",rows_before - rows_after)
print("Columns in Sample Data after delete empty rows: ",sampled_df.columns)


#Select Features of Interest for Clustering
features = ['air_pressure', 'air_temp', 'avg_wind_direction','avg_wind_speed',
            'max_wind_direction','max_wind_speed','relative_humidity']
select_df = sampled_df[features]
print("Features Columns:", select_df.columns)
print("Features Data" ,select_df)

#Scale the Features using StandardScaler
X = StandardScaler().fit_transform(select_df)
print("Scaling the Features:\n", X)

#Using k-Means Clustering------
kmeans = KMeans(n_clusters= 12)
model = kmeans.fit(X)
print("Model:\n", model)

#What are the centers of 12 clusters we formed ?
centers = model.cluster_centers_
print("Centers: ", centers)

#Plots----
#Create some utility functions which will help us in plotting graphs:
# Function that creates a DataFrame with a column for Cluster Number

def pd_centers(featuresUsed, centers):
    colNames = list(featuresUsed)
    colNames.append('prediction')
    # Zip with a column called 'prediction' (index)-----
    Z = [np.append(A, index) for index, A in enumerate(centers)]

    # Convert to pandas data frame for plotting........
    P = pd.DataFrame(Z, columns=colNames)
    P['prediction'] = P['prediction'].astype(int)
    return P

# Function that creates Parallel Plots------

def parallel_plot(data):
    my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None,len(data)))
    plt.figure(figsize=(15, 8)).gca().axes.set_ylim([-3, +3])
    parallel_coordinates(data, 'prediction', color=my_colors, marker='o')
    plt.show()
P = pd_centers(features, centers)
print("pd_centers: ",P)

#Dry Days-----
parallel_plot(P[P['relative_humidity'] < -0.5])

#Warm Days----
parallel_plot(P[P['air_temp'] > 0.5])

#Cool Days-----
parallel_plot(P[(P['relative_humidity'] > 0.5) & (P['air_temp'] <0.5)])

'''
Unsupervised learning technique, which doesn't use labeled data (true labels or targets) to evaluate model performance....
metrics like Accuracy Score, MAE, MSE, RMSE, and R-Square Score are not directly applicable to k-Means clustering....
'''
from sklearn.metrics import silhouette_score, davies_bouldin_score
# Inertia: Sum of squared distances of samples to their closest cluster center
inertia = model.inertia_
print("Inertia: ", inertia)

# Silhouette Score: Measures the separation distance between the resulting clusters
silhouette_avg = silhouette_score(X,model.labels_)
print("Silhouette Score: ", silhouette_avg)

# Davies-Bouldin Index: Average similarity ratio of each cluster with its most similar cluster
davies_bouldin = davies_bouldin_score(X,model.labels_)
print("Davies-Bouldin Index: ", davies_bouldin)

'''
Inertia:  319279.7856233307
Silhouette Score:  0.21890744964966605
Davies-Bouldin Index:  1.3726176389085067

1. Inertia: Lower inertia indicates that points are closer to the centroids.
2. Silhouette Score: A score closer to 1 to -1 (closer to 1 is good cluster and closer to -1 is wrong cluster indicates) 
3. Davies-Bouldin Index: Lower values indicate better clustering with less overlap between clusters
'''

