file = "C:/Users/dubey/Downloads/hotel_bookings.csv"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(file)
print("Size of dataset:",df.shape)
print("Datatypes of dataset:\n",df.dtypes)

df_numeric = df.select_dtypes(include=[np.number])
df_objects = df.select_dtypes(exclude=[np.number])
print("Number columns:\n",df_numeric)
print("Object columns:\n",df_objects)

# Missing Value Visualization:
plt.style.use("ggplot")
sns.heatmap(data=df.isnull(),cmap= sns.color_palette(['#F7CB06','#EE4B2B']))
plt.show()

# go through each column and find the number of missing values
print("1. checking all missing values")
for col in df.columns:
    pt_miss = np.mean(df[col].isnull())*100
    if pt_miss >0.0:
        print(f"{col} - {pt_miss}")
# columns with more than 70% missing:
print("3. checking again after dropping columns")
for col in df.columns:
    pt_miss = np.mean(df[col].isnull())*100
    if pt_miss > 0.0:
        print(f"{col} - {pt_miss}")
        df[f'{col}_Missing']=df[col].isnull()

# now lets turn our attention to the rows
ismiss_cols = [col for col in df.columns if '_Missing' in col]
df['Num_Missing'] = df[ismiss_cols].sum(axis=1)
df['Num_Missing'].value_counts().sort_index().plot.bar(x='index', y="Num_Missing")
plt.show()


# columns with more than 70% missing:
print("2. checking columns >70% missing")
for col in df.columns:
    pt_miss = np.mean(df[col].isnull())*100
    if pt_miss >70.0:
        print(f"{col} - {pt_miss}")

#company is the result :
col_to_drop = ['company']
# axis = 0 - look for the values in the rows
# axis = 1 - look for the values in the columns
df = df.drop(col_to_drop,axis=1)

ind_missing = df[df['Num_Missing']>12].index
print("Rows with more than 10 missing values:\n",ind_missing)
df = df.drop(ind_missing,axis=0)

## after dropping rows and columns check again for missing values:
print("11. checking all missing values")
for col in df.columns: # ['A','B','C']
    pt_miss = np.mean(df[col].isnull())*100
    if pt_miss > 0.0:
        print(f"{col} - {pt_miss}")

'''
children - 2.0498257606219004 - numeric
babies - 11.311318858061922 - numeric
meal - 11.467129071170085 - categorical
country - 0.40879238707947996 - categorical
deposit_type - 8.232810615199035 - categorical
agent - 13.687005763302507 - categorical
'''
#Missing numeric values in children and babies are replaced with their mean
avg = df['children'].mean()
df['children'] = df['children'].fillna(avg)
avg = df['babies'].mean()
df['babies'] = df['babies'].fillna(avg)
#replace with specific values

#covert them into categorical values
df['agent'] = pd.Categorical(df.agent)
df['meal'] = pd.Categorical(df.meal)
df['country'] = pd.Categorical(df.country)
df['deposit_type'] = pd.Categorical(df.deposit_type)

#lets recreate numeric and object set pf columns
df_numeric = df.select_dtypes(include=[np.number])
df_objects = df.select_dtypes(exclude=[np.number])

#replacing categorical values
for col in df.columns:
    #if col is non numeric , perform below code
    if col in df_objects:
        num_miss = np.sum(df[col].isnull())
        if num_miss > 0:
           #calculate mode and replace with mode
           top = df[col].describe()['top']
           df[col] = df[col].fillna(top)
        # if col is in numeric , perform below code
        if col in df_numeric:
            num_miss = np.sum(df[col].isnull())
            if num_miss > 0:
                # calculate median and replace with mode
                med = df[col].median()
                df[col] = df[col].fillna(med)
## after handling missing values:
print("111. checking all missing values")
for col in df.columns: # ['A','B','C']
    pt_miss = np.mean(df[col].isnull())*100
    if pt_miss > 0.0:
        print(f"{col} - {pt_miss}")

print("Size of the dataset:",df.shape)

# delete those extra add columns - Missing
ismiss_cols = [col for col in df.columns if '_Missing' in col]
df = df.drop(ismiss_cols,axis=1)

print("Size of the dataset after dropping_missing:",df.shape)

#Outliers - identifying outliers and fixing them
#fixing outliers are very similar to handling missing values
'''
Identify Outliers :
Histograms and box plots are used to visualize outliers in children and total_of_special_requests
'''
#histogram
df['children'].hist(bins=5)
plt.title('Children')
plt.show()

print(df['total_of_special_requests'].hist(bins=100))
plt.title('Total_of_special_requests')
plt.show()
#boxplot
df.boxplot(column=['total_of_special_requests'])
plt.show()
#descriptive statistics
print(df['total_of_special_requests'].describe())



'''
This project focuses on analyzing and cleaning a hotel bookings dataset using Python and key data visualization 
libraries like Matplotlib and Seaborn. The dataset is initially explored for its structure, missing values, and data types.
A heatmap is used to visualize missing data patterns, followed by filling in missing numeric and categorical values through
mean and mode imputation. The project also detects outliers using histograms and boxplots, providing insights into key variables
like children and total_of_special_requests. The goal of the project is to prepare the dataset for further analysis by
handling missing data, outliers, and duplicates, while ensuring data integrity.
'''
