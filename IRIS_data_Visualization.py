import pandas as pd
link="https://raw.githubusercontent.com/swapnilsaurav/Dataset/master/Iris.csv"
df = pd.read_csv(link)
print("Top 5 rows:\n",df.head())
print("Bottom 6 rows:\n",df.tail(6))
print("Size of the dataset (rows.columns):", df.shape)
print("Basic Statistics:\n",df.describe())
print("Columns and their data types:\n",df.info())
print("Checking for missing values:\n",df.isnull().sum())
print("Checking for Duplicate values:\n")
df_2 = df.drop_duplicates(subset="Species")
print("Size of the dataset after dup removal:", df_2.shape)
print("DF_2:\n",df_2)
print("Balancing the dataset:\n")
print(df.value_counts("Species"))

#### Performing visualization
#### EDA - Exploratory Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns
## pip install matplotlib seaborn
sns.countplot(data=df,x="Species")
plt.title("Count Plot - Species")
plt.show()

# find correlation between Petel length and Sepal length
sns.scatterplot(data=df,x="PetalLengthCm",y="SepalLengthCm")
plt.title("Scatterplot: PetalLength v SepalLength")
plt.show()

## Histogram
fig,axes=plt.subplots(2,2, figsize=(10,10))
axes[0,0].set_title('PetalLengthCm')
axes[0,0].hist(df['PetalLengthCm'],bins=10)

axes[0,1].set_title('SepalLengthCm')
axes[0,1].hist(df['SepalLengthCm'],bins=10)

axes[1,0].set_title('PetalWidthCm')
axes[1,0].hist(df['PetalWidthCm'],bins=10)

axes[1,1].set_title('SepalWidthCm')
axes[1,1].hist(df['SepalWidthCm'],bins=10)
plt.show()

## BOXPLOT
sns.boxplot(data=df,x="Species",y="PetalLengthCm")
plt.title("Boxplot: Species v PetalLengthCm")
plt.show()

