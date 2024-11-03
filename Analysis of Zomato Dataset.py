import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
file = "C:/Users/dubey/OneDrive/Desktop/test_zomato.csv"
df = pd.read_csv(file)

# Display basic information about the dataset
print("Columns Name", df.columns)
print("Top 10 rows:\n", df.head(10))
print("Size of the dataset (rows.columns):", df.shape)
print("Basic Statistics:\n", df.describe())
print("Columns and their data types:\n", df.info())
print("Checking for missing values:\n", df.isnull().sum())
# Remove missing Values:
print("Drop Missing Values ",df.dropna())
print("Drop Duplicates Values", df.drop_duplicates(inplace=True))

# Top 10 cities have the Most Expensive Restaurants-------------
# Convert 'Average Cost for two' to numeric, coercing errors to NaN
df['Average Cost for two'] = pd.to_numeric(df['Average Cost for two'], errors='coerce')
# Drop rows where 'Average Cost for two' is NaN
df = df.dropna(subset=['Average Cost for two'])
# Group by city and calculate the mean cost for two
top_expensive_cities = df.groupby('City')['Average Cost for two'].mean().sort_values(ascending=False).head(10)
print("Top 10 cities with the most expensive restaurants:\n", top_expensive_cities)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_expensive_cities.values, y=top_expensive_cities.index, palette="viridis")
plt.xlabel("Average Restaurant Cost")
plt.ylabel("City")
plt.title("Top 10 Cities with the Most Expensive Restaurants")
plt.show()

# Top 10 cities have Cheapest Restaurants-------------
# Group by city and calculate the mean cost for two, then sort in ascending order for cheapest restaurants
cheapest_cities = df.groupby('City')['Average Cost for two'].mean().sort_values(ascending=True).head(10)
print("Top 10 cities with the cheapest restaurants:\n", cheapest_cities)

plt.figure(figsize=(10, 6))
sns.barplot(x=cheapest_cities.values, y=cheapest_cities.index, palette="viridis")
plt.xlabel("Average Restaurant Cost")
plt.ylabel("City")
plt.title("Top 10 Cities with the Cheapest Restaurants")
plt.show()

# Top 10 cities have highest rated restaurants-------------
# Convert 'Aggregate Rating' (or 'Rating') to numeric, coercing errors to NaN
df['Aggregate rating'] = pd.to_numeric(df['Aggregate rating'], errors='coerce')
# Drop rows where 'Aggregate Rating' is NaN
df = df.dropna(subset=['Aggregate rating'])
# Group by city and calculate the mean rating, then sort in descending order for highest ratings
top_rated_cities = df.groupby('City')['Aggregate rating'].mean().sort_values(ascending=False).head(10)
print("Top 10 cities with the highest-rated restaurants:\n", top_rated_cities)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_rated_cities.values, y=top_rated_cities.index, palette="viridis")
plt.xlabel("Average Restaurant Rating")
plt.ylabel("City")
plt.title("Top 10 Cities with the Highest Rated Restaurants")
plt.show()

# Restaurants that do not provide a table booking facility----------
no_table_booking = df[df['Has Table booking'] == 'No']
print("Restaurants that do not provide a table booking facility:\n", no_table_booking[['Restaurant Name', 'City', 'Address']])

# Count of the Restaurants which provide Table Booking Facility
table_booking_counts = df['Has Table booking'].value_counts()

plt.figure(figsize=(8, 6))
sns.barplot(x=table_booking_counts.index, y=table_booking_counts.values, palette="viridis")
plt.xlabel("Table Booking Availability")
plt.ylabel("Number of Restaurants")
plt.title("Number of Restaurants Providing Table Booking Facility")
plt.show()

# Count the number of restaurants in each country code
country_counts = df['Country Code'].value_counts()
# Get the top 3 countries with the most restaurants
top_countries = country_counts.head(3)
print("Top 3 countries according to the number of restaurants:\n", top_countries)

plt.figure(figsize=(8, 6))
plt.pie(top_countries.values, labels=top_countries.index, autopct='%1.1f%%', startangle=140)
plt.title("Top 3 Country Code by Number of Restaurants")
plt.axis('equal')
plt.show()

# Less than 4/5 rating Restaurant in between index numbers 1400 to 1405(excluding)
# Convert 'Aggregate Rating' (or the relevant rating column) to numeric
df['Aggregate rating'] = pd.to_numeric(df['Aggregate rating'], errors='coerce')
# Filter for Indian restaurants (assuming the country code for India is 1)
indian_restaurants = df[df['Country Code'] == 1]
# Select rows between index 1400 and 1405 (excluding 1405)
indian_restaurants_subset = indian_restaurants.iloc[1400:1405]
# Further filter for restaurants with ratings less than 4
low_rated_restaurants = indian_restaurants_subset[indian_restaurants_subset['Aggregate rating'] < 4]
print("Indian restaurants with less than 4 ratings between index 1400 and 1405:\n", low_rated_restaurants[['Restaurant Name', 'Aggregate rating']])

# Most Expensive Restaurant in between index numbers 1400 to 1405(excluding)---------
# Convert 'Average Cost for two' to numeric, coercing errors to NaN
indian_restaurants_subset['Average Cost for two'] = pd.to_numeric(indian_restaurants_subset['Average Cost for two'], errors='coerce')
# Drop rows where 'Average Cost for two' is NaN
indian_restaurants_subset = indian_restaurants_subset.dropna(subset=['Average Cost for two'])
most_expensive_restaurant = indian_restaurants_subset.loc[indian_restaurants_subset['Average Cost for two'].idxmax()]
print("Most expensive Indian restaurant between index 1400 and 1405 (excluding 1405):\n")
print(most_expensive_restaurant[['Restaurant Name', 'City', 'Average Cost for two']])

# Cheapest Restaurant in between index numbers 1400 to 1405(excluding)
cheapest_restaurant = indian_restaurants_subset.loc[indian_restaurants_subset['Average Cost for two'].idxmin()]
print("Cheapest Indian Restaurant between index 1400 and 1405 (excluding 1405):")
print(cheapest_restaurant[['Restaurant Name','City', 'Average Cost for two']])





