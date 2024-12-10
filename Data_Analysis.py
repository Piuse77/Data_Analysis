
#step one

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
df = pd.read_csv(url)

# Display the first few rows of the dataset
df.head()

#step two
# Check the data types and missing values
df.info()

# Check for missing values
df.isnull().sum()

#TASK TWO (Basic Data Analysis)
# Compute basic statistics for numerical columns
df.describe()

# Grouping by species and computing the mean of numerical columns
df_grouped = df.groupby('species').mean()
df_grouped

#TASK 3 (Data Visualisation)
# Create a line chart showing trends in sepal length by species
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="species", y="sepal_length", marker='o')
plt.title('Trends of Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.show()


# Create a bar chart comparing average sepal length per species
plt.figure(figsize=(10, 6))
sns.barplot(x="species", y="sepal_length", data=df)
plt.title('Average Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Sepal Length (cm)')
plt.show()

# Create a histogram to visualize the distribution of sepal length
plt.figure(figsize=(10, 6))
sns.histplot(df['sepal_length'], bins=15, kde=True)
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()


# Create a scatter plot to visualize the relationship between petal length and petal width
plt.figure(figsize=(10, 6))
sns.scatterplot(x="petal_length", y="petal_width", data=df, hue="species")
plt.title('Petal Length vs. Petal Width')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(title='Species')
plt.show()

#Final Remarks:
#Each plot is customized with titles, axis labels, and legends to make the visuals informative.
#The dataset was loaded and explored to understand its structure.
#Basic analysis was conducted using .describe() and .groupby().
#Visualizations were created to provide insights into the distribution of data and relationships between features.
