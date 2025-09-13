# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Set style for better looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Task 1: Load and Explore the Dataset
print("TASK 1: LOAD AND EXPLORE THE DATASET")
print("=" * 50)

try:
    # Load the Iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    
    # Display first few rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Explore dataset structure
    print("\nDataset information:")
    print(df.info())
    
    # Check for missing values
    print("\nMissing values in each column:")
    print(df.isnull().sum())
    
    # Since there are no missing values in the Iris dataset, we'll demonstrate cleaning with a hypothetical scenario
    # Let's create a copy with some missing values to demonstrate cleaning
    df_demo = df.copy()
    # Introduce some missing values for demonstration
    np.random.seed(42)
    mask = np.random.random(df_demo.shape) < 0.05  # 5% missing values
    df_demo = df_demo.mask(mask)
    
    print("\nAfter introducing some missing values for demonstration:")
    print("Missing values in each column:")
    print(df_demo.isnull().sum())
    
    # Clean the dataset by filling missing values with column means
    df_cleaned = df_demo.fillna(df_demo.mean(numeric_only=True))
    print("\nAfter filling missing values with column means:")
    print("Missing values in each column:")
    print(df_cleaned.isnull().sum())
    
    # For categorical columns, we might need a different approach
    # But in our case, the species column has no missing values
    
except Exception as e:
    print(f"Error loading dataset: {e}")

# Task 2: Basic Data Analysis
print("\n\nTASK 2: BASIC DATA ANALYSIS")
print("=" * 50)

# Compute basic statistics
print("Basic statistics of numerical columns:")
print(df.describe())

# Perform grouping on the species column and compute means
print("\nMean values for each measurement by species:")
species_means = df.groupby('species').mean()
print(species_means)

# Additional analysis - find the maximum values for each species
print("\nMaximum values for each measurement by species:")
species_max = df.groupby('species').max()
print(species_max)

# Identify patterns or interesting findings
print("\nINTERESTING FINDINGS:")
print("- Setosa species has significantly smaller petal measurements compared to other species")
print("- Versicolor and Virginica have overlapping but distinct measurements")
print("- Sepal width has the smallest variation across species")

# Task 3: Data Visualization
print("\n\nTASK 3: DATA VISUALIZATION")
print("=" * 50)

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Iris Dataset Analysis Visualizations', fontsize=16)

# 1. Line chart showing trends (using index as pseudo-time)
# For this demonstration, we'll sort by sepal length and use it as a trend
df_sorted = df.sort_values('sepal length (cm)')
axes[0, 0].plot(df_sorted['sepal length (cm)'], df_sorted['sepal width (cm)'], 
                label='Sepal Width', color='blue')
axes[0, 0].plot(df_sorted['sepal length (cm)'], df_sorted['petal length (cm)'], 
                label='Petal Length', color='green')
axes[0, 0].plot(df_sorted['sepal length (cm)'], df_sorted['petal width (cm)'], 
                label='Petal Width', color='red')
axes[0, 0].set_title('Trend of Measurements Relative to Sepal Length')
axes[0, 0].set_xlabel('Sepal Length (cm)')
axes[0, 0].set_ylabel('Measurement Value (cm)')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. Bar chart showing comparison of numerical values across categories
species_means.plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title('Average Measurements by Species')
axes[0, 1].set_xlabel('Species')
axes[0, 1].set_ylabel('Measurement Value (cm)')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Histogram of a numerical column
axes[1, 0].hist(df['sepal length (cm)'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Distribution of Sepal Length')
axes[1, 0].set_xlabel('Sepal Length (cm)')
axes[1, 0].set_ylabel('Frequency')

# 4. Scatter plot to visualize relationship between two numerical columns
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
for species, color in colors.items():
    species_data = df[df['species'] == species]
    axes[1, 1].scatter(species_data['sepal length (cm)'], 
                       species_data['petal length (cm)'], 
                       color=color, label=species, alpha=0.7)
axes[1, 1].set_title('Sepal Length vs Petal Length by Species')
axes[1, 1].set_xlabel('Sepal Length (cm)')
axes[1, 1].set_ylabel('Petal Length (cm)')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# Additional visualizations
print("\nAdditional Visualizations:")

# Box plot to show distribution of measurements by species
plt.figure(figsize=(12, 8))
df.boxplot(by='species', layout=(2, 2))
plt.suptitle('Distribution of Measurements by Species')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.show()

# Pairplot for comprehensive visualization
sns.pairplot(df, hue='species', palette=colors)
plt.suptitle('Pairplot of Iris Dataset by Species', y=1.02)
plt.show()

print("\nASSIGNMENT COMPLETE")
print("All tasks have been executed successfully!")
