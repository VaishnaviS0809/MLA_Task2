# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Step 2: Load the dataset
df = pd.read_csv('titanic.csv')

# Step 3: Basic info and summary stats
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Step 4: Univariate Analysis - Histograms & Boxplots
numeric_cols = ['Age', 'Fare']
for col in numeric_cols:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram of {col}')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()

# Step 5: Pairplot and Correlation Matrix
sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']].dropna(), hue='Survived')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df[['Age', 'Fare', 'Pclass', 'Survived']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Step 6: Skewness Detection
print("Skewness:\n", df[numeric_cols].skew())

# Step 7: Optional - Interactive Plotly Visual
fig = px.histogram(df, x='Age', color='Survived', nbins=30, title='Age Distribution by Survival')
fig.show()
