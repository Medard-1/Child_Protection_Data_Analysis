# **Name**: MUSINGUZI Medard
# **ID**: 26601

#### **Instructor**: Dr. Eric Maniraguha  
#### **Institution**: Adventist University of Central Africa  
#### **Course Name**: Introduction to Big Data Analytics  
#### **Course Code**: INSY 8413  
#### **Date**: 31st , July, 2025

---

# 📘 Child Welfare Vulnerability Mapping Project

This project is a comprehensive application of Big Data Analytics focused on child welfare. Leveraging data from UNICEF, it aims to uncover patterns of vulnerability among children to better support the work of humanitarian organizations such as Save the Children. The project integrates data cleaning, exploratory analysis, machine learning (clustering), and visualization using Power BI.

---

## 🎯 Project Goal
To analyze child protection indicators and use data-driven methods to identify patterns and high-risk groups, supporting NGOs and other sectors in making targeted child protection interventions.

---

## 🔍 Problem Definition
Can we identify geographic patterns of child vulnerability using socio-economic and protection-related data to guide targeted child welfare interventions?? This project seeks to answer that using real-world data through clustering, visualization, and interactivity.

---

## 📂 Dataset Overview
- **Dataset Title:** UNICEF Child Protection Indicators – Multi-Country Dataset
- **Source:** [UNICEF Data Portal](https://data.unicef.org/topic/child-protection/)
- **Format:** CSV
- **Rows:** 4080
- **Columns:** 17 including Indicator, Age, Sex, Residence, Time Period, and Observation Value
- **Issues:** Missing values, inconsistent formats, categorical fields

---

## 🧠 PART 2: Python Analytics Tasks

### 1️⃣ Data Cleaning
```python
    import pandas as pd
    def load_data(filepath=r'ChildProtection_UNICEF.csv'):
         """Load dataset with correct encoding."""
         df = pd.read_csv(filepath, encoding='latin1')
         print(" Dataset loaded successfully.")
         return df
```
```python

df = load_data()
def clean_data(df):
    # Drop columns with too many missing values
    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)
    
    # Fill remaining missing values
    df = df.fillna("Unknown")
    
    # Strip column names and standardize 'Country' and 'Year'
    df.columns = df.columns.str.strip()
    if 'Country' in df.columns:
        df['Country'] = df['Country'].astype(str).str.strip().str.title()
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    return df

clean_data(df)
```
### Output
![<img width="1023" height="560" alt="Loaded Cleaned dataset output" src="https://github.com/user-attachments/assets/fdf94cee-68a0-4ba5-a762-7f09cc90c9f5" />
]

## 2️⃣ Exploratory Data Analysis (EDA)
```python
   def perform_eda(df, numeric_column):
      """Generate basic statistics and visualizations."""
      print(" Descriptive Statistics:")
      print(df[numeric_column].describe())
  # Describe numeric values
  print(" Descriptive Statistics for 'OBS_VALUE'")
  print(df['OBS_VALUE:Observation Value'].describe())

  # Distribution of indicators
  print("\n Number of records per Indicator:")
  print(df['INDICATOR:Indicator'].value_counts())

  # Optional: Group by indicator and get stats
  print("\n Average OBS_VALUE per Indicator:")
  print(df.groupby('INDICATOR:Indicator')['OBS_VALUE:Observation Value'].mean().sort_values(ascending=False))
```
### Output
![<img width="400" height="178" alt="image" src="https://github.com/user-attachments/assets/68627d75-7c65-435a-b8b2-ce7ecd098fa0" />
]
```python
# Histogram
def perform_eda(df, numeric_column):
    """Generate basic statistics and visualizations."""
    print(" Descriptive Statistics:")
    print(df[numeric_column].describe())
import matplotlib.pyplot as plt
import seaborn as sns
def perform_eda(df, numeric_column):
    """Generate basic statistics and visualizations."""
    print(" Descriptive Statistics:")
    print(df[numeric_column].describe())

plt.figure(figsize=(10, 5))
sns.histplot(df['OBS_VALUE:Observation Value'], bins=30, kde=True)
plt.title("Distribution of Observation Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

```
### Output
![<img width="898" height="492" alt="Histogram" src="https://github.com/user-attachments/assets/17b205c7-f4d5-4e40-b365-8462e0789500" />
]
```python
def perform_eda(df, numeric_column):
    """Generate basic statistics and visualizations."""
    print(" Descriptive Statistics:")
    print(df[numeric_column].describe())
# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='SEX:Sex', y='OBS_VALUE:Observation Value', data=df)
plt.title("Observation Value by Sex")
plt.show()
```
### Output
![<img width="874" height="565" alt="boxplot1" src="https://github.com/user-attachments/assets/9d2bb00b-fa41-4285-8a8b-f021166c9811" />
]
```python
def perform_eda(df, numeric_column):
    """Generate basic statistics and visualizations."""
    print(" Descriptive Statistics:")
    print(df[numeric_column].describe())
#Residence, Wealth Quintile, or Education Level:
plt.figure(figsize=(12, 6))
sns.boxplot(x='RESIDENCE:Residence', y='OBS_VALUE:Observation Value', data=df)
plt.title("Observation Value by Urban vs Rural")
plt.show()
```
### Output
![<img width="964" height="531" alt="boxplot2" src="https://github.com/user-attachments/assets/c5ea35a9-c080-4f60-8d0d-134a7c01c6b5" />
]
```python
def perform_eda(df, numeric_column):
    """Generate basic statistics and visualizations."""
    print(" Descriptive Statistics:")
    print(df[numeric_column].describe())
#Bar Plot – mean values by indicator
indicator_stats = df.groupby('INDICATOR:Indicator')['OBS_VALUE:Observation Value'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
indicator_stats.plot(kind='bar')
plt.title("Average Observation Value by Indicator")
plt.ylabel("Average Value")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

```
### Output
![<img width="970" height="562" alt="bar plot" src="https://github.com/user-attachments/assets/6a619004-4b17-4aa8-aa4f-16689f275277" />]

```python
def perform_eda(df, numeric_column):

    print(" Descriptive Statistics:")
    print(df[numeric_column].describe())
#Heatmap (Optional) – correlation matrix (if you have multiple numeric columns)
# If there are other numeric columns after cleaning
numeric_df = df.select_dtypes(include=['float64', 'int64'])

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
```
![
<img width="966" height="538" alt="correlation matrix" src="https://github.com/user-attachments/assets/b9554a63-f601-47b3-9cdf-3e29449bd1c8" />
]
- I applied Unsupervised Machine Learning: Clustering
This fits my goal of:**“Mapping vulnerability: identifying high-risk areas for child protection interventions.”**

#### **✅ Why Clustering?**
- My data does not have a label/target column (like "Risk Level = High/Low"), so classification/regression is not ideal.
- Clustering will help group observations (e.g., by region, sex, age group) into similar vulnerability profiles.
  
```python
def run_kmeans(df, features, k=3):
    """Apply KMeans clustering and return labeled data and feature matrix."""
    df_copy = df.copy()

    # Encode categorical features
    for col in features:
        if df_copy[col].dtype == 'object':
            df_copy[col] = LabelEncoder().fit_transform(df_copy[col].astype(str))

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(df_copy[features])

    # Train KMeans
    model = KMeans(n_clusters=k, random_state=42)
    df_copy['Cluster'] = model.fit_predict(X)

    return df_copy, X
```
```python
# ...existing code...
from sklearn.preprocessing import LabelEncoder, StandardScaler
# ...existing code...
```
```python
# Define the features you want to use for clustering
features = [
    'OBS_VALUE:Observation Value', 
    'SEX:Sex', 
    'RESIDENCE:Residence'
    # Add more relevant features if needed
]

# Prepare data and run KMeans
df_cluster, X = run_kmeans(df, features, k=3)

print(" Clustering complete!")
print(df_cluster[['Cluster'] + features].head())
# ...existing code...
```
### Output
![<img width="590" height="158" alt="image" src="https://github.com/user-attachments/assets/ed33ec54-ac18-4f88-b3ef-ba0262dd27d5" />
]
```python
#Visualize the Clusters (Optional)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=df_cluster['Cluster'], palette='viridis')
plt.title("K-Means Clustering of Vulnerability Profiles")
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.show()

```
### Output
![<img width="731" height="572" alt="clustering 1" src="https://github.com/user-attachments/assets/260c3aa8-8ddc-4ea6-8713-dfb73d994fd0" />
]
```python
# Compute Silhouette Score
#It measures how well each data point fits within its cluster — higher scores mean better clustering.

from sklearn.metrics import silhouette_score

# X is your scaled feature matrix
score = silhouette_score(X, df_cluster['Cluster'])

print(f" Silhouette Score for K-Means Clustering: {score:.3f}")
```
### Output
![<img width="367" height="27" alt="image" src="https://github.com/user-attachments/assets/e3f0ca66-f885-4fd6-819c-1adde40b43a3" />
]
```python
# Visualize Silhouette Score for Multiple k Values (Optional)
from sklearn.metrics import silhouette_score

scores = []
K = range(2, 10)

for k in K:
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    scores.append(score)

# Plot the silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(K, scores, marker='o')
plt.title("Silhouette Score vs Number of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()
```
### Output
![<img width="747" height="501" alt="silhouette" src="https://github.com/user-attachments/assets/a00b44d1-882d-4f4c-8196-ea1ad3688929" />
]
#### ❓ What About Accuracy, Precision, RMSE?
These metrics are for supervised learning (like classification or regression) and require ground truth labels, which we don’t have in clustering. So, in your project:
- ✅ Silhouette Score was the  best tool
- ❌ Accuracy / Precision / RMSE are not applicable
- 
# POWER BI DASHBOARD TASKS
## 🧩 Step 1: Communicate the Problem & Insights Clearly
-Many children are exposed to risks such as child labor, psychological violence, or lack of legal protection. However, these cases are often underreported or scattered across regions, making it hard to know where interventions are most urgently needed.
Using big data analytics, this project answers:
“Where are children most at risk, and what patterns can we uncover in their protection needs using data?”
### Combined Charts created
![<img width="896" height="508" alt="Combined charts" src="https://github.com/user-attachments/assets/e28b944b-950e-4c7d-a9ed-0cf7ae0bb45c" />]
#### Slicers applied
![<img width="684" height="485" alt="Slicers" src="https://github.com/user-attachments/assets/8cbd96fc-92c3-4f37-9d4d-699123f1a786" />
]
### Dashboboard
-This dashboard supports the detection of high-risk child populations using real-world data and clustering models. It enables visual exploration of patterns in child protection indicators such as child labor, psychological violence, and birth registration accordingly.
#### Main dashboard
![<img width="894" height="506" alt="Dashboard1" src="https://github.com/user-attachments/assets/b1f00c18-2b88-4c2d-bab0-7e92973e38d4" />
]
#### Dashboard with a Map
![<img width="889" height="503" alt="Dashboard 2" src="https://github.com/user-attachments/assets/3e5948a4-dcb0-4ef2-baaa-933d7d1806f7" />
]
# THANK YOU !









