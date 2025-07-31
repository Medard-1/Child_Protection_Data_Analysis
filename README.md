# Name: MUSINGUZI Medard
# ID: 26601

#### **Instructor**: Dr. Eric Maniraguha  
#### **Institution**: Adventist University of Central Africa  
#### **Course Name**: Introduction to Big Data Analytics  
#### **Course Code**: INSY 8413  
#### **Date**:31st , July, 2025

---

# 📘 Child Welfare Vulnerability Mapping Project

This project is a comprehensive application of Big Data Analytics focused on child welfare. Leveraging data from UNICEF, it aims to uncover patterns of vulnerability among children to better support the work of humanitarian organizations such as Save the Children. The project integrates data cleaning, exploratory analysis, machine learning (clustering), and visualization using Power BI.

---

## 🎯 Project Goal
To analyze child protection indicators and use data-driven methods to identify patterns and high-risk groups, supporting NGOs and other sectors in making targeted child protection interventions.

---

## 🔍 Problem Definition
Can we identify vulnerable groups of children and map areas of high risk using big data techniques applied to socio-demographic and protection indicators? This project seeks to answer that using real-world data through clustering, visualization, and interactivity.

---

## 📂 Dataset Overview
- **Dataset Title:** UNICEF Child Protection Indicators – Multi-Country Dataset
- **Source:** [UNICEF Data Portal](https://data.unicef.org/topic/child-protection/)
- **Format:** CSV
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
![<img width="970" height="562" alt="bar plot" src="https://github.com/user-attachments/assets/6a619004-4b17-4aa8-aa4f-16689f275277" />
]
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
![<img width="970" height="562" alt="bar plot" src="https://github.com/user-attachments/assets/8c075fd8-f366-4804-8816-85832dedb182" />
]
```python
```

