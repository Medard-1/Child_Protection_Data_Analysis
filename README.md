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
##Output
![<img width="526" height="397" alt="cleaned dat preview" src="https://github.com/user-attachments/assets/f09aadb7-a61e-44e9-9ef5-8a943c1b152f" />]

