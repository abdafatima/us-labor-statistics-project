# 📊 U.S. OCCUPATIONAL EMPLOYMENT AND WAGE ANALYSIS (OEWS 2023)

#### A Capstone Project for ADTA 5410 – Application and Deployment of Advanced Analytics  
**Team 5**: Naga Dheeraj Sudanagunta · Sai Dheeraj Sambaraju Narasimha Santosh · Abda Fatima Syeda · Akansha Karmarkar

---

## 🔍 Overview

This project focuses on analyzing the **Occupational Employment and Wage Statistics (OEWS)** dataset from the U.S. Bureau of Labor Statistics (BLS). The primary objective was to explore employment trends, wage distribution, and occupational insights to understand the American workforce better. We conducted extensive data cleaning, exploratory data analysis (EDA), and developed visual insights to assist in decision-making and workforce planning.

---

## 📂 About the Dataset

The OEWS dataset provides detailed employment and wage estimates for over 800 occupations across industries and geographic regions in the U.S. (as of May 2023). It includes median and mean wage estimates, employment levels, and percentile wage breakdowns. This dataset serves as a critical resource for economists, policymakers, job seekers, and workforce planners.

---

## ✅ Key Steps

### 1. **Data Acquisition**
- Imported the official OEWS dataset using the Pandas library
- Reviewed metadata and variable descriptions for meaningful analysis

### 2. **Data Cleaning**
- Removed missing and duplicate records
- Renamed columns for clarity and consistency
- Standardized data types (e.g., float conversion for numeric columns)
- Filtered out invalid occupation or region codes

### 3. **Exploratory Data Analysis (EDA)**
- **Descriptive Statistics**: Used `df.describe()` and grouped summaries to get a statistical overview
- **Univariate Analysis**: Visualized wage distributions using histograms and box plots
- **Bivariate/Multivariate Analysis**:
  - Analyzed wage variation across states and occupations
  - Scatter plots and bar graphs to compare employment levels and median wages
  - Heatmaps to identify regions with the highest average earnings
  - Correlation matrices to explore relationships between wage levels and employment volume

---

## 💡 Key Insights

- **Top-Paying Occupations**: Surgeons, anesthesiologists, and psychiatrists lead national wage rankings.
- **Industry Variations**: The tech and finance sectors show significantly higher median salaries compared to education and hospitality.
- **Regional Differences**: States like California, Massachusetts, and New York consistently rank high in occupational pay scales.
- **Employment Density**: Administrative, retail, and food service roles dominate in terms of employment count but are typically lower paid.

---

## 📊 Technologies Used
- **Programming Language**: Python
- **Libraries**: Pandas · NumPy · Seaborn · Matplotlib · Scikit-learn
- **Platform**: Jupyter Notebook
- **Version Control**: Git & GitHub

---

## 🧪 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/oews-wage-analysis.git
   cd oews-wage-analysis
   ```

2. Launch Jupyter Notebook and open `Final Project Report.ipynb`:
   ```bash
   jupyter notebook
   ```

3. Run all cells to generate results, visualizations, and insights.

---

## 👥 Acknowledgements

- **Bureau of Labor Statistics** for providing the OEWS dataset.
- Our team members for collaboration, analysis, and commitment to delivering impactful results.
