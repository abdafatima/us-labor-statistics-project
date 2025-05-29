#!/usr/bin/env python
# coding: utf-8

# <center><h1 style="text-align: center; font-family: 'Times New Roman', sans-serif; color: blue; font-size: 40px;">FINAL PROJECT REPORT</h1></center>

# <center><h1 style="text-align: center; font-family: 'Times New Roman', sans-serif; color: green; font-size: 40px;">ADTA 5410 SECTION 003 - APPLICATION AND DEPLOYMENT OF ADVANCED ANALYTICS</h1></center>

# # Group 5
# ### Naga Dheeraj Sudanagunta
# ### Sai Dheeraj Sambaraju Narasimha Santosh
# ### Abda Fatima Syeda
# ### Akansha Karmarkar
# 

# <center><h2>Introduction</h2></center>

# The Occupational Employment and Wage Statistics (OEWS) is a program of BLS that generates annual employment and wage data by occupation. May 2023 also creates detailed employment distribution, median wages, and percentile wage estimates for regions and industries across the U.S. is a thorough one of how Americans work, get paid, and contribute to America’s economy.
# 
# As with any form of data information, this set of data is a valuable asset to workforce planning, policy making and, in general, economic analysis. Earlier research has described how analytics has been used for measures of wages differentiation, employment relations configuration, and designing education and training interventions for skills mismatch. For example, prior work using the OEWS data have investigated aspects of wage discontinuity and the relationship between wages and industry growth. This dataset does exactly that by giving a first ever glimpse into the complex reality of jobs and wages. It’s a street plan that not only reveals where individuals are employed, but how much they are paid and the sectors they belong to, as well as spreading across different areas.
# 

# <center><h2>Abstract</h2></center>

# This report seeks to analyze wage issues in the United States using analytical data. Some of the steps included data cleaning, feature engineering and data exploration more specifically, we looked at the factors that impact wages. This also reveals significant patterns and offers enriched information about wage disparity by industries and occupations.

# <center><h2>Data Introduction and Description</h2></center>

# 
# | **Field**       | **Field Description**                                                                                                                                                                                                 |
# |-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# | **area**        | U.S. (99), state FIPS code, Metropolitan Statistical Area (MSA) or New England City and Town Area (NECTA) code, or OEWS-specific nonmetropolitan area code                                                             |
# | **area_title**  | Area name                                                                                                                                                                                                              |
# | **area_type**   | Area type: 1= U.S.; 2= State; 3= U.S. Territory; 4= Metropolitan Statistical Area (MSA) or New England City and Town Area (NECTA); 6= Nonmetropolitan Area                                                              |
# | **prim_state**  | The primary state for the given area. "US" is used for the national estimates.                                                                                                                                          |
# | **naics**       | North American Industry Classification System (NAICS) code for the given industry                                                                                                                                       |
# | **naics_title** | North American Industry Classification System (NAICS) title for the given industry                                                                                                                                      |
# | **i_group**     | Industry level. Indicates cross-industry or NAICS sector, 3-digit, 4-digit, 5-digit, or 6-digit industry. For industries that OEWS no longer publishes at the 4-digit NAICS level, the "4-digit" designation indicates the most detailed industry breakdown available: either a standard NAICS 3-digit industry or an OEWS-specific combination of 4-digit industries. Industries that OEWS has aggregated to the 3-digit NAICS level (for example, NAICS 327000) will appear twice, once with the "3-digit" and once with the "4-digit" designation. |
# | **own_code**    | Ownership type: 1= Federal Government; 2= State Government; 3= Local Government; 123= Federal, State, and Local Government; 235=Private, State, and Local Government; 35 = Private and Local Government; 5= Private; 57=Private, Local Government Gambling Establishments (Sector 71), and Local Government Casino Hotels (Sector 72); 58= Private plus State and Local Government Hospitals; 59= Private and Postal Service; 1235= Federal, State, and Local Government and Private Sector |
# | **occ_code**    | The 6-digit Standard Occupational Classification (SOC) code or OEWS-specific code for the occupation                                                                                                                    |
# | **occ_title**   | SOC title or OEWS-specific title for the occupation                                                                                                                                                                     |
# | **o_group**     | SOC occupation level. For most occupations, this field indicates the standard SOC major, minor, broad, and detailed levels, in addition to all-occupations totals. For occupations that OEWS no longer publishes at the SOC detailed level, the "detailed" designation indicates the most detailed data available: either a standard SOC broad occupation or an OEWS-specific combination of detailed occupations. Occupations that OEWS has aggregated to the SOC broad occupation level will appear in the file twice, once with the "broad" and once with the "detailed" designation. |
# | **tot_emp**     | Estimated total employment rounded to the nearest 10 (excludes self-employed).                                                                                                                                           |
# | **emp_prse**    | Percent relative standard error (PRSE) for the employment estimate. PRSE is a measure of sampling error, expressed as a percentage of the corresponding estimate. Sampling error occurs when values for a population are estimated from a sample survey of the population, rather than calculated from data for all members of the population. Estimates with lower PRSEs are typically more precise in the presence of sampling error. |
# | **jobs_1000**   | The number of jobs (employment) in the given occupation per 1,000 jobs in the given area. Only available for the state and MSA estimates; otherwise, this column is blank.                                                |
# | **loc_quotient**| The location quotient represents the ratio of an occupation’s share of employment in a given area to that occupation’s share of employment in the U.S. as a whole. For example, an occupation that makes up 10 percent of employment in a specific metropolitan area compared with 2 percent of U.S. employment would have a location quotient of 5 for the area in question. Only available for the state, metropolitan area, and nonmetropolitan area estimates; otherwise, this column is blank. |
# | **pct_total**   | Percent of industry employment in the given occupation. Percents may not sum to 100 because the totals may include data for occupations that could not be published separately. Only available for the national industry estimates; otherwise, this column is blank. |
# | **pct_rpt**     | Percent of establishments reporting the given occupation for the cell. Only available for the national industry estimates; otherwise, this column is blank.                                                               |
# | **h_mean**      | Mean hourly wage                                                                                                                                                                                                        |
# | **a_mean**      | Mean annual wage                                                                                                                                                                                                        |
# | **mean_prse**   | Percent relative standard error (PRSE) for the mean wage estimate. PRSE is a measure of sampling error, expressed as a percentage of the corresponding estimate. Sampling error occurs when values for a population are estimated from a sample survey of the population, rather than calculated from data for all members of the population. Estimates with lower PRSEs are typically more precise in the presence of sampling error. |
# | **h_pct10**     | Hourly 10th percentile wage                                                                                                                                                                                             |
# | **h_pct25**     | Hourly 25th percentile wage                                                                                                                                                                                             |
# | **h_median**    | Hourly median wage (or the 50th percentile)                                                                                                                                                                              |
# | **h_pct75**     | Hourly 75th percentile wage                                                                                                                                                                                             |
# | **h_pct90**     | Hourly 90th percentile wage                                                                                                                                                                                             |
# | **a_pct10**     | Annual 10th percentile wage                                                                                                                                                                                             |
# | **a_pct25**     | Annual 25th percentile wage                                                                                                                                                                                             |
# | **a_median**    | Annual median wage (or the 50th percentile)                                                                                                                                                                              |
# | **a_pct75**     | Annual 75th percentile wage                                                                                                                                                                                             |
# | **a_pct90**     | Annual 90th percentile wage                                                                                                                                                                                             |
# | **annual**      | Contains "TRUE" if only annual wages are released. The OEWS program releases only annual wages for some occupations that typically work fewer than 2,080 hours per year, but are paid on an annual basis, such as teachers, pilots, and athletes. |
# | **hourly**      | Contains "TRUE" if only hourly wages are released. The OEWS program releases only hourly wages for some occupations that typically work fewer than 2,080 hours per year and are paid on an hourly basis, such as actors, dancers, and musicians and singers. |
# 

# Notes:	
# 
# - `*` = indicates that a wage estimate is not available
# - `**`  = indicates that an employment estimate is not available	
# - `#`  = indicates a wage equal to or greater than $115.00 per hour or $239,200 per year 	
# - `~` =indicates that the percent of establishments reporting the occupation is less than 0.5%	
# 

# <center><h2>Understanding the Dataset</h2></center>

# The dataset included several characteristics of the working force. And here is the data of that occupation sector  
# - **Geography**: Places, positions and provinces where workers are likely to be hired.  
# - **Industry**: Economists’ classification codes and titles based on the various industries.  
# - **Occupations**: Designations, positions, job numbers and classification into groups.  
# - **Employment**: The overall employment statistics and employments ratios.  
# - **Wages**: Hourly and annual mean wages as well as selected percentiles (10 th , 25 th , 50 th , and so on).

# ### Importing Necessary libraries and Data Loading
# This is to emphasize that in order to properly assess the results of the analyzed data, we used the most widespread Python packages. These tools helped us clean, process, and visualize the data:  
# - **Pandas**: managed large data nicely.  
# - **Matplotlib** and **Seaborn**: Some fun facts and statistics steroids made real through beautiful use of charts and graphs.  
# - **Numpy**, **Scipy**, and **Statsmodels**: Use of numerical and statistical means in its powering.  
# - **Sklearn**: Combined support for data transformation and therefore modeling.

# In[3]:


# Data Analysis: Step-by-Step Approach
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
import statsmodels.api as sm


# In[11]:


# Step 1: Data Introduction and Description
# Load the entire dataset and provide an introduction to the dataset
file_path = "all_data_M_2023.xlsx"
data_df_full = pd.read_excel(file_path)


# In[13]:


# Provide basic information about the dataset
print("Dataset Information:")


# In[15]:


data_df_full.info()


# The dataset of **413327 rows** and **32 variables** depicting wages, employment, industries and locations.

# In[17]:


data_df_full.describe()


# # Question of Interest
# - Question 1: What factors are most strongly correlated with hourly and annual wages?
# - Question 2: Can we predict wage levels based on features such as occupation group, area, and employment statistics?
# 

# # Data Understanding and EDA

# In[21]:


# Step 3: Drop columns with mostly missing values
# We remove columns that have many missing values and aren't useful for our analysis.
columns_to_drop = ['JOBS_1000', 'LOC_QUOTIENT', 'PCT_TOTAL', 'PCT_RPT', 'ANNUAL', 'HOURLY']
cleaned_data_df = data_df_full.drop(columns=columns_to_drop)
cleaned_data_df.replace({'*': np.nan, '**': np.nan, '~': np.nan, '': np.nan}, inplace=True)


# Focusing on Relevant Details
# For ease of comparison, some columns that were deemed unimportant for wage and employment were dropped; JOBS_1000 and LOC_QUOTIENT.
# 
# Delivering Data Irregularities:
# There would be some special symbols in the wage columns (*. **, ~*), which were changed to NaN.
# 

# In[24]:


# For hourly wages
wage_cap_hourly = 115.00  # $115 per hour
cleaned_data_df['H_MEAN'] = cleaned_data_df['H_MEAN'].replace('#', wage_cap_hourly)
cleaned_data_df['H_PCT10'] = cleaned_data_df['H_PCT10'].replace('#', wage_cap_hourly)
cleaned_data_df['H_PCT25'] = cleaned_data_df['H_PCT25'].replace('#', wage_cap_hourly)
cleaned_data_df['H_PCT75'] = cleaned_data_df['H_PCT75'].replace('#', wage_cap_hourly)
cleaned_data_df['H_PCT90'] = cleaned_data_df['H_PCT90'].replace('#', wage_cap_hourly)

# For annual wages
wage_cap_annual = 239200  # $239,200 per year
cleaned_data_df['A_MEAN'] = cleaned_data_df['A_MEAN'].replace('#', wage_cap_annual)
cleaned_data_df['A_PCT10'] = cleaned_data_df['A_PCT10'].replace('#', wage_cap_annual)
cleaned_data_df['A_PCT25'] = cleaned_data_df['A_PCT25'].replace('#', wage_cap_annual)
cleaned_data_df['A_PCT75'] = cleaned_data_df['A_PCT75'].replace('#', wage_cap_annual)
cleaned_data_df['A_PCT90'] = cleaned_data_df['A_PCT90'].replace('#', wage_cap_annual)


# - Wages were capped at reasonable limits:  
#   Hourly wage: Limited to **115/hour**.  
#   Annual wage: Limited to **239,200/year**.  
# - In this section, revisiting the problem of data types and missing values becomes crucial to tackle in more detail.
# 

# In[27]:


# Convert specific wage-related columns to numbers, replacing any invalid values with NaN.
wage_columns = ['H_MEAN', 'A_MEAN', 'H_PCT10', 'H_PCT25', 'H_MEDIAN', 'H_PCT75', 'H_PCT90',
                'A_PCT10', 'A_PCT25', 'A_MEDIAN', 'A_PCT75', 'A_PCT90']

for col in wage_columns:
    cleaned_data_df[col] = pd.to_numeric(cleaned_data_df[col], errors='coerce')


# For numeric columns, replace missing values (NaN) with the average (mean) of that column.
numeric_columns = cleaned_data_df.select_dtypes(include='number').columns
cleaned_data_df[numeric_columns] = cleaned_data_df[numeric_columns].fillna(cleaned_data_df[numeric_columns].mean())

# For categorical columns (non-numeric), replace missing values (NaN) with the most common value (mode).
categorical_columns = cleaned_data_df.select_dtypes(exclude='number').columns
cleaned_data_df[categorical_columns] = cleaned_data_df[categorical_columns].apply(lambda col: col.fillna(col.mode()[0]))


# - Wage columns were numerized and non-numeric values were treated as missing value.<br> 
# - Missing values were addressed:<br>
# Numeric columns: Counted in separately at the average value of the column.<br>
# Categorical columns: Contained the highest frequency value in the set known as the mode.<br>
# - These steps proved useful in getting the dataset to the required level of cleanliness and quality of data analysis.<br>

# In[30]:


cleaned_data_df.head()


# In[32]:


cleaned_data_df.tail()


# In[34]:


# Step 5: Exploratory Data Analysis (EDA)
# Summary Statistics
print("\nSummary Statistics of Cleaned Dataset:\n", cleaned_data_df.describe())


# In[36]:


# Distribution of Hourly Mean Wages
plt.figure(figsize=(12, 6))
sns.histplot(cleaned_data_df['H_MEAN'], kde=True, bins=30)
plt.title('Distribution of Hourly Mean Wages')
plt.xlabel('Hourly Mean Wage')
plt.ylabel('Frequency')
plt.show()


# - The distribution of wages was also positively skewed which implies that while many employees earned low wages some earned very high wages.

# In[39]:


# Distribution of Annual Mean Wages
plt.figure(figsize=(12, 6))
sns.histplot(cleaned_data_df['A_MEAN'], kde=True, bins=30)
plt.title('Distribution of Annual Mean Wages')
plt.xlabel('Annual Mean Wage')
plt.ylabel('Frequency')
plt.show()


# - As with the previous figure and the calculation of hourly and annual wages these percentages also provided a similar result.

# In[42]:


# Relationship between Total Employment and Hourly Mean Wage
plt.figure(figsize=(12, 6))
sns.scatterplot(x='TOT_EMP', y='H_MEAN', data=cleaned_data_df)
plt.title('Total Employment vs Hourly Mean Wage')
plt.xlabel('Total Employment')
plt.ylabel('Hourly Mean Wage')
plt.show()


# - Here, an individual analysis for total employment was in a scatter plot which was not correlated with total employment and hourly wages.
# - Virtually all the occupations had a low total employment and their respective wages were less than $50 per hour.
# 

# In[45]:


# Average Hourly Wage by Occupation Group
plt.figure(figsize=(14, 8))
occupation_group_avg_wage = cleaned_data_df.groupby('O_GROUP')['H_MEAN'].mean().sort_values()
occupation_group_avg_wage.plot(kind='barh')
plt.title('Average Hourly Mean Wage by Occupation Group')
plt.xlabel('Hourly Mean Wage')
plt.ylabel('Occupation Group')
plt.show()


# - It was evident that various job groups had wide variations in average wages.
# - Some workers received higher wages more frequently, thus there is a clear pattern signifying stratification of wages by type of job.
# 

# In[48]:


# Correlation Heatmap for Numerical Features
plt.figure(figsize=(12, 12))
numeric_cols = cleaned_data_df.select_dtypes(include=[np.number])
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# - Strong Relationships, in the situation with wages, all percentile ranks, from the lower decile, quartile, or nin. to higher values were almost equally connected.<br>
# - Expected Correlations, there was a significant correlation between wages by hours and wages by year.<br>
# - There was disconnection between employment numbers and wages.<br>
# - These results imply that the type of job and industry are more critical to wages than the total number of employees in an occupation.

# #### Highly Correlated Features:
# 
# Wage Percentile Correlations (H_PCT10, H_PCT25, H_MEDIAN, H_PCT75, H_PCT90):
# These wage percentile variables which are similarly named wage percentile columns show extremely positive relationships between them and H_MEAN (mean hourly wage). This interconnection is logical since these percentiles denote distinct positions in the sharply defined wage distribution. There is an extremely strong positive coefficient here which indicates that there may be multicollinearity and these different percentiles can’t be used in a statistical model all together.
# 
# #### Mean Hourly Wage (H_MEAN) and Annual Mean Wage (A_MEAN):
# 
# As expected there is very high correlation between H_MEAN and A_MEAN. It also found that hourly wage can be converted to the annual wage to establish a direct proportionality, therefore, these variables can usually be interchanged when performing wage analyses.
# 
# #### Total Employment (TOT_EMP):
# 
# As got in the above results, the post-estimate of TOT_EMP has a rather low coefficient of determination between the H_MEAN and the A_MEAN. This means that occupational wage determination does not heavily relied on the number of employees in a given occupation. It’s also important to understand that the total employment volume does not tell more about compensation structures of a given job type.
# 
# #### Area Code (AREA) and Ownership Code (OWN_CODE):
# 
# Both AREA and OWN_CODE show rather a weak to no association with wage related variables. This leads to the understanding of the fact that geographical location and ownership may not be determinants of wages in this dataset.
# 
# #### Percentage Standard Error (PRSE):
# 
# As could be observed from the correlation coefficients EMP_PRSE and MEAN_PRSE have relatively low correlation with the other wage features. This infers that, the error bar associated with estimates of employment and wages does not inflate the actual wage quantification.

# In[51]:


# Step 6: Regression Analysis
# Prepare the data for regression
X = cleaned_data_df[['H_PCT10', 'H_MEDIAN', 'H_PCT90']]
y = cleaned_data_df['H_MEAN']  


# - (H_PCT10, H_PCT25, H_PCT75, H_PCT90) these features (percentiles of the wage distribution) are highly correlated with H_MEAN (mean hourly wage), which is target variable in the regression models.
# - H_MEAN and A_MEAN are highly correlated with each other (since annual wages are derived from hourly wages).Choose one of these two features for wage prediction. H_MEAN is often more intuitive when looking at hourly wage data.
# 

# In[54]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[56]:


# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)


# In[58]:


# Model Metrics
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)

print(f"Linear Regression - MSE: {mse_lin}, R²: {r2_lin}")


# In[60]:


# Residual Plot for Linear Regression
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_lin - y_test, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Linear Regression: Residual Plot')
plt.show()


# In[62]:


# Histogram of Residuals
plt.figure(figsize=(12, 6))
sns.histplot(y_pred_lin - y_test, kde=True, bins=30, label='Linear', color='blue', alpha=0.5)
plt.title('Linear Regression: Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[64]:


# QQ Plot 
stats.probplot(y_test-y_pred_lin,dist="norm",plot =plt)
plt.xlabel('Quantiles ')
plt.ylabel('Sample Quantiles')
plt.title('Linear Regression: Residual Plot')
plt.grid(True)
plt.show()


# In[66]:


# OLS Regression Result 
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# Fit the OLS model
ols_model = sm.OLS(y_train, X_train_const).fit()

# Get predictions
y_pred_ols = ols_model.predict(X_test_const)

# Model summary
print(ols_model.summary())

# Evaluating the OLS model
mse_ols = mean_squared_error(y_test, y_pred_ols)
r2_ols = r2_score(y_test, y_pred_ols)

print(f"OLS Regression - MSE: {mse_ols}, R²: {r2_ols}")


# The OLS model performs very well, with an R-squared value of 0.89, meaning it explains 89.8% of the variance in the hourly mean wage.

# In[69]:


# Residual plot: Fitted values vs. residuals
plt.figure(figsize=(12, 6))
plt.scatter(y_pred_ols, y_pred_ols - y_test, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('OLS Regression: Residuals vs Fitted Values')
plt.show()


# In[71]:


# Histogram of residuals
plt.figure(figsize=(12, 6))
sns.histplot(y_pred_ols - y_test, kde=True, bins=30, label='OLS Residuals', color='purple', alpha=0.5)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[73]:


# Cooks Distance 
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# Add constant to the features
X_train_const = sm.add_constant(X_train)  # Replace X_train with your dataset
y_train = np.array(y_train)  # Ensure the target variable is a NumPy array

# Fit the OLS model
ols_model = sm.OLS(y_train, X_train_const).fit()

# Get influence metrics
influence = ols_model.get_influence()
cooks_d = influence.cooks_distance  # Cook's Distance

# Plot Cook's Distance
plt.figure(figsize=(10, 6))  # Larger size for better readability
plt.stem(
    np.arange(len(cooks_d[0])),  # Observation index
    cooks_d[0],  # Cook's Distance values
    markerfmt=",",  # Use a small marker
    linefmt="b-",  # Blue lines
    basefmt=" "  # Remove baseline
)
# Add the threshold line for influential points
threshold = 4 / len(cooks_d[0])  # Common threshold for Cook's Distance
plt.axhline(threshold, color='red', linestyle='--', linewidth=1, label=f'Threshold = {threshold:.4f}')

# Adjust the y-axis limit to a maximum of 5
plt.ylim(0, 0.5)

# Add labels, title, and legend
plt.title("Cook's Distance (Outlier Detection)", fontsize=16)
plt.xlabel("Observation Index", fontsize=14)
plt.ylabel("Cook's Distance", fontsize=14)
plt.legend(fontsize=12)
plt.grid(color='gray', linestyle='--', linewidth=0.5)  # Subtle grid for clarity
plt.tight_layout()
plt.show()

# Identify influential points
influential_points = np.where(cooks_d[0] > threshold)[0]
print(f"Number of Influential Points: {len(influential_points)}")
if len(influential_points) > 0:
    print(f"Influential Points (Indices): {influential_points}")


# A Linear Regression model is trained on the standardized training data, and predictions are made on the testing data.
# Residuals should be scattered randomly around zero if the model is fitting the data well. In this plot, the residuals seem fairly random, indicating that the linear regression model has a good fit.the residuals are normally distributed around zero, this supports the assumptions of linear regression (normality of errors). The histogram shows that residuals follow a roughly normal distribution, supporting the model's assumptions.

# NO TRANSFORMATION ARE USED.

# In[77]:


# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# The features are standardized using StandardScaler(), meaning they are centered (mean of 0) and scaled (standard deviation of 1). This is important for models like Lasso and Ridge, which are sensitive to the scale of the data.

# In[80]:


# Lasso Regression
lasso = Lasso(alpha=0.1)  # You can adjust the alpha value
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)


# In[82]:


# Model Metrics
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f"Lasso Regression - MSE: {mse_lasso}, R²: {r2_lasso}")


# In[84]:


# Residual Plot for Lasso Regression
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_lasso - y_test, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Lasso Regression: Residual Plot')
plt.show()


# In[86]:


# Histogram of Residuals
plt.figure(figsize=(12, 6))
sns.histplot(y_pred_lasso - y_test, kde=True, bins=30, label='Lasso', color='green', alpha=0.5)
plt.title('Lasso Regression: Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[88]:


# QQ Plot 
stats.probplot(y_test-y_pred_lasso,dist="norm",plot =plt)
plt.xlabel('Quantiles ')
plt.ylabel('Sample Quantiles')
plt.title('Lasso Regression: Residual Plot')
plt.grid(True)
plt.show()


# - Similar to the linear regression plot, we expect the residuals to be randomly distributed, suggesting that the Lasso model fits the data well without obvious overfitting or underfitting. However, the L1 regularization in Lasso could reduce the impact of less relevant features.
# - we expect the residuals to follow a normal distribution, similar to linear regression. The residuals should be centered around 0, indicating that the model’s predictions are accurate on average, and there is no systematic bias.

# # Lasso Log

# In[92]:


# Lasso Log

from sklearn.model_selection import GridSearchCV

# Log-transform target variable
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)


# In[94]:


# Define a range of alpha values for Lasso regularization
alpha_values = np.logspace(-4, 1, 50)  # From 0.0001 to 10 in 50 steps

# Hyperparameter tuning with GridSearchCV for Lasso Regression
lasso1 = Lasso(max_iter=10000)
param_grid = {'alpha': alpha_values}
grid_search = GridSearchCV(lasso1, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train_log)


# In[36]:


# Best alpha
best_alpha_lasso = grid_search.best_params_['alpha']
print(f"Best alpha for Lasso: {best_alpha_lasso}")


# In[37]:


# Fit Lasso with best alpha
lasso_best = Lasso(alpha=best_alpha_lasso, max_iter=10000)
lasso_best.fit(X_train, y_train_log)
y_pred_lasso_log = lasso_best.predict(X_test)


# In[38]:


# Convert predictions back to original scale
y_pred_lasso_original = np.expm1(y_pred_lasso_log)

# Model metrics
mse_lasso1 = mean_squared_error(np.expm1(y_test_log), y_pred_lasso_original)
r2_lasso1 = r2_score(np.expm1(y_test_log), y_pred_lasso_original)

print(f"Lasso Regression - MSE (original scale): {mse_lasso1:.2f}, R² (original scale): {r2_lasso1:.4f}")


# In[39]:


# Residuals in original scale
residuals_lasso_original = np.expm1(y_test_log) - y_pred_lasso_original

# Residual Plot
plt.figure(figsize=(12, 6))
plt.scatter(np.expm1(y_test_log), residuals_lasso_original, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Actual Values (Original Scale)')
plt.ylabel('Residuals (Original Scale)')
plt.title('Lasso Regression: Residual Plot (After Log Transformation)')
plt.grid(True)
plt.show()


# In[40]:


# Histogram of Residuals
plt.figure(figsize=(12, 6))
sns.histplot(residuals_lasso_original, kde=True, bins=30, label='Lasso', color='green', alpha=0.5)
plt.title('Lasso Regression: Residuals Distribution (Original Scale, After Log Transformation)')
plt.xlabel('Residuals (Original Scale)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()


# In[41]:


# Ridge Regression
ridge = Ridge(alpha=0.1)  # You can adjust the alpha value
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)


# In[42]:


# Model Metrics
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"Ridge Regression - MSE: {mse_ridge}, R²: {r2_ridge}")


# In[43]:


# Residual Plot for Ridge Regression
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_ridge - y_test, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Ridge Regression: Residual Plot')
plt.show()


# In[44]:


# Histogram of Residuals
plt.figure(figsize=(12, 6))
sns.histplot(y_pred_ridge - y_test, kde=True, bins=30, label='Ridge', color='orange', alpha=0.5)
plt.title('Ridge Regression: Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[45]:


# QQ Plot 
stats.probplot(y_test-y_pred_ridge,dist="norm",plot =plt)
plt.xlabel('Quantiles ')
plt.ylabel('Sample Quantiles')
plt.title('Ridge Regression: Residual Plot')
plt.grid(True)
plt.show()


# The residuals should follow a similar normal distribution as with the other models, suggesting that Ridge regression has a good fit and that its regularization was helpful in stabilizing the model without overly shrinking the coefficients.
# 

# In[46]:


# Ridge Log


# In[47]:


# Define range for alpha tuning
alpha_values = np.logspace(-4, 1, 50)

# Hyperparameter tuning with GridSearchCV for Ridge Regression
ridge1 = Ridge(max_iter=10000)
param_grid = {'alpha': alpha_values}
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train_log)

# Best alpha
best_alpha_ridge = grid_search.best_params_['alpha']
print(f"Best alpha for Ridge: {best_alpha_ridge}")

# Fit Ridge with best alpha
ridge_best = Ridge(alpha=best_alpha_ridge, max_iter=10000)
ridge_best.fit(X_train, y_train_log)
y_pred_ridge_log = ridge_best.predict(X_test)


# In[48]:


# Convert predictions back to original scale
y_pred_ridge_original = np.expm1(y_pred_ridge_log)

# Model metrics
mse_ridge1 = mean_squared_error(np.expm1(y_test_log), y_pred_ridge_original)
r2_ridge1 = r2_score(np.expm1(y_test_log), y_pred_ridge_original)

print(f"Ridge Regression - MSE (original scale): {mse_ridge:.2f}, R² (original scale): {r2_ridge:.4f}")


# In[49]:


# Residuals in original scale
residuals_ridge_original = np.expm1(y_test_log) - y_pred_ridge_original

# Residual Plot
plt.figure(figsize=(12, 6))
plt.scatter(np.expm1(y_test_log), residuals_ridge_original, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Actual Values (Original Scale)')
plt.ylabel('Residuals (Original Scale)')
plt.title('Ridge Regression: Residual Plot (After Log Transformation)')
plt.grid(True)
plt.show()


# In[50]:


# Histogram of Residuals
plt.figure(figsize=(12, 6))
sns.histplot(residuals_ridge_original, kde=True, bins=30, label='Ridge', color='orange', alpha=0.5)
plt.title('Ridge Regression: Residuals Distribution (Original Scale, After Log Transformation)')
plt.xlabel('Residuals (Original Scale)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()


# In[52]:


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# Random Forest Regressor
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)


# In[53]:


# Model Metrics
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest Regressor - MSE: {mse_rf}, R²: {r2_rf}")


# In[54]:


# Residual Plot for Random Forest
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_rf - y_test, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Random Forest: Residual Plot')
plt.show()


# In[55]:


# Feature Importance
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': random_forest.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(feature_importances)


# In[56]:


# QQ Plot 
stats.probplot(y_test-y_pred_rf,dist="norm",plot =plt)
plt.xlabel('Quantiles ')
plt.ylabel('Sample Quantiles')
plt.title('Random forest: Residual Plot')
plt.grid(True)
plt.show()


# In[57]:


# Gradient Boosting Regressor
gradient_boost = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gradient_boost.fit(X_train, y_train)
y_pred_gb = gradient_boost.predict(X_test)


# In[58]:


# Model Metrics
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print(f"Gradient Boosting Regressor - MSE: {mse_gb}, R²: {r2_gb}")


# In[59]:


# Residual Plot for Gradient Boosting
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_gb - y_test, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Gradient Boosting: Residual Plot')
plt.show()


# ## AI tool Usage

# Our team Used AI tool chatgpt for code assistance in debugging errors and dataset merging during initial stage, resolving missing values

# ## Dataset link

# https://www.bls.gov/oes/tables.htm
