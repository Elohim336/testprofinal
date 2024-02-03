#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_df=pd.read_csv(boston_url)
boston_df.head(2)


# ### Median value of owner-occupied homes Boxplot

# In[2]:


boston_df.describe()


# In[3]:


boston_df.head()


# In[4]:


boston_df.tail(4)


# #### Median value of owner-occupied homes

# In[5]:


sns.boxplot(boston_df['MEDV'])
plt.title('Median value of owner-occupied homes')
plt.xlabel('Median Value MEDV')
plt.ylabel('Values')
plt.show()


# #### Median value is between 20 and 30 for owner occupied homes

# ### Bar plot for the Charles river variable

# In[6]:


boston_df['CHAS'].unique()


# In[7]:


chas_counts = boston_df['CHAS'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=chas_counts.index, y=chas_counts.values)
plt.xlabel('Charles River (CHAS)')
plt.ylabel('Count')
plt.title('Distribution of Charles River (CHAS)')


# ##### Non Tract bound is more

# #### Boxplot for the MEDV variable vs the AGE variable

# In[8]:


age_group = [0,35,70,float('inf')]
age_groups = ['35','35-70','above70']


# In[9]:


boston_df["aged_class"] = pd.cut(boston_df['AGE'],bins = age_group ,labels= age_groups,include_lowest=False)
print(boston_df['aged_class'].head(4))


# In[10]:


sns.boxplot(data = boston_df,x='aged_class', y='MEDV')
plt.title('MEDV variable vs the AGE variable')
plt.xlabel('aged_class')
plt.ylabel('MEDV')
plt.show()


# ##### Median value of owner-occupied homes in $1000's is 
# ##### more for 35 age and below group
# ##### and the lowest is for above 70 aged.
# 

# ### scatter plot to show the relationship between Nitric oxide & the proportion of non-retail business acres per town

# In[11]:


plt.scatter(boston_df['NOX'],boston_df['INDUS'],color='blue')
plt.title("Rltnshp b/w No2 and non retail business")
plt.xlabel('nitric oxide')
plt.ylabel('non retail business')
plt.grid(True)
plt.show()


# #### NOX have higher concentration in towns were INDUS is low, when INDUS proportion increases the NOX becomes less

# ### Histogram for the pupil to teacher ratio

# In[12]:


sns.histplot(boston_df['PTRATIO'])
plt.title('Histogram for the pupil to teacher ratio')
plt.xlabel('pupil teacher ratio')
plt.show()


# ### Difference in median value of houses bounded by the Charles river 

# #### Negative skweness is showed in PTRATIO, the mean is less than the median, the mode is between 20 and 22

# ## Difference in median value of houses bounded by the Charles river

# #### (H0): There is no significant difference in the MEDV between those bounded and not bounded by the Charles River.
# 
# #### Alternate Hypothesis (H1): There is a significant difference in the MEDV between those bounded & not bounded by the Charles River.

# In[13]:


import statsmodels.api as sm
from scipy.stats import ttest_ind
from statsmodels.formula.api import ols


# In[14]:


group1_data = boston_df[boston_df['CHAS'] == 1]['MEDV']
group2_data = boston_df[boston_df['CHAS'] == 0]['MEDV']

t_statistic, p_value = ttest_ind(group1_data, group2_data, equal_var=False)

print("T-statistic:", t_statistic)
print("P-value:", p_value)


# #### We reject the null hypothesis.There is sufficient evidence to suggest that there is a significant difference in the median value of houses between those bounded and not bounded by the Charles River.

# ## Difference in Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940 

# #### Null Hypothesis (H0): There is no significant difference in the MEDV across different AGE.
# #### Alternate Hypothesis (H1): There is a significant difference in the MEDV across different AGE.

# In[15]:


from scipy.stats import f_oneway

boston_df['age_group'] = pd.cut(boston_df['AGE'], bins=[0, 20, 40, 60, 80, 100], labels=['0-20', '20-40', '40-60', '60-80', '80-100'])

f_statistic, p_value = f_oneway(
    boston_df[boston_df['age_group'] == '0-20']['MEDV'],
    boston_df[boston_df['age_group'] == '20-40']['MEDV'],
    boston_df[boston_df['age_group'] == '40-60']['MEDV'],
    boston_df[boston_df['age_group'] == '60-80']['MEDV'],
    boston_df[boston_df['age_group'] == '80-100']['MEDV']
)

print("ANOVA F-statistic:", f_statistic)
print("P-value:", p_value)


# #### The p-value is significantly less than the common significance level of 0.05, we reject the null hypothesis.
# #### There is strong evidence to suggest that there is a significant difference in the median values of houses across different proportions of owner-occupied units built prior to 1940.

# ### Relationship between Nitric oxide concentrations and proportion of non-retail business acres per town

# #### Null Hypothesis (H0): There is no significant relationship between NOX and  INDUS (correlation coefficient = 0).
# #### Alternate Hypothesis (H1): There is significant linear relationship between NOX and  INDUS 

# In[17]:


from scipy.stats import pearsonr
correlation = boston_df['NOX'].corr(boston_df['INDUS'])
correlation, p_value = pearsonr(boston_df['NOX'], boston_df['INDUS'])

print("Pearson Correlation Coefficient:", correlation)
print("P-value:", p_value)
alpha = 0.05
if p_value < alpha:
    print("The correlation is statistically significant.")
else:
    print("There is no significant correlation.")


# #### we reject the null hypothesis,there is a significant positive linear relationship between NOX and INDUS

# ###  Additional DIS on the MEDV(Median value of owner-occupied homes in $1000's)

# #### Null Hypothesis (H0): The additional weighted distance to the 5 Boston employment centers has no impact on the MEDV
# #### Alternate Hypothesis (H1): The additional weighted distance to the five Boston employment centers has a significant impact on the MEDV

# In[18]:


boston_df['const'] = 1
X = boston_df[['const', 'DIS']]
y = boston_df['MEDV']
model = sm.OLS(y, X).fit()
print(model.summary())


# #### The model suggests a statistically significant relationship between the weighted distances to (DIS) and (MEDV).
# #### R-squared value, suggests that other factors in the model may also influence the median value of homes. 

# In[ ]:




