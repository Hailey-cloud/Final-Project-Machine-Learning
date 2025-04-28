#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().system('pip install pandasql')
from pandasql import sqldf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

import warnings
warnings.filterwarnings('ignore')


# In[9]:


# Read input data
train = pd.read_csv('data/train.zip', compression='zip')
test = pd.read_csv('data/test.csv')
stores = pd.read_csv('data/stores.csv')
sample_submission = pd.read_csv('data/sample_submission.csv')
oil = pd.read_csv('data/oil.csv')
holidays_events = pd.read_csv('data/holidays_events.csv')
transactions = pd.read_csv('data/transactions.csv')


# In[11]:


print("Train Data:")
print(train.head())
print(train.describe())
print(train.info())

print("\nTest Data:")
print(test.head())
print(test.describe())
print(test.info())

print("\nStores Data:")
print(stores.head())
print(stores.describe())
print(stores.info())

print("\nOil Data:")
print(oil.head())
print(oil.describe())
print(oil.info())

print("\nHolidays and Events Data:")
print(holidays_events.head())
print(holidays_events.describe())
print(holidays_events.info())

print("\nTransactions Data:")
print(transactions.head())
print(transactions.describe())
print(transactions.info())

print("\nSample Submission:")
print(sample_submission.head())


# ## Data Preprocessing

# ### Handle missing values

# In[13]:


# Find missing values:

print("Train Data:")
print(train.isnull().sum())

print("\nTest Data:")
print(test.isnull().sum())

print("\nStores Data:")
print(stores.isnull().sum())

print("\nOil Data:")
print(oil.isnull().sum())

print("\nHolidays and Events Data:")
print(holidays_events.isnull().sum())

print("\nTransactions Data:")
print(transactions.isnull().sum())


# #### Missing values in Oil

# In[15]:


oil.head(16)


# In[17]:


# dcoilwtico mean
oil['dcoilwtico'].mean()


# In[19]:


# fill missing values with interpolate
oil['dcoilwtico'] = oil['dcoilwtico'].interpolate()

# Find missing values
print(oil.isnull().sum())


# In[21]:


oil[oil.isna().any(axis=1)]


# In[23]:


# bfill to clean 1st entry as nan
oil['dcoilwtico'] = oil['dcoilwtico'].bfill()

# Find missing values
print(oil.isnull().sum())


# In[25]:


oil.head(16)


# ### Find inconsistencies

# In[27]:


# Find duplicate rows

print(f"duplicates in train are: {len(train[train.duplicated()])}")
print(f"duplicates in test are: {len(test[test.duplicated()])}")
print(f"duplicates in stores are: {len(stores[stores.duplicated()])}")
print(f"duplicates in oil are: {len(oil[oil.duplicated()])}")
print(f"duplicates in holidays_events are: {len(holidays_events[holidays_events.duplicated()])}")
print(f"duplicates in transactions are: {len(transactions[transactions.duplicated()])}")


# In[29]:


# Find negative sales
print("Negative sales count in train:", (train['sales'] < 0).sum())


# In[31]:


# Find is stores match between dataset
print("Stores in train but not in stores.csv:", set(train['store_nbr']) - set(stores['store_nbr']), )
print("Stores in test but not in stores.csv:", set(test['store_nbr']) - set(stores['store_nbr']))
print("Stores in transactions but not in stores.csv:", set(transactions['store_nbr']) - set(stores['store_nbr']))


# All the data is cleaned and there are not inconsistencies.

# ## Exploratory Data Analysis

# In[33]:


# date to datetime
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])
oil['date'] = pd.to_datetime(oil['date'])
holidays_events['date'] = pd.to_datetime(holidays_events['date'])
transactions['date'] = pd.to_datetime(transactions['date'])


# In[35]:


# Range from dates
print("Train date range:", train['date'].min(), "to", train['date'].max())
print("Test date range:", test['date'].min(), "to", test['date'].max())


# ### Sales and Oil prices

# In[37]:


# Total Sales Over Time
daily_sales = train.groupby('date')['sales'].sum().reset_index()

sb.set_style("whitegrid")
plt.figure(figsize=(15, 6))
sb.lineplot(data=daily_sales, x='date', y='sales', color='blue', label="Total Sales")
plt.title("Total Sales over Time")
plt.ylabel("Sales")
plt.xlabel("Date")
plt.show()


# In[39]:


# Lineplot from oil over time
plt.figure(figsize=(15,6))
sb.lineplot(data=oil, y='dcoilwtico', x='date', color='brown', label="oil price")
plt.title("Oil price over Time") 
plt.show()


# In[41]:


# Resample by month
monthly_sales = train.resample('M', on='date')['sales'].mean()
monthly_oil = oil.resample('M', on='date')['dcoilwtico'].mean()

fig, ax1 = plt.subplots(figsize=(13, 5))
# Plot Sales
ax1.plot(monthly_sales.index, monthly_sales.values, color='blue', label='Average Sales')
ax1.set_ylabel('Sales')

# Plot Oil price
ax2 = ax1.twinx()
ax2.plot(monthly_oil.index, monthly_oil.values, color='brown', label='Average Oil Price')
ax2.set_ylabel('Oil Price')

plt.title('Monthly Average Sales vs Oil Price comparisson')
fig.tight_layout()
fig.legend()
plt.show()


# After 2014, oil prices dropped significantly, while average sales began to rise steadily. This suggests that lower oil prices may have supported higher consumer spending or reduced transportation costs. Although sales show some seasonal changes, the overall trend moves in the opposite direction of oil prices.

# ### Total Sales by Store

# In[43]:


# Unique values from store_nbr
print("stores:", train['store_nbr'].nunique())


# In[45]:


# Grouping by Stores (TOP)
top_store_sales = train.groupby('store_nbr')[['sales']].sum().sort_values('sales', ascending=False).head(5)
print(top_store_sales)


# In[47]:


# Grouping by Stores (BOTTOM)
bottom_store_sales = train.groupby('store_nbr')[['sales']].sum().sort_values('sales', ascending=True).head(5)
print(bottom_store_sales)


# In[49]:


# Grouping by Stores
store_sales = train.groupby('store_nbr')[['sales']].sum()

# Barplot from total sales by store
plt.figure(figsize=(13,5))
sb.barplot(data=store_sales, y='sales', x='store_nbr', palette='RdYlBu')
plt.title("Total Sales by Store")
plt.show()


# ### Total Sales by Family over the time

# In[51]:


# Total Sales
family_sales = train.groupby(['date', 'family'])[['sales']].sum().reset_index()
families = sorted(family_sales['family'].unique())

# lineplot from total sales by store
fig, axes = plt.subplots(11, 3, figsize=(18, 35))
axes = axes.flatten()

# Plot each family
for i, fam in enumerate(families):
    ax = axes[i]
    sb.lineplot(data=family_sales[family_sales['family'] == fam], x='date', y='sales', ax=ax, color='blue')
    ax.set_title(fam)

plt.tight_layout()
plt.suptitle("Sales Over Time by Family")
plt.show()


# Between 2013 and 2017, sales for drinks, groceries, and cleaning products kept growing steadily. Frozen foods, school supplies, and liquor sold more during certain times of the year. On the other hand, book and baby product sales dropped a lot, while personal care and pet products became more popular over time.
# 
# Overall, the data reflects consistent growth in consumer spending, especially on essentials and lifestyle products.

# ### Finding Seasonality in Sales by Weekday, Month or Year

# In[53]:


family_sales['month'] = family_sales['date'].dt.month
family_sales['weekday'] = family_sales['date'].dt.weekday
family_sales['year'] = family_sales['date'].dt.year

# Finding seasonality
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes = axes.flatten()

sb.barplot(data=family_sales, x='weekday', y='sales', ax=axes[0], color='dodgerblue')
axes[0].set_title("Weekday total sales")

sb.barplot(data=family_sales, x='month', y='sales', ax=axes[1], color='dodgerblue')
axes[1].set_title("Montly total sales")

sb.barplot(data=family_sales, x='year', y='sales', ax=axes[2], color='dodgerblue')
axes[2].set_title("Year total sales")

plt.tight_layout()
plt.show()


# Sales peaked on weekends, especially Sundays. December was the top-selling month, likely driven by holiday shopping. Yearly sales showed an increasing trend from 2013 to 2017, indicating overall business growth.

# ### Holidays impact on Sales

# In[39]:


holiday_avg = train[train['date'].isin(holidays_events['date'])]['sales'].mean()
non_holiday_avg = train[~train['date'].isin(holidays_events['date'])]['sales'].mean()

# Plot
sb.barplot(x=['Holiday', 'Not Holiday'], y=[holiday_avg, non_holiday_avg], palette='Set2')
plt.title('Average Sales: Holidays vs Non-Holidays')
plt.tight_layout()
plt.show()


# Sales were noticeably higher on holidays, suggesting that special dates drive a significant boost in customer purchases.

# ## Feature Engineering

# In[42]:


holidays_events.head()


# In[43]:


# Find holidays transferred
holidays_events[holidays_events['transferred']]


# ### Handle holiday dataset 

# In[45]:


print(holidays_events.shape)
holidays_events.head()


# In[46]:


holidays_events['type'].value_counts()


# In[47]:


holidays_events.drop(holidays_events[(holidays_events['type'] == 'Work Day') | (holidays_events['transferred'] == True)].index,inplace=True)
holidays_events.drop('transferred', axis=1, inplace=True)

holidays_events['type'].value_counts()


# In[48]:


holidays_events['holiday_flg'] =1 


# ### Merge datasets

# In[50]:


train['is_test'] = 0
test['is_test'] = 1


# In[51]:


#merged holiday and oil data with train data
exclude_col = 'date'

new_columns = {col: col + '_holiday' for col in holidays_events.columns if col != exclude_col}

holidays_events = holidays_events.rename(columns=new_columns)


merged_df = pd.concat([train, test], ignore_index=True)
merged_df = pd.merge(merged_df, holidays_events, on='date', how='left')
merged_df = pd.merge(merged_df, oil, on='date', how='left')


merged_df.head()


# In[52]:


print(merged_df.shape)
merged_df.isnull().sum()


# In[53]:


merged_df[merged_df['dcoilwtico'].isnull()]


# In[54]:


#Handle missing values of merged data by filling 'not holiday'
columns_to_fill = ['type_holiday', 'locale_holiday', 'locale_name_holiday', 'description_holiday']
merged_df[columns_to_fill] = merged_df[columns_to_fill].fillna('Not holiday')
merged_df['holiday_flg_holiday'] = merged_df['holiday_flg_holiday'].fillna(0)


# In[55]:


#oil price data which is dcoilwtico have some missing dates, so fill them by mean of value of previous/next date
date_oil = merged_df[['date', 'dcoilwtico']].drop_duplicates(subset='date')
date_oil = date_oil.sort_values('date').reset_index(drop=True)

date_oil['dcoilwtico'] = date_oil['dcoilwtico'].interpolate(method='linear', limit_direction='both')
date_oil.head()


# In[56]:


merged_df = merged_df.drop(columns=['dcoilwtico'])

merged_df = pd.merge(merged_df, date_oil, on='date', how='left')
print(merged_df.shape)
merged_df.isnull().sum()


# ## ML

# In[ ]:




