#!/usr/bin/env python
# coding: utf-8

# In[119]:


get_ipython().system('pip install pandasql')
from pandasql import sqldf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

import warnings
warnings.filterwarnings('ignore')


# In[120]:


# Read input data
train = pd.read_csv('data/train.zip', compression='zip')
test = pd.read_csv('data/test.csv')
stores = pd.read_csv('data/stores.csv')
sample_submission = pd.read_csv('data/sample_submission.csv')
oil = pd.read_csv('data/oil.csv')
holidays_events = pd.read_csv('data/holidays_events.csv')
transactions = pd.read_csv('data/transactions.csv')


# In[121]:


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

# In[124]:


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

# In[126]:


oil.head(16)


# In[127]:


# dcoilwtico mean
oil['dcoilwtico'].mean()


# In[128]:


# fill missing values with interpolate
oil['dcoilwtico'] = oil['dcoilwtico'].interpolate()

# Find missing values
print(oil.isnull().sum())


# In[129]:


oil[oil.isna().any(axis=1)]


# In[130]:


# bfill to clean 1st entry as nan
oil['dcoilwtico'] = oil['dcoilwtico'].bfill()

# Find missing values
print(oil.isnull().sum())


# In[131]:


oil.head(16)


# ### Find inconsistencies

# In[133]:


# Find duplicate rows

print(f"duplicates in train are: {len(train[train.duplicated()])}")
print(f"duplicates in test are: {len(test[test.duplicated()])}")
print(f"duplicates in stores are: {len(stores[stores.duplicated()])}")
print(f"duplicates in oil are: {len(oil[oil.duplicated()])}")
print(f"duplicates in holidays_events are: {len(holidays_events[holidays_events.duplicated()])}")
print(f"duplicates in transactions are: {len(transactions[transactions.duplicated()])}")


# In[134]:


# Find negative sales
print("Negative sales count in train:", (train['sales'] < 0).sum())


# In[135]:


# Find is stores match between dataset
print("Stores in train but not in stores.csv:", set(train['store_nbr']) - set(stores['store_nbr']), )
print("Stores in test but not in stores.csv:", set(test['store_nbr']) - set(stores['store_nbr']))
print("Stores in transactions but not in stores.csv:", set(transactions['store_nbr']) - set(stores['store_nbr']))


# All the data is cleaned and there are not inconsistencies.

# ## Exploratory Data Analysis

# In[138]:


# date to datetime
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])
oil['date'] = pd.to_datetime(oil['date'])
holidays_events['date'] = pd.to_datetime(holidays_events['date'])
transactions['date'] = pd.to_datetime(transactions['date'])


# In[139]:


# Range from dates
print("Train date range:", train['date'].min(), "to", train['date'].max())
print("Test date range:", test['date'].min(), "to", test['date'].max())


# ### Sales and Oil prices

# In[141]:


# Total Sales Over Time
daily_sales = train.groupby('date')['sales'].sum().reset_index()

sb.set_style("whitegrid")
plt.figure(figsize=(15, 6))
sb.lineplot(data=daily_sales, x='date', y='sales', color='blue', label="Total Sales")
plt.title("Total Sales over Time")
plt.ylabel("Sales")
plt.xlabel("Date")
plt.show()


# In[142]:


# Lineplot from oil over time
plt.figure(figsize=(15,6))
sb.lineplot(data=oil, y='dcoilwtico', x='date', color='brown', label="oil price")
plt.title("Oil price over Time") 
plt.show()


# In[143]:


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

# In[146]:


# Unique values from store_nbr
print("stores:", train['store_nbr'].nunique())


# In[147]:


# Grouping by Stores (TOP)
top_store_sales = train.groupby('store_nbr')[['sales']].sum().sort_values('sales', ascending=False).head(5)
print(top_store_sales)


# In[148]:


# Grouping by Stores (BOTTOM)
bottom_store_sales = train.groupby('store_nbr')[['sales']].sum().sort_values('sales', ascending=True).head(5)
print(bottom_store_sales)


# In[149]:


# Grouping by Stores
store_sales = train.groupby('store_nbr')[['sales']].sum()

# Barplot from total sales by store
plt.figure(figsize=(13,5))
sb.barplot(data=store_sales, y='sales', x='store_nbr', palette='RdYlBu')
plt.title("Total Sales by Store")
plt.show()


# ### Total Sales by Family over the time

# In[151]:


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

# In[154]:


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

# In[157]:


holiday_avg = train[train['date'].isin(holidays_events['date'])]['sales'].mean()
non_holiday_avg = train[~train['date'].isin(holidays_events['date'])]['sales'].mean()

# Plot
sb.barplot(x=['Holiday', 'Not Holiday'], y=[holiday_avg, non_holiday_avg], palette='Set2')
plt.title('Average Sales: Holidays vs Non-Holidays')
plt.tight_layout()
plt.show()


# Sales were noticeably higher on holidays, suggesting that special dates drive a significant boost in customer purchases.

# ## Feature Engineering

# In[160]:


holidays_events.head()


# In[161]:


# Find holidays transferred
holidays_events[holidays_events['transferred']]


# ### Handle holiday dataset 

# In[163]:


print(holidays_events.shape)
holidays_events.head()


# In[164]:


holidays_events['type'].value_counts()


# In[165]:


holidays_events.drop(holidays_events[(holidays_events['type'] == 'Work Day') | (holidays_events['transferred'] == True)].index,inplace=True)
holidays_events.drop('transferred', axis=1, inplace=True)

holidays_events['type'].value_counts()


# In[166]:


holidays_events['holiday_flg'] =1 


# ### Merge datasets

# In[168]:


train['is_test'] = 0
test['is_test'] = 1


# In[169]:


#merged holiday and oil data with train data
exclude_col = 'date'

new_columns = {col: col + '_holiday' for col in holidays_events.columns if col != exclude_col}

holidays_events = holidays_events.rename(columns=new_columns)


merged_df = pd.concat([train, test], ignore_index=True)
merged_df = pd.merge(merged_df, holidays_events, on='date', how='left')
merged_df = pd.merge(merged_df, oil, on='date', how='left')


merged_df.head()


# In[170]:


print(merged_df.shape)
merged_df.isnull().sum()


# In[171]:


merged_df[merged_df['dcoilwtico'].isnull()]


# In[172]:


#Handle missing values of merged data by filling 'not holiday'
columns_to_fill = ['type_holiday', 'locale_holiday', 'locale_name_holiday', 'description_holiday']
merged_df[columns_to_fill] = merged_df[columns_to_fill].fillna('Not holiday')
merged_df['holiday_flg_holiday'] = merged_df['holiday_flg_holiday'].fillna(0)


# In[173]:


#oil price data which is dcoilwtico have some missing dates, so fill them by mean of value of previous/next date
date_oil = merged_df[['date', 'dcoilwtico']].drop_duplicates(subset='date')
date_oil = date_oil.sort_values('date').reset_index(drop=True)

date_oil['dcoilwtico'] = date_oil['dcoilwtico'].interpolate(method='linear', limit_direction='both')
date_oil.head()


# In[174]:


merged_df = merged_df.drop(columns=['dcoilwtico'])

merged_df = pd.merge(merged_df, date_oil, on='date', how='left')
print(merged_df.shape)
merged_df.isnull().sum()


# ## ML

# In[214]:


try:
    from sklearn.model_selection import TimeSeriesSplit
    
    # 時系列交差検証の設定
    tscv = TimeSeriesSplit(n_splits=3)
    
    # CV結果格納用
    cv_scores = []
    
    # サンプリングしたデータでのみ実行（計算時間短縮のため）
    sample_size = min(50000, len(X_train))
    np.random.seed(42)  # 再現性のため
    sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
    X_train_sample = X_train.iloc[sample_idx]
    y_train_sample = y_train.iloc[sample_idx]
    
    # ここでlog1p変換
    y_train_sample_log = np.log1p(y_train_sample)
    
    print(f"クロスバリデーション用データサイズ: {X_train_sample.shape}")
    
    # 時系列クロスバリデーション実行
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_sample)):
        X_tr, X_val = X_train_sample.iloc[train_idx], X_train_sample.iloc[val_idx]
        y_tr, y_val = y_train_sample_log.iloc[train_idx], y_train_sample_log.iloc[val_idx]  # log1p版
        
        print(f"Fold {fold+1} - 訓練: {X_tr.shape}, 検証: {X_val.shape}")
        
        try:
            d_train = xgb.DMatrix(X_tr, label=y_tr)
            d_val = xgb.DMatrix(X_val, label=y_val)
            
            cv_model = xgb.train(
                xgb_params,
                d_train,
                num_boost_round=200,
                early_stopping_rounds=20,
                evals=[(d_val, 'validation')],
                verbose_eval=False
            )
            
            val_preds = cv_model.predict(d_val)
            
        except Exception as e:
            print(f"XGBoostでエラーが発生しました: {e}")
            print("RandomForestを代替として使用します")
            
            rf_cv = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_cv.fit(X_tr, y_tr)
            val_preds = rf_cv.predict(X_val)
        
        # log1p-RMSE計算
        rmse = sqrt(mean_squared_error(y_val, val_preds))
        cv_scores.append(rmse)
        
        print(f"Fold {fold+1} log1p-RMSE: {rmse:.4f}")
    
    print(f"\n平均 log1p-RMSE: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
    
except Exception as e:
    print(f"クロスバリデーション中にエラーが発生しました: {e}")


# In[216]:


get_ipython().system('pip install lightgbm')


# In[218]:


try:
    from sklearn.model_selection import TimeSeriesSplit
    import lightgbm as lgb
    
    # Define Parame
    xgb_params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 1,
        'seed': 42,
        'nthread': -1
    }

    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 64,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 42,
        'n_jobs': -1
    }

    # Prepare the data
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []
    
    sample_size = min(50000, len(X_train))
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
    X_train_sample = X_train.iloc[sample_idx]
    y_train_sample = y_train.iloc[sample_idx]
    y_train_sample_log = np.log1p(y_train_sample)
    
    print(f"data size for Cross Valida: {X_train_sample.shape}")
    
    # Cross Validation
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_sample)):
        X_tr, X_val = X_train_sample.iloc[train_idx], X_train_sample.iloc[val_idx]
        y_tr, y_val = y_train_sample_log.iloc[train_idx], y_train_sample_log.iloc[val_idx]
        
        print(f"Fold {fold+1} - train: {X_tr.shape}, test: {X_val.shape}")
        
        # 1.XGBoost
        d_train = xgb.DMatrix(X_tr, label=y_tr)
        d_val = xgb.DMatrix(X_val, label=y_val)
        xgb_model = xgb.train(
            xgb_params,
            d_train,
            num_boost_round=200,
            early_stopping_rounds=20,
            evals=[(d_val, 'validation')],
            verbose_eval=False
        )
        xgb_val_preds = xgb_model.predict(d_val)
        
        # 2.LightGBM
        lgb_train = lgb.Dataset(X_tr, label=y_tr)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

        lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        valid_sets=[lgb_val],
        num_boost_round=200,
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
        lgb_val_preds = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)

        
        # 3.Model Ensemble
        blended_preds = (xgb_val_preds + lgb_val_preds) / 2
        
        # 4.log1p-RMSE Caliculation
        rmse = sqrt(mean_squared_error(y_val, blended_preds))
        cv_scores.append(rmse)
        
        print(f"Fold {fold+1} log1p-RMSE (XGB+LGBM): {rmse:.4f}")
    
    print(f"\nEnsemble Average log1p-RMSE: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

except Exception as e:
    print(f"Error: {e}")


# In[104]:


# optimizing the weights
best_rmse = float('inf')
best_weight = None

for weight in np.arange(0, 1.05, 0.05):  # 0.0, 0.05, 0.1, ..., 1.0
    blended_preds = weight * xgb_val_preds + (1 - weight) * lgb_val_preds
    rmse = sqrt(mean_squared_error(y_val, blended_preds))
    print(f"Weight XGB:{weight:.2f} LGBM:{1-weight:.2f} --> log1p-RMSE: {rmse:.5f}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_weight = weight

print(f"\n【optimal weights】 XGB:{best_weight:.2f} / LGBM:{1-best_weight:.2f} with log1p-RMSE: {best_rmse:.5f}")


# In[ ]:





# In[ ]:




