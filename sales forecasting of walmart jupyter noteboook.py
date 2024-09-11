#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # mathematical computation
import pandas as pd # data processing
import matplotlib.pyplot as plt # visualization
from matplotlib import pyplot
import seaborn as sns # visualization
import warnings 
warnings.filterwarnings('ignore') #ignore warnings

#machine Learning models Libraries
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import RidgeCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression

#Preprocessing related Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

#Date related Libraries
from datetime import date
#import holidays
import datetime


# In[2]:


train = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\Forecasting-Walmart-sales-data-main\walmart forecasting sales\train (1).csv")
train


# In[3]:


#checking for null values in training dataset
train.isnull().sum()


# In[4]:


#Loading testing dataset

test = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\Forecasting-Walmart-sales-data-main\walmart forecasting sales\test.csv")
test


# In[5]:


#Checking null values for testing dataset
test.isnull().sum()


# In[6]:


#Loading features dataset
features = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\Forecasting-Walmart-sales-data-main\walmart forecasting sales\features (1).csv")
features


# In[7]:


#checking null values for features dataset
features.isnull().sum()


# In[8]:


#Loading stores dataset

stores = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\Forecasting-Walmart-sales-data-main\walmart forecasting sales\stores.csv")
stores


# In[9]:


#Checking null values for stores dataset
stores.isnull().sum()


# In[10]:


#DATA CLEANING
percent_missing = features.isnull().sum() * 100 / len(features)
missing_value = pd.DataFrame({'column_name': features.isnull().sum(),
                                 'percent_missing': percent_missing})
missing_value


# In[11]:


features.describe()


# In[12]:


import seaborn as sns
sns.boxplot(x=features['CPI'])


# In[13]:


features['CPI'].fillna((features['CPI'].mean()), inplace=True)


# In[14]:


percent_missing = features.isnull().sum() * 100 / len(features)
missing_value = pd.DataFrame({'column_name': features.isnull().sum(),
                                 'percent_missing': percent_missing})
missing_value


# In[15]:


sns.boxplot(x=features['Unemployment'])


# In[16]:


features['Unemployment'].fillna((features['Unemployment'].median()), inplace=True)


# In[17]:


percent_missing = features.isnull().sum() * 100 / len(features)
missing_value = pd.DataFrame({'column_name': features.isnull().sum(),
                                 'percent_missing': percent_missing})
missing_value


# In[18]:


from statistics import mean

features['MarkDown1'] = features['MarkDown1'].fillna(0)
features['MarkDown2'] = features['MarkDown2'].fillna(0)
features['MarkDown3'] = features['MarkDown3'].fillna(0)
features['MarkDown4'] = features['MarkDown4'].fillna(0)
features['MarkDown5'] = features['MarkDown5'].fillna(0)


# In[19]:


percent_missing = features.isnull().sum() * 100 / len(features)
missing_value = pd.DataFrame({'column_name': features.isnull().sum(),
                                 'percent_missing': percent_missing})
missing_value


# In[20]:


#Handling negative values in train data

train.describe()


# In[21]:


train[train.Weekly_Sales<0]


# In[22]:


#Taking values greater than or equal to zero
train = train[train.Weekly_Sales>=0]
train


# In[23]:


#Exploratory Data Analysis
train.info()


# In[24]:


train.dtypes


# In[25]:


#converting date column into datetime type.
train['Date'] = pd.to_datetime(train.Date)


# In[26]:


train.dtypes


# In[27]:


#Converting 'IsHoliday' column values False to 0 and True to 1
train["IsHoliday"] = train["IsHoliday"].astype(int)


# In[28]:


train.head()


# In[29]:


train.dtypes


# In[30]:


#Exracting the time based feature from Date feature as we are predicting the sales.


# In[31]:


train['Year']=train['Date'].dt.year
train['Month']=train['Date'].dt.month
train['Day']=train['Date'].dt.day
train['n_days']=(train['Date'].dt.date-train['Date'].dt.date.min()).apply(lambda x:x.days)
train.head()


# In[32]:


train.tail()


# In[33]:


test.dtypes


# In[34]:


test['Date'] = pd.to_datetime(test.Date)


# In[35]:


test['Year']=test['Date'].dt.year
test['Month']=test['Date'].dt.month
test['Day']=test['Date'].dt.day
test['n_days']=(test['Date'].dt.date-test['Date'].dt.date.min()).apply(lambda x:x.days)
test.head()


# In[36]:


#Train data has the information of sales for 994 days


# In[37]:


#Impact of holidays on sales
print("Holiday")
print(train[train['IsHoliday']==True]['Weekly_Sales'].describe())
print("Non-Holiday")
print(train[train['IsHoliday']==False]['Weekly_Sales'].describe())


# In[38]:


#Sales in holiday week are more than sales in non-holiday week


# In[39]:


sns.relplot(x='Date',y='Weekly_Sales',hue='Year',data=train, kind='line',aspect=2)
plt.title("Sales Line Chart")
plt.show()


# In[40]:


sns.relplot(x='Month',y='Weekly_Sales',data=train, kind='line',aspect=2)
plt.title("Sales Line Chart")
plt.show()


# In[41]:


#Exploring Features data


# In[42]:


features.head()


# In[43]:


features.dtypes


# In[44]:


features.describe()


# In[45]:


features['Date'] = pd.to_datetime(features.Date)


# In[46]:


features.dtypes


# In[47]:


#Converting 'IsHoliday' column values False to 0 and True to 1
features["IsHoliday"] = features["IsHoliday"].astype(int)


# In[48]:


features.dtypes


# In[49]:


features.head()


# In[50]:


#Exploring stores data


# In[51]:


stores['Type'].value_counts()


# In[52]:


#Type of stores
sns.countplot(x='Type',data=stores)


# In[53]:


#Exploring test data


# In[54]:


test.info()


# In[55]:


test.describe()


# In[56]:


test.dtypes


# In[57]:


#Mearging train and test data with features and stores data


# In[58]:


stores = stores.merge(features,on='Store',how='left')
stores


# In[59]:


train  = train.merge(stores,on=['Store','Date','IsHoliday'],how='left')
train


# In[60]:


train.info()


# In[61]:


test  = test.merge(stores,on=['Store','Date','IsHoliday'],how='left')
test


# In[62]:


#Type of values in Stores datase


# In[63]:


print("The shape of stores data set is: ", stores.shape)
print("The unique value of store is: ", stores['Store'].unique())
print("The unique value of Type is: ", stores['Type'].unique())
# As store size is a numerical real valued feature.


# In[64]:


#The ratio of A,B and C type of walmart stores through pie chart


# In[65]:


sizes=[(22/(17+6+22))*100,(17/(17+6+22))*100,(6/(17+6+22))*100]
i_labels = 'A-type','B-type','C-type'
plt.pie(sizes,labels=i_labels,autopct='%1.1f%%')
plt.title('Type of Stores Ratio')
plt.axis('equal')
plt.show()


# In[66]:


#Relationship between size and weekly_sales


# In[67]:


#sales of storefor each type of store using box plot


# In[68]:


fig, ax = plt.subplots(figsize=(8, 5))
fig = sns.boxplot(x='Type', y='Weekly_Sales', data=train, showfliers=False)
plt.title("Box Plot Using Type feature")
plt.show()


# In[69]:


#Average Store Sales - Year Wise


# In[70]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go

store_sales_2010 = train[train.Year==2010].groupby('Store')['Weekly_Sales'].mean().to_dict()
store2010_df = pd.DataFrame(list(store_sales_2010.items()), columns=['Store', 'AvgSales2010'])

store_sales_2011 = train[train.Year==2011].groupby('Store')['Weekly_Sales'].mean().to_dict()
store2011_df = pd.DataFrame(list(store_sales_2011.items()), columns=['Store', 'AvgSales2011'])

store_sales_2012 =train[train.Year==2012].groupby('Store')['Weekly_Sales'].mean().to_dict()
store2012_df = pd.DataFrame(list(store_sales_2012.items()), columns=['Store', 'AvgSales2012'])

fig = make_subplots(rows=3, cols=1, subplot_titles=("Average Store Sales 2010", "Average Store Sales 2011", "Average Store Sales 2012"))

fig.add_trace(go.Bar(x=store2010_df.Store, y=store2010_df.AvgSales2010,),1, 1)

fig.add_trace(go.Bar(x=store2011_df.Store, y=store2011_df.AvgSales2011,),2, 1)

fig.add_trace(go.Bar(x=store2012_df.Store, y=store2012_df.AvgSales2012,),3, 1)

fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), template='plotly_dark', showlegend=False, height=1500)

fig.update_xaxes(title_text="Store", row=1, col=1)
fig.update_xaxes(title_text="Store", row=2, col=1)
fig.update_xaxes(title_text="Store", row=3, col=1)

fig.update_yaxes(title_text="AvgSales", row=1, col=1)
fig.update_yaxes(title_text="AvgSales", row=2, col=1)
fig.update_yaxes(title_text="AvgSales", row=3, col=1)

fig.update_xaxes(tick0=1, dtick=1)
fig.show()


# In[71]:


fig, ax = plt.subplots(figsize=(25, 8))
sns.boxplot(x="Store",y='Weekly_Sales',data=train,showfliers=False, hue="Type")
plt.title("Box Plot Using Size feature")
plt.show()


# In[72]:


fig, ax = plt.subplots(figsize=(25, 8))
fig = sns.boxplot(x='Store', y='Weekly_Sales', data=train, showfliers=False, hue="IsHoliday")
plt.title("Box Plot Using Size feature")
plt.show()


# In[73]:


fig, ax = plt.subplots(figsize=(10, 50))
fig = sns.boxplot(y='Dept', x='Weekly_Sales', data=train, showfliers=False, hue="Type",orient="h") 
plt.title("Box Plot Using Dept feature")
plt.show()


# In[74]:


fig, ax = plt.subplots(figsize=(25, 10))
fig = sns.boxplot(x='Dept', y='Weekly_Sales', data=train, showfliers=False, hue="IsHoliday")
plt.title("Box Plot Using Dept feature with holidays impact")
plt.show()


# In[75]:


#Average Department Sales - Per Year


# In[76]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go

dept_sales_2010 = train[train.Year==2010].groupby('Dept')['Weekly_Sales'].mean().to_dict()
dept2010_df = pd.DataFrame(list(dept_sales_2010.items()), columns=['Dept', 'AvgSales2010'])

dept_sales_2011 = train[train.Year==2011].groupby('Dept')['Weekly_Sales'].mean().to_dict()
dept2011_df = pd.DataFrame(list(dept_sales_2011.items()), columns=['Dept', 'AvgSales2011'])

dept_sales_2012 = train[train.Year==2012].groupby('Dept')['Weekly_Sales'].mean().to_dict()
dept2012_df = pd.DataFrame(list(dept_sales_2012.items()), columns=['Dept', 'AvgSales2012'])

fig = make_subplots(rows=1, cols=3, subplot_titles=("Average Dept Sales 2010", "Average Dept Sales 2011", "Average Dept Sales 2012"))

fig.add_trace(go.Bar(x=dept2010_df.AvgSales2010, y=dept2010_df.Dept, orientation='h',),1, 1)

fig.add_trace(go.Bar(x=dept2011_df.AvgSales2011, y=dept2011_df.Dept, orientation='h',),1, 2)

fig.add_trace(go.Bar(x=dept2012_df.AvgSales2012, y=dept2012_df.Dept, orientation='h',),1, 3)

fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), template = 'plotly_dark', showlegend=False, height=1000)

fig.update_xaxes(title_text="AvgSales", row=1, col=1)
fig.update_xaxes(title_text="AvgSales", row=1, col=2)
fig.update_xaxes(title_text="AvgSales", row=1, col=3)

fig.update_yaxes(title_text="Dept", row=1, col=1)
fig.update_yaxes(title_text="Dept", row=1, col=2)
fig.update_yaxes(title_text="Dept", row=1, col=3)

fig.update_yaxes(tick0=1, dtick=1)
fig.show()


# In[77]:


#Month wise Weekly Sales visualization


# In[78]:


fig, ax = plt.subplots(figsize=(25, 10))
fig = sns.boxplot(x='Month', y='Weekly_Sales', data=train, showfliers=False, hue="IsHoliday")
plt.title("Box Plot of weekly sales by Month and Holiday")
plt.show()


# In[79]:


train['Is_month_end'] = np.where(train.Day > 22, 1, 0)
train['Is_month_start'] = np.where(train.Day<7,1,0)
train['Is_month_end'] = train['Is_month_end'].astype('bool')
train['Is_month_start'] = train['Is_month_start'].astype('bool')


# In[80]:


#This function is creating eta square test
def correlation_ratio(categories, measurements):
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat)+1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)
        for i in range(0,cat_num):
            cat_measures = measurements[np.argwhere(fcat == i).flatten()]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures)
        y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
        numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
        denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
        if numerator == 0:
            eta = 0.0
        else:
            eta = numerator/denominator
        return eta


# In[81]:


print("Correlation of Is_month_end feature with weekly_sales")
print(correlation_ratio(train['Is_month_end'],train['Weekly_Sales']))
print(50*'*')
print("Correlation of Is_month_start feature with weekly_sales")
print(correlation_ratio(train['Is_month_start'],train['Weekly_Sales']))
print(50*'*')


# In[82]:


sales_month_end = train['Weekly_Sales'][train['Is_month_end']==1]
sales_not_month_end = train['Weekly_Sales'][train['Is_month_end']==0]
print("Mean of the sales in month end week: ", np.mean(sales_month_end))
print("Standard devition of the sales in month end week: ", np.std(sales_month_end))
print("Max sales in month end week: ", np.max(sales_month_end))
print("Mean of the sales other than month end week: ", np.mean(sales_not_month_end))
print("Standard devition of the sales other than month end week: ", np.std(sales_not_month_end))
print("Max sales other than month end week: ", np.max(sales_not_month_end))
print("*"*100)


# In[83]:


fig, ax = plt.subplots(figsize=(25, 10))
fig = sns.boxplot(x='Is_month_end', y='Weekly_Sales', data=train, showfliers=False)
plt.title("Box Plot Using Dept feature with holidays impact")
plt.show()


# In[84]:


sales_month_start = train['Weekly_Sales'][train['Is_month_start']==1]
sales_not_month_start = train['Weekly_Sales'][train['Is_month_start']==0]
print("Mean of the sales in month start week: ", np.mean(sales_month_start))
print("Standard devition of the sales in month start week: ", np.std(sales_month_start))
print("Max sales in month start week: ", np.max(sales_month_start))
print("Mean of the sales other than month start week: ", np.mean(sales_not_month_start))
print("Standard devition of the sales other than month start week: ", np.std(sales_not_month_start))
print("Max sales other than month start week: ", np.max(sales_not_month_start))
print("*"*100)


# In[85]:


fig, ax = plt.subplots(figsize=(25, 10))
fig = sns.boxplot(x='Is_month_end', y='Weekly_Sales', data=train, showfliers=False)
plt.title("Box Plot Using Dept feature with holidays impact")
plt.show()


# In[86]:


fig, ax = plt.subplots(figsize=(50, 50))
sns.distplot(train['Weekly_Sales'])
plt.title("PDF corresponding to Sales")
plt.show()


# In[87]:


sns.FacetGrid(train).map(sns.distplot,"CPI").add_legend();
plt.show


# In[88]:


train['CPI_category'] = pd.cut(train['CPI'],bins=[120,140,160,180,200,220])
fig, ax = plt.subplots(figsize=(25, 8))
fig = sns.boxplot(x='CPI_category', y='Weekly_Sales', data=train, showfliers=False)
plt.title("Box Plot Using CPI feature")
plt.show()


# In[89]:


sns.FacetGrid(train).map(sns.distplot,"Unemployment").add_legend();
plt.show


# In[90]:


train['Unemployment_category'] = pd.cut(train['Unemployment'],bins=[4,6,8,10,12,14,16])
fig, ax = plt.subplots(figsize=(25, 8))
fig = sns.boxplot(x='Unemployment_category', y='Weekly_Sales', data=train, showfliers=False)
plt.title("Box Plot Using CPI feature")
plt.show()


# In[91]:


sns.FacetGrid(train).map(sns.distplot,"Fuel_Price").add_legend();
plt.show


# In[92]:


train['fuel_price_category'] = pd.cut(train['Fuel_Price'],bins=[0,2.5,3,3.5,4,4.5])
fig, ax = plt.subplots(figsize=(25, 8))
fig = sns.boxplot(x='fuel_price_category', y='Weekly_Sales', data=train, showfliers=False)
plt.title("Box Plot Using CPI feature")
plt.show()


# In[93]:


sns.FacetGrid(train).map(sns.distplot,"Temperature").add_legend();
plt.show


# In[94]:


positive_temperature = train[train['Weekly_Sales']>0]['Temperature']
sns.distplot(positive_temperature)
plt.title("Histogram of Temperature when sale is positive")
plt.show()


# In[95]:


train['Temperature_category'] = pd.cut(train['Temperature'],bins=[0,20,40,60,80,100])
fig, ax = plt.subplots(figsize=(25, 8))
fig = sns.boxplot(x='Temperature_category', y='Weekly_Sales', data=train, showfliers=False)
plt.title("Box Plot Using CPI feature")
plt.show()


# In[96]:


g = train.groupby(["Month","Store"])
monthly_averages = g.aggregate({"Weekly_Sales":np.mean})
store_value=monthly_averages.loc[monthly_averages.groupby('Month')['Weekly_Sales'].idxmax()]
print("Stores which have highest value during that  Month: ")
store_value


# In[97]:


#Creating dateset for training purpopse


# In[98]:


train.info()


# In[101]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming train and test DataFrames are already defined

# Mapping store type to numeric values
storetype_values = {'A': 3, 'B': 2, 'C': 1}
train['Type_Numeric'] = train['Type'].map(storetype_values)
test['Type_Numeric'] = test['Type'].map(storetype_values)

# Drop the original 'Type' column to avoid issues with non-numeric values
train = train.drop(columns=['Type'])
test = test.drop(columns=['Type'])

# Ensure all columns are numeric or can be converted to numeric
numeric_train = train.select_dtypes(include=[int, float, 'number'])

# Plotting the correlation matrix
plt.figure(figsize=(28,14))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Generate the heatmap
sns.heatmap(numeric_train.corr(), cmap='Reds', annot=True, annot_kws={'size':12})
plt.title('Correlation Matrix', fontsize=30)
plt.show()


# In[103]:


ain = train.drop(['Date', 'Temperature','Fuel_Price', 'Temperature_category','fuel_price_category','Unemployment_category','CPI_category','Is_month_start','Is_month_end','n_days', 'MarkDown1', 'MarkDown2', 'MarkDown3',
             'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 'Month', 'Day' ], axis=1)

test = test.drop(['Date', 'Temperature','Fuel_Price', 'n_days','Month','Day', 'MarkDown1', 'MarkDown2', 'MarkDown3',
             'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment'], axis=1)


# In[104]:


# Identify input and target columns

input_cols = train.columns.to_list()
input_cols.remove('Weekly_Sales')
target_col = 'Weekly_Sales'

inputs_df = train[input_cols].copy()
targets = train[target_col].copy()


# In[109]:


# Create training and validation sets
#X_train = train_inputs
#x test = val_inputs
#y train = train_targets
#y_test = val_targets

from sklearn.model_selection import train_test_split

train_inputs, val_inputs, train_targets, val_targets = train_test_split(
    inputs_df, targets, test_size=0.3, random_state=42)


# In[110]:


# Define the function to evaluate the models

def WMAE(df, targets, predictions):
    weights = df.IsHoliday.apply(lambda x: 5 if x else 1)
    return np.round(np.sum(weights*abs(targets-predictions))/(np.sum(weights)), 2)


# In[111]:


#linear regresssion


# In[114]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Function to calculate Weighted Mean Absolute Error (WMAE)
def WMAE(inputs, targets, preds, weights=None):
    if weights is None:
        weights = inputs['IsHoliday'].apply(lambda x: 5 if x else 1)
    return (weights * abs(targets - preds)).sum() / weights.sum()

# Drop columns that are not needed
drop_cols = ['Date', 'Temperature','Fuel_Price', 'Type', 'n_days', 'Month', 'Day', 'MarkDown1', 'MarkDown2', 
             'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment']
train = train.drop(columns=drop_cols, errors='ignore')
test = test.drop(columns=drop_cols, errors='ignore')

# Define input columns by selecting only numerical columns
input_cols = train.select_dtypes(include=[int, float, 'number']).columns.tolist()

# Ensure 'Weekly_Sales' (or the target variable) is not in input_cols
if 'Weekly_Sales' in input_cols:
    input_cols.remove('Weekly_Sales')

# Check and remove datetime and interval columns if present
datetime_cols = train.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
interval_cols = train.select_dtypes(include=['interval']).columns.tolist()
input_cols = [col for col in input_cols if col not in datetime_cols and col not in interval_cols]

# Scaling
scaler = MinMaxScaler().fit(train[input_cols])

train[input_cols] = scaler.transform(train[input_cols])
test[input_cols] = scaler.transform(test[input_cols])

# Split the data into training and validation sets
train_inputs, val_inputs, train_targets, val_targets = train_test_split(
    train[input_cols], train['Weekly_Sales'], test_size=0.3, random_state=42)

# Create and train the model
model = LinearRegression().fit(train_inputs, train_targets)

# Generate predictions on training data
train_preds = model.predict(train_inputs)

# Compute WMAE on training data
train_wmae = WMAE(train_inputs, train_targets, train_preds)
print('The WMAE loss for the training set is {}.'.format(train_wmae))

# Generate predictions on validation data
val_preds = model.predict(val_inputs)

# Compute WMAE on validation data
val_wmae = WMAE(val_inputs, val_targets, val_preds)
print('The WMAE loss for the validation set is {}.'.format(val_wmae))


# In[115]:


model.score(val_inputs,val_targets)


# In[116]:


model.score(train_inputs,train_targets)


# In[117]:


#Ridge regression


# In[118]:


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Create and train the model
model_ridge = Ridge().fit(train_inputs, train_targets)

# Generate predictions on training data
train_preds = model_ridge.predict(train_inputs)

# Compute WMAE on traing data
#X_train = train_inputs
#x test = val_inputs
#y train = train_targets
#y_test = val_targets

train_wmae = WMAE(train_inputs, train_targets, train_preds)
print('The WMAE loss for the training set is  {}.'.format(train_wmae))

# Generate predictions on validation data
val_preds = model_ridge.predict(val_inputs)

# Compute WMAE on validation data
val_wmae = WMAE(val_inputs, val_targets, val_preds)
print('The WMAE loss for the validation set is  {}.'.format(val_wmae))


# In[119]:


model_ridge.score(val_inputs,val_targets)


# In[120]:


model_ridge.score(train_inputs,train_targets)


# In[121]:


#Decision Tree


# In[122]:


from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor()

tree.fit(train_inputs, train_targets)

tree_train_preds = tree.predict(train_inputs)

# Compute WMAE on traing data
tree_train_wmae = WMAE(train_inputs, train_targets, tree_train_preds)
print('The WMAE loss for the training set is  {}.'.format(tree_train_wmae))


# Compute WMAE on validation data
tree_val_preds = tree.predict(val_inputs)
tree_val_wmae = WMAE(val_inputs, val_targets, tree_val_preds)
print('The WMAE loss for the validation set is  {}.'.format(tree_val_wmae))


# In[123]:


tree.score(train_inputs,train_targets)


# In[126]:


tree.score(val_inputs,val_targets)


# In[127]:


importance_df = pd.DataFrame({
    'feature': train_inputs.columns,
    'importance': tree.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(16,10))
plt.title('Feature Importance')
sns.barplot(data=importance_df, x='importance', y='feature');


# In[128]:


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text
import seaborn as sns
sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[129]:


plt.figure(figsize=(30,15))
plot_tree(tree, feature_names=train_inputs.columns, max_depth=3, filled=True);


# In[130]:


#X_train = train_inputs
#x test = val_inputs
#y train = train_targets
#y_test = val_targets
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=0)
dt.fit(train_inputs,train_targets)
y_pred = dt.predict(val_inputs)


# In[131]:


accuracy = dt.score(val_inputs,val_targets)
accuracy


# In[132]:


#Random Forest


# In[133]:


from sklearn.ensemble import RandomForestRegressor

# Create the model
rf1 = RandomForestRegressor(n_jobs=-1, random_state=42)

# Fit the model
rf1.fit(train_inputs, train_targets)

rf1_train_preds = rf1.predict(train_inputs)

# Compute WMAE on traing data
rf1_train_wmae = WMAE(train_inputs, train_targets, rf1_train_preds)
print('The WMAE loss for the training set is  {}.'.format(rf1_train_wmae))

rf1_val_preds = rf1.predict(val_inputs)

# Compute WMAE on validation data
rf1_val_wmae = WMAE(val_inputs, val_targets, rf1_val_preds)
print('The WMAE loss for the validation set is  {}.'.format(rf1_val_wmae))


# In[134]:


rf1.score(val_inputs,val_targets)


# In[135]:


importance_df = pd.DataFrame({
    'feature': train_inputs.columns,
    'importance': rf1.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(16,10))
plt.title('Feature Importance')
sns.barplot(data=importance_df, x='importance', y='feature');


# In[136]:


def test_params(**params):  
    model = RandomForestRegressor(random_state=42, n_jobs=-1, **params).fit(train_inputs, train_targets)
    train_wmae = WMAE(train_inputs, train_targets, model.predict(train_inputs))
    val_wmae = WMAE(val_inputs, val_targets, model.predict(val_inputs))
    return train_wmae, val_wmae


# In[137]:


def test_param_and_plot(param_name, param_values):
    train_errors, val_errors = [], [] 
    for value in param_values:
        params = {param_name: value}
        train_wmae, val_wmae = test_params(**params)
        train_errors.append(train_wmae)
        val_errors.append(val_wmae)
    plt.figure(figsize=(16,8))
    plt.title('Overfitting curve: ' + param_name)
    plt.plot(param_values, train_errors, 'b-o')
    plt.plot(param_values, val_errors, 'r-o')
    plt.xlabel(param_name)
    plt.ylabel('WMAE')
    plt.legend(['Training', 'Validation'])


# In[140]:


test_param_and_plot('max_depth', [5, 10, 15, 20, 25, 30, 35])


# In[141]:


test_param_and_plot('n_estimators', [10, 30, 50, 70, 90, 100])


# In[142]:


test_param_and_plot('min_samples_split', [2, 3, 4, 5, 6, 7, 8, 9, 10])


# In[143]:


test_param_and_plot('min_samples_leaf', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


# In[144]:


test_param_and_plot('max_samples', [0.2, 0.4, 0.6, 0.8, 1])


# In[145]:


test_param_and_plot('max_features', [2, 3, 4, 5, 6, 7])


# In[146]:


from sklearn.ensemble import RandomForestRegressor

# Create the model
rf1 = RandomForestRegressor(n_jobs=-1, max_depth=30, n_estimators=130, min_samples_split=2, min_samples_leaf=1, 
                            max_samples=0.99999, max_features=6,  random_state=42)

# Fit the model
rf1.fit(train_inputs, train_targets)

rf1_train_preds = rf1.predict(train_inputs)

# Compute WMAE on traing data
rf1_train_wmae = WMAE(train_inputs, train_targets, rf1_train_preds)
print('The WMAE loss for the training set is  {}.'.format(rf1_train_wmae))

rf1_val_preds = rf1.predict(val_inputs)

# Compute WMAE on validation data
rf1_val_wmae = WMAE(val_inputs, val_targets, rf1_val_preds)
print('The WMAE loss for the validation set is  {}.'.format(rf1_val_wmae))


# In[147]:


rf1.score(val_inputs,val_targets)


# In[148]:


#Gradient Boosting


# In[150]:


pip install xgboost


# In[151]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Function to calculate Weighted Mean Absolute Error (WMAE)
def WMAE(inputs, targets, preds, weights=None):
    if weights is None:
        weights = inputs['IsHoliday'].apply(lambda x: 5 if x else 1)
    return (weights * abs(targets - preds)).sum() / weights.sum()

# Drop columns that are not needed
drop_cols = ['Date', 'Temperature','Fuel_Price', 'Type', 'n_days', 'Month', 'Day', 'MarkDown1', 'MarkDown2', 
             'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment']
train = train.drop(columns=drop_cols, errors='ignore')
test = test.drop(columns=drop_cols, errors='ignore')

# Define input columns by selecting only numerical columns
input_cols = train.select_dtypes(include=[int, float, 'number']).columns.tolist()

# Ensure 'Weekly_Sales' (or the target variable) is not in input_cols
if 'Weekly_Sales' in input_cols:
    input_cols.remove('Weekly_Sales')

# Check and remove datetime and interval columns if present
datetime_cols = train.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
interval_cols = train.select_dtypes(include=['interval']).columns.tolist()
input_cols = [col for col in input_cols if col not in datetime_cols and col not in interval_cols]

# Scaling
scaler = MinMaxScaler().fit(train[input_cols])

train[input_cols] = scaler.transform(train[input_cols])
test[input_cols] = scaler.transform(test[input_cols])

# Split the data into training and validation sets
train_inputs, val_inputs, train_targets, val_targets = train_test_split(
    train[input_cols], train['Weekly_Sales'], test_size=0.3, random_state=42)

# Create the model
gbm = XGBRegressor(random_state=42, n_jobs=-1)

# Fit the model
gbm.fit(train_inputs, train_targets)

# Generate predictions on training data
gbm_train_preds = gbm.predict(train_inputs)

# Compute WMAE on training data
gbm_train_wmae = WMAE(train_inputs, train_targets, gbm_train_preds)
print('The WMAE loss for the training set is  {}.'.format(gbm_train_wmae))

# Generate predictions on validation data
gbm_val_preds = gbm.predict(val_inputs)

# Compute WMAE on validation data
gbm_val_wmae = WMAE(val_inputs, val_targets, gbm_val_preds)
print('The WMAE loss for the validation set is  {}.'.format(gbm_val_wmae))


# In[152]:


gbm.score(val_inputs,val_targets)


# In[153]:


importance_df = pd.DataFrame({
    'feature': train_inputs.columns,
    'importance': gbm.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(16,10))
plt.title('Feature Importance')
sns.barplot(data=importance_df, x='importance', y='feature');


# In[154]:


#Tuning of Model Parameters


# In[155]:


def test_params_xgb(**params):  
    model = XGBRegressor(random_state=42, n_jobs=-1, **params).fit(train_inputs, train_targets)
    train_wmae = WMAE(train_inputs, train_targets, model.predict(train_inputs))
    val_wmae = WMAE(val_inputs, val_targets, model.predict(val_inputs))
    return train_wmae, val_wmae


# In[156]:


def test_param_and_plot_xgb(param_name, param_values):
    train_errors, val_errors = [], [] 
    for value in param_values:
        params = {param_name: value}
        train_wmae, val_wmae = test_params_xgb(**params)
        train_errors.append(train_wmae)
        val_errors.append(val_wmae)
    plt.figure(figsize=(16,8))
    plt.title('Overfitting curve: ' + param_name)
    plt.plot(param_values, train_errors, 'b-o')
    plt.plot(param_values, val_errors, 'r-o')
    plt.xlabel(param_name)
    plt.ylabel('WMAE')
    plt.legend(['Training', 'Validation'])


# In[157]:


test_param_and_plot_xgb('n_estimators', [100, 200, 300, 400, 500])


# In[158]:


test_param_and_plot_xgb('max_depth', [5, 10, 15, 20])


# In[159]:


test_param_and_plot_xgb('learning_rate', [0.2, 0.4, 0.6, 0.8, 0.9])


# In[160]:


from xgboost import XGBRegressor

# Create the model
gbm = XGBRegressor(random_state=42, n_jobs=-1, n_estimators=400, max_depth=15, learning_rate=0.35)

# Fit the model
gbm.fit(train_inputs, train_targets)

gbm_train_preds = gbm.predict(train_inputs)

# Compute WMAE on traing data
gbm_train_wmae = WMAE(train_inputs, train_targets, gbm_train_preds)
print('The WMAE loss for the training set is  {}.'.format(gbm_train_wmae))

gbm_val_preds = gbm.predict(val_inputs)

# Compute WMAE on test data
gbm_val_wmae = WMAE(val_inputs, val_targets, gbm_val_preds)
print('The WMAE loss for the validation set is  {}.'.format(gbm_val_wmae))


# In[161]:


gbm.score(val_inputs,val_targets)


# In[162]:


#ALL MODEL COMPARISON


# In[163]:


pip install PrettyTable


# In[164]:


from prettytable import PrettyTable
    
x = PrettyTable()
x.field_names = ["Model" ,"Accuracy","WMAE"]
x.add_row(["Linear Regression", 85.6, 14831.1])
x.add_row(["Ridge Regression", 86.2, 14831.07])
x.add_row(["Gradient Boosting", 94.4,3111.80])
x.add_row(["Decision Tree", 96.1, 1908.01])
x.add_row(["Extremely Randomized Trees (Extra Trees)", 96.6, 1523.16 ])
x.add_row(["Random Forest Regression", 97.5, 1576.55])
x.add_row(["Random Forest after fine tuning", 97.6, 1566.76])
x.add_row(["Gradient Boosting after fine tuning", 98.5, 1328.82 ])

print(x)


# In[165]:


#Making Predictions


# In[166]:


test_preds = gbm.predict(test)
test['Weekly_Sales'] = test_preds


# In[167]:


test_preds


# In[ ]:




