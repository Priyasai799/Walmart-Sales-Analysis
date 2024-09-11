#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np      # To use np.arrays
import pandas as pd     # To use dataframes
from pandas.plotting import autocorrelation_plot as auto_corr

# To plot
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import seaborn as sns

#For date-time
import math
from datetime import datetime
from datetime import timedelta

# Another imports if needs
import itertools
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose as season
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.linear_model import LinearRegression 
from sklearn import preprocessing

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
get_ipython().system('pip install pmdarima')
from pmdarima.utils import decomposed_plot
from pmdarima.arima import decompose
from pmdarima import auto_arima


import warnings
warnings.filterwarnings("ignore")
import seaborn as sns


# In[2]:


#machine learning models libraries
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNet


# In[3]:


#Preprocessing related Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


# In[4]:


pd.options.display.max_columns=100


# In[5]:


df_store = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\walmart sales\walmart stores.csv")


# In[6]:


df_train = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\walmart sales\train.csv\train.csv")


# In[7]:


df_features = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\walmart sales\walmart features.csv")


# In[8]:


df_store.head()


# In[9]:


df_train.head()


# In[10]:


df_features.head()


# In[11]:


# merging 3 different sets
df = df_train.merge(df_features, on=['Store', 'Date'], how='inner').merge(df_store, on=['Store'], how='inner')
df.head(5)


# In[12]:


df.drop(['IsHoliday_y'], axis=1,inplace=True) # removing dublicated column


# In[13]:


df.rename(columns={'IsHoliday_x':'IsHoliday'},inplace=True) # rename the column


# In[14]:


df.head() 


# In[15]:


df.shape


# In[16]:


df['Store'].nunique()


# In[17]:


df['Dept'].nunique()


# In[18]:


store_dept_table = pd.pivot_table(df, index='Store', columns='Dept',
                                  values='Weekly_Sales', aggfunc=np.mean)
display(store_dept_table)


# In[19]:


df.loc[df['Weekly_Sales']<=0]


# In[20]:


df.loc[df['Weekly_Sales'] > 0]


# In[21]:


df = df.loc[df['Weekly_Sales'] > 0]


# In[22]:


df.shape  # new data shape


# In[23]:


df['Date'].head(5)


# In[24]:


df['Date'].tail(5)


# In[25]:


#IS HOLIDAY COLUMN
sns.barplot(x='IsHoliday', y='Weekly_Sales', data=df)


# In[26]:


df_holiday = df.loc[df['IsHoliday']==True]
df_holiday['Date'].unique() 


# In[27]:


# Super bowl dates in train set
df.loc[(df['Date'] == '2010-02-12')|(df['Date'] == '2011-02-11')|(df['Date'] == '2012-02-10'),'Super_Bowl'] = True
df.loc[(df['Date'] != '2010-02-12')&(df['Date'] != '2011-02-11')&(df['Date'] != '2012-02-10'),'Super_Bowl'] = False


# In[28]:


# Labor day dates in train set
df.loc[(df['Date'] == '2010-09-10')|(df['Date'] == '2011-09-09')|(df['Date'] == '2012-09-07'),'Labor_Day'] = True
df.loc[(df['Date'] != '2010-09-10')&(df['Date'] != '2011-09-09')&(df['Date'] != '2012-09-07'),'Labor_Day'] = False


# In[29]:


# Thanksgiving dates in train set
df.loc[(df['Date'] == '2010-11-26')|(df['Date'] == '2011-11-25'),'Thanksgiving'] = True
df.loc[(df['Date'] != '2010-11-26')&(df['Date'] != '2011-11-25'),'Thanksgiving'] = False


# In[30]:


#Christmas dates in train set
df.loc[(df['Date'] == '2010-12-31')|(df['Date'] == '2011-12-30'),'Christmas'] = True
df.loc[(df['Date'] != '2010-12-31')&(df['Date'] != '2011-12-30'),'Christmas'] = False


# In[31]:


sns.barplot(x='Christmas', y='Weekly_Sales', data=df) # Christmas holiday vs not-Christmas


# In[32]:


sns.barplot(x='Thanksgiving', y='Weekly_Sales', data=df) # Thanksgiving holiday vs not-thanksgiving


# In[33]:


sns.barplot(x='Super_Bowl', y='Weekly_Sales', data=df) # Super bowl holiday vs not-super bowl


# In[34]:


sns.barplot(x='Labor_Day', y='Weekly_Sales', data=df) # Labor day holiday vs not-labor day


# In[35]:


df.groupby(['Christmas','Type'])['Weekly_Sales'].mean()  # Avg weekly sales for types on Christmas


# In[36]:


df.groupby(['Labor_Day','Type'])['Weekly_Sales'].mean()  # Avg weekly sales for types on Labor Day


# In[37]:


df.groupby(['Thanksgiving','Type'])['Weekly_Sales'].mean()  # Avg weekly sales for types on Thanksgiving


# In[38]:


df.groupby(['Super_Bowl','Type'])['Weekly_Sales'].mean()  # Avg weekly sales for types on Super Bowl


# In[39]:


my_data = [48.88, 37.77 , 13.33 ]  #percentages
my_labels = 'Type A','Type B', 'Type C' # labels
plt.pie(my_data,labels=my_labels,autopct='%1.1f%%', textprops={'fontsize': 15}) #plot pie type and bigger the labels
plt.axis('equal')
mpl.rcParams.update({'font.size': 20}) #bigger percentage labels

plt.show()


# In[40]:


df.groupby('IsHoliday')['Weekly_Sales'].mean()


# In[41]:


# Plotting avg wekkly sales according to holidays by types
plt.style.use('seaborn-poster')
labels = ['Thanksgiving', 'Super_Bowl', 'Labor_Day', 'Christmas']
A_means = [27397.77, 20612.75, 20004.26, 18310.16]
B_means = [18733.97, 12463.41, 12080.75, 11483.97]
C_means = [9696.56,10179.27,9893.45,8031.52]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(16, 8))
rects1 = ax.bar(x - width, A_means, width, label='Type_A')
rects2 = ax.bar(x , B_means, width, label='Type_B')
rects3 = ax.bar(x + width, C_means, width, label='Type_C')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Weekly Avg Sales')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.axhline(y=17094.30,color='r') # holidays avg
plt.axhline(y=15952.82,color='green') # not-holiday avg

fig.tight_layout()

plt.show()


# In[42]:


df.sort_values(by='Weekly_Sales',ascending=False).head(5)


# In[43]:


#TO SEE THE SIZE - TYPE RELATION


# In[44]:


df_store.groupby('Type').describe()['Size'].round(2) # See the Size-Type relation


# In[45]:


plt.figure(figsize=(10,8)) # To see the type-size relation
fig = sns.boxplot(x='Type', y='Size', data=df, showfliers=False)


# In[46]:


df.isna().sum()


# In[47]:


df = df.fillna(0) # filling null's with 0


# In[48]:


df.isna().sum() # last null check


# In[49]:


df.describe() # to see weird statistical things


# In[50]:


# DEEPER LOOK IN SALES


# In[51]:


x = df['Dept']
y = df['Weekly_Sales']
plt.figure(figsize=(15,5))
plt.title('Weekly Sales by Department')
plt.xlabel('Departments')
plt.ylabel('Weekly Sales')
plt.scatter(x,y)
plt.show()


# In[52]:


plt.figure(figsize=(30,10))
fig = sns.barplot(x='Dept', y='Weekly_Sales', data=df)


# In[53]:


x = df['Store']
y = df['Weekly_Sales']
plt.figure(figsize=(15,5))
plt.title('Weekly Sales by Store')
plt.xlabel('Stores')
plt.ylabel('Weekly Sales')
plt.scatter(x,y)
plt.show()


# In[54]:


plt.figure(figsize=(20,6))
fig = sns.barplot(x='Store', y='Weekly_Sales', data=df)


# In[55]:


# CHANGING DATE TO DATETIME AND CREATING NEW COLUMNS


# In[56]:


df["Date"] = pd.to_datetime(df["Date"]) # convert to datetime
df['week'] =df['Date']
df['month'] =df['Date'].dt.month 
df['year'] =df['Date'].dt.year


# In[57]:


df.groupby('month')['Weekly_Sales'].mean() # to see the best months for sales


# In[58]:


df.groupby('year')['Weekly_Sales'].mean() # to see the best years for sales


# In[59]:


monthly_sales = pd.pivot_table(df, values = "Weekly_Sales", columns = "year", index = "month")
monthly_sales.plot()


# In[60]:


fig = sns.barplot(x='month', y='Weekly_Sales', data=df)


# In[61]:


df.groupby('week')['Weekly_Sales'].mean().sort_values(ascending=False).head()


# In[62]:


weekly_sales = pd.pivot_table(df, values = "Weekly_Sales", columns = "year", index = "week")
weekly_sales.plot()


# In[63]:


plt.figure(figsize=(20,6))
fig = sns.barplot(x='week', y='Weekly_Sales', data=df)


# In[64]:


# FUEL PRICE, CPI, UNEMPLOYEMENT, TEMPERATURE EFFECTS


# In[65]:


fuel_price = pd.pivot_table(df, values = "Weekly_Sales", index= "Fuel_Price")
fuel_price.plot()


# In[66]:


temp = pd.pivot_table(df, values = "Weekly_Sales", index= "Temperature")
temp.plot()


# In[67]:


unemployment = pd.pivot_table(df, values = "Weekly_Sales", index= "Unemployment")
unemployment.plot()

