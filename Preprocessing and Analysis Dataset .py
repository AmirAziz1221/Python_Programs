#!/usr/bin/env python
# coding: utf-8

# # PreProcessing 

# # importing libraries

# In[95]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Data Loading and Exploration | Cleaning 

# In[96]:


df=pd.read_csv('googleplaystore.csv')
df.head()


# In[97]:


# set options to be maximum for rows and columns
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


# In[98]:


# Hide all warnings
import warnings
warnings.filterwarnings('ignore')


# In[99]:


df.columns


# In[100]:


print("The number of rows are",df.shape[0]," and columns are",df.shape[1])


# In[101]:


df.info()


# In[102]:


df.describe()


# # how to make Size a numeric column?

# In[103]:


df['Size'].unique()


# ### Observations
# - veries with device
# - M
# - K

# In[104]:


df['Size'].isnull().sum()


# - No missing value in size 

# In[105]:


#verify the number of values| obsrtvation
df['Size'].loc[df['Size'].str.contains('M')].value_counts().sum()


# In[106]:


df['Size'].loc[df['Size'].str.contains('k')].value_counts().sum()


# In[107]:


df['Size'].loc[df['Size'].str.contains('Varies with device')].value_counts().sum()


# In[108]:


8830+316+1695


# In[109]:


len(df)


# In[110]:


# convert the whole size column into bytes
# let's define a function
def convert_size(size):
    if isinstance(size, str):
        if 'k' in size:
            return float(size.replace('k', ""))*1024
        elif 'M' in size:
            return float(size.replace('M', ""))*1024*1024
        elif 'Varies with  device' in size:
            return np.nan
    return size


# In[111]:


df['Size']


# In[112]:


# let's apply this funtion
df['Size']=df['Size'].apply(convert_size)


# In[113]:


df['Size']


# In[114]:


#rename
df.rename(columns={'Size':"Size_in_bytes"},inplace=True)


# In[115]:


df.head()


# In[116]:


df['Size_in_bytes'] = pd.to_numeric(df['Size_in_bytes'], errors='coerce')
df['Size_in_Mb'] = df['Size_in_bytes'] / (1024 * 1024)


# In[117]:


#let's take care of installs
df['Installs'].unique()


# In[118]:


df['Installs'].value_counts()


# In[119]:


df['Installs'].isnull().sum()


# - no missing values

# ### Observations
# - remove + sign
# - remove ,
# - Convert the column into an integer

# In[120]:


df['Installs'] = df['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)


# In[121]:


df['Installs'] = df['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)


# In[122]:


df['Installs']= df['Installs'].apply(lambda x: int(x))


# In[123]:


df["Installs"].value_counts()


# In[124]:


df["Installs"].value_counts()


# In[125]:


df.describe()


# # Price column 

# In[126]:


df['Price'].value_counts()


# ### Observations 
# - $ sign

# In[127]:


df['Price'].loc[df['Price'].str.contains('\$')].value_counts().sum()


# In[128]:


df['Price'].loc[(df['Price'].str.contains('0')) & (~df['Price'].str.contains('\$'))].value_counts().sum()


# In[129]:


df['Price'] = df['Price'].apply(lambda x: x.replace('$', '') if '$' in str(x) else x)


# In[130]:


df['Price'].value_counts()


# In[131]:


# Now we can conver it into numerics


# In[132]:


df['Price'] = df['Price'].apply(lambda x: float(x))


# In[133]:


df['Price']


# In[134]:


df.describe()


# In[135]:


# using  string print min, max and average prices of the app
print("Min Price is", df['Price'].min())
print("Max Price is", df['Price'].max())
print("Average Price is", df['Price'].mean())


# In[136]:


# missing values inside the Data


# In[137]:


df.isnull().sum().sort_values(ascending=False)


# In[138]:


# find missing values percentage in data 
round(df.isnull().sum()/len(df)*100,2).sort_values(ascending=False)


# In[139]:


# total number of missing values
df.isnull().sum().sum()


# In[140]:


#Plot missing values
plt.figure(figsize=(16,8))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False, cmap='viridis')


# In[141]:


import matplotlib.pyplot as plt

plt.figure(figsize=(16, 8))
missing_percentage = df.isnull().sum() / len(df) * 100
missing_percentage.plot(kind='bar')

# Adding labels and title
plt.xlabel('Columns')
plt.ylabel('Percentage')
plt.title('Percentage of missing values in each column')

plt.show()  # Display the plot


# In[142]:


plt.figure(figsize=(16, 8))
missing_percentage = df.isnull().sum() / len(df) * 100
missing_percentage[missing_percentage < 1].plot(kind='bar')

# Adding labels and title
plt.xlabel('Columns')
plt.ylabel('Percentage')
plt.title('Percentage of missing values in each column')

plt.show()  # Display the plot


# # From Last Updated Extract 
# - day
# - month
# - year

# In[143]:


# Convert 'Last Updated' column to datetime format
df['Last Updated'] = pd.to_datetime(df['Last Updated'])

# Extract day, month, and year into separate columns
df['Day'] = df['Last Updated'].dt.day
df['Month'] = df['Last Updated'].dt.month
df['Year'] = df['Last Updated'].dt.year


# In[144]:


df.head()


# In[145]:


from scipy import stats


# # Dealing with the Missing values

# In[146]:


(df.isnull().sum()/len(df)*100).sort_values(ascending=False)


# In[147]:


df.describe()


# In[ ]:





# In[148]:


# make a correlation matrix of numeric columns
plt.figure(figsize=(16,10))
numeric_cols = ['Rating','Reviews','Size_in_bytes','Installs','Price','Size_in_Mb']
sns.heatmap(df[numeric_cols].corr(), annot = True)


# In[149]:


df[numeric_cols].corr()


# In[150]:


#remove rows containing NaN or infinite Values
df_clean=df.dropna()
# Calculate Pearson's R between Reviews and Installs
pearson_r = stats.pearsonr(df_clean['Reviews'], df_clean['Installs'])
print("Pearson's R between Reviews and Installs:\n ", pearson_r)


# In[151]:


print("Lenth if DataFrame before removinig null values:",len(df))


# In[152]:


#Remove the rows having null values in the 
df.dropna(subset=['Current Ver','Android Ver','Category','Type','Genres'], inplace=True)


# In[153]:


# length after removing null values
print("Lenght of the DataFrame after romoving null values:",len(df))


# - we have removed 12 rows having null values in the Current Ver,Android Ver,Category,Type and Genres columns.

# In[154]:


print(df.columns)


# In[155]:


# lets check again null values
df.isnull().sum().sort_values(ascending=False)
df['Installs'].loc[df['Rating'].isnull()].value_counts()


# In[156]:


df.columns


# In[157]:


# use groupby function to find the trend of Rating in each Installs_category
df.groupby('Installs')['Rating'].describe()


# In[158]:


plt.figure(figsize=(16,8))
sns.boxplot(x='Installs', y='Rating',data=df)


# In[159]:


plt.figure(figsize=(16,6))
sns.scatterplot(x='Rating',y='Reviews',hue='Installs',data=df)


# In[160]:


plt.figure(figsize=(16,6))
sns.scatterplot(x='Reviews',y='Installs',data=df)


# In[161]:


plt.figure(figsize=(16,6))
sns.scatterplot(x=np.log10(df['Reviews']), y=np.log10(df['Installs']),data=df)


# In[162]:


plt.figure(figsize=(16,6))
sns.lmplot(x='Reviews',y='Installs',data=df)


# # Observation 
#   - Rating and Reviews is directly proportional to the Installation

# # Duplicates
#   - romoving duplicate values because for accuracy  

# In[163]:


# Total duplicate values
df.duplicated().sum()


# In[164]:


# find duplicate if any in the 'App' column
df['App'].duplicated().sum()


# In[165]:


for col in df.columns:
    print('Number of duplicates in ', col,' column are:',df[col].duplicated().sum())


#   - this mean that the only better way to find duplicates is to check for whole data

# In[166]:


# print the number of duplicated in df
print('Number of duplicated in df are: ', df.duplicated().sum())


#   - find and watch all duplicates if they are real!

# In[167]:


#find exact duplicates and print them
df[df['App'].duplicated(keep=False)].sort_values(by='App')


# # Now 
#   - Remove Duplicates

# In[168]:


df.drop_duplicates(inplace=True)


# In[169]:


# Print the number of rows and columns after removing duplicates
print('Number of rows after removinig Duplicates: ', df.shape[0],' and Columns: ' , df.shape[1])


#   - Now we have removed 483 duplicates from the dataset and have 10346 rows left

# ---

# # 3. Insights from Data

# ### a). which category has the highest number of apps?

# In[170]:


# To show top 10 highest Category number of apps
df['Category'].value_counts().head(10)


# ### b). Which category has the highest number od installs?

# In[171]:


df.groupby('Category')['Installs'].sum().sort_values(ascending = False).head(10)


# ### c). Which category has the highest number of reviews?

# In[172]:


df.groupby('Category')['Reviews'].sum().sort_values(ascending=False).head(10)


# ### d). Which category has the highest rating ?

# In[173]:


# Highest Rating
df.groupby('Category')['Rating'].max().sort_values(ascending=False).head(10)


# In[174]:


# Average highest rationg
df.groupby('Category')['Rating'].mean().sort_values(ascending=False).head(10)


# ### Plot Rating Density

# In[177]:


# Plot the rating distribution
plt.figure(figsize=(16,6))
sns.kdeplot(df['Rating'], color='blue', shade=True)


# ---
# # Very important for All

# In[182]:


# plot number of install for free vs paid apps make a bar plot
plt.figure(figsize=(18,8))
sns.barplot(x='Type', y='Installs', data= df)


# In[183]:


#show scatter plot as well where x-axis is Installs ans y-axis is price ans hue is Type
plt.figure(figsize=(18,8))
sns.scatterplot(x='Installs', y='Price', hue='Type', data=df)


# In[ ]:




