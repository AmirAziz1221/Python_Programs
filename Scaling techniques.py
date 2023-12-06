#!/usr/bin/env python
# coding: utf-8

# # 1). Preprocessing 
#  - Anamelly
#       - outliers
#       - Structures
#       - Data
#        - Type
#   - Missing values
#   - Feature Scalling 
#       - bring data into a scale
#       - min_max scaling
#       - standard scaling(Z-Score normalization)
#       - Roburst scaling
#       - logrithmic scaling
#   - Lables (dependent or y)
#   - Features (Independent or X)

# # 1). Min Max Scaling

# In[30]:


from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


# In[31]:


#simple data
data={
    'Value': [10,30,20,40,50]
}
df=pd.DataFrame(data)


# In[32]:


df


# In[33]:


scaler = MinMaxScaler()
df['Scaled_value'] = scaler.fit_transform(df[['Value']])
df


# # 2). Standard Scaler or Z-Score normalization

# In[34]:


from sklearn.preprocessing import StandardScaler
scaler2 = StandardScaler()
df['Scaled_value'] = scaler2.fit_transform(df[['Value']])
df


# # 3). Robust Scaler

# In[35]:


from sklearn.preprocessing import RobustScaler
scaler3 = RobustScaler()
df['Scaled_value'] = scaler3.fit_transform(df[['Value']])
df


# # 4). Logrithmic Scaling or Normalization

# In[47]:


data1={
   'Numbers':[10000,30000,50000,70000,90000]
}
df1=pd.DataFrame(data1)


# In[48]:


df1


# In[51]:


df1['log10_of_Numbers'] = np.log10(df1['Numbers'])


# In[52]:


df1['log_of_Numbers'] = np.log(df1['Numbers'])


# In[54]:


df1['log2_of_Numbers'] = np.log2(df1['Numbers'])


# In[55]:


df1

