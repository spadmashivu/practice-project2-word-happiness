#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv("worldhappy.csv")


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.describe()


# In[ ]:





# In[7]:


df['Happiness Score'].unique()


# In[8]:


df['Happiness Score'].unique().shape


# In[9]:


dfcorr=df.corr()
dfcorr


# In[10]:


plt.figure(figsize=(10,5))
sns.heatmap(dfcorr,cmap='Greens',annot=True)


# In[11]:


df1=df['Region'].unique()
df1


# In[12]:


sns.pairplot(df)


# In[13]:


df.plot(kind='box',subplots=True,layout=(3,4))


# In[16]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder


# In[17]:


le=LabelEncoder()
df['Region']=le.fit_transform(df['Region'])
df['Region']


# In[18]:


df=df.drop(['Country'],axis=1)
df


# In[19]:


from scipy.stats import zscore
z=np.abs(zscore(df))
z


# In[20]:


threshold=3
print(np.where(z>3))


# In[21]:


df_new=df[(z<3).all(axis=1)]
df_new


# In[22]:


y=df['Happiness Score']
y


# In[23]:


x=df.drop(['Happiness Score'],axis=1)
x


# In[24]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=42)


# In[25]:


lm=LinearRegression()
lm.fit(x_train,y_train)


# In[26]:


lm.score(x_train,y_train)


# In[27]:


#testing => Predict method

pred=lm.predict(x_test)
print("predicted Happiness score :", pred)
print("actual Happiness score :", y_test)


# In[28]:


print('errors:')
print('Mean absolute error:',mean_absolute_error(y_test,pred))
print('Mean squared error:',mean_squared_error(y_test,pred))
print('Root Mean absolute error:',np.sqrt(mean_squared_error(y_test,pred)))


# In[29]:


#checking for performance of model

from sklearn.metrics import r2_score
print(r2_score(y_test,pred))


# In[30]:


a=np.array([4,78,0.046,1.4,1.2,0.72,0.38,0.67,0.18,1.37])


# In[33]:


a.shape
a=a.reshape(1,-1)
a.shape


# In[34]:


lm.predict(a)


# In[35]:


print("Happiness score for given x values in a : ",lm.predict(a))


# In[ ]:




