#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd
from pandas import Series,DataFrame


# In[3]:


titanic_df=pd.read_csv('E:/data science project/titenic data set/titanic_train.csv')


# In[4]:


titanic_df.head()


# In[5]:


titanic_df.info()


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


sns.catplot(x='sex',data=titanic_df)


# In[8]:


sns.catplot(x='sex',y='passenger_id',data=titanic_df,hue='pclass');


# In[9]:


sns.catplot(x='pclass',y='passenger_id',data=titanic_df,hue='sex');


# In[10]:


def male_female_child(passenger):
    age,sex=passenger
    if age<16:
        return 'child'
    else:
        return sex


# In[11]:


titanic_df['person']=titanic_df[['age','sex']].apply(male_female_child,axis=1)


# In[12]:


titanic_df[0:50]


# In[13]:


sns.catplot(x='pclass',y='passenger_id',data=titanic_df,hue='person');


# In[14]:


titanic_df['age'].hist(bins=70)


# In[15]:


titanic_df['age'].mean()


# In[16]:


titanic_df['person'].value_counts()


# In[17]:


fig=sns.FacetGrid(titanic_df,hue='sex',aspect=4)


# In[18]:


fig.map(sns.kdeplot,'age',shade=True)


# In[19]:


oldest=titanic_df['age'].max()


# In[20]:


fig.set(xlim=(0,oldest))
fig.add_legend()


# In[21]:


fig=sns.FacetGrid(titanic_df,hue='sex',aspect=4)
fig.map(sns.kdeplot,'age',shade=True)
oldest=titanic_df['age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()


# In[22]:


fig=sns.FacetGrid(titanic_df,hue='person',aspect=4)
fig.map(sns.kdeplot,'age',shade=True)
oldest=titanic_df['age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()


# In[23]:


fig=sns.FacetGrid(titanic_df,hue='pclass',aspect=4)
fig.map(sns.kdeplot,'age',shade=True)
oldest=titanic_df['age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()


# In[24]:


titanic_df.head()


# In[25]:


deck=titanic_df['cabin'].dropna()


# In[26]:


deck.head()


# In[27]:


levels=[]
for level in deck:
    levels.append(level[0])
cabin_df=DataFrame(levels)
cabin_df.columns=['cabin']
sns.catplot(x='cabin',data=cabin_df,palette='winter_d')


# In[28]:


cabin_df=cabin_df[cabin_df.cabin!='T']
sns.catplot(x='cabin',data=cabin_df,palette='summer')


# In[29]:


sns.catplot(x='embarked',y='passenger_id',data=titanic_df,hue='pclass',x_order=['C','Q','S']);


# In[ ]:


sns.catplot(x='embarked',data=titanic_df,hue='pclass',x_order=['C','Q','S']);


# In[30]:


titanic_df['alone']=titanic_df.sibsp + titanic_df.parch                


# In[32]:


titanic_df['alone']


# In[35]:


titanic_df['alone'].loc[titanic_df['alone']>0]='with family'
titanic_df['alone'].loc[titanic_df['alone']==0]='alone'


# In[37]:


sns.catplot(x='alone',y='passenger_id',data=titanic_df,palette='Blues')


# In[38]:


titanic_df['survivor']=titanic_df.survived.map({0:'no',1:'yes'})


# In[40]:


sns.catplot(x='survivor',y='passenger_id',data=titanic_df)


# In[42]:


sns.catplot(x='pclass',y='survived',hue='person',data=titanic_df)


# In[43]:


sns.lmplot('age','survived',data=titanic_df)


# In[44]:


sns.lmplot('age','survived',data=titanic_df,hue='pclass')


# In[46]:


generatiton=[10,20,40,60,80]
sns.lmplot('age','survived',hue='pclass',data=titanic_df,x_bins=generatiton)


# In[47]:


sns.lmplot('age','survived',hue='sex',data=titanic_df,x_bins=generatiton)


# In[ ]:




