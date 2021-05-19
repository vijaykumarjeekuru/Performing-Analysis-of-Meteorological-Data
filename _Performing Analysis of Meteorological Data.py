#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as pk


# In[2]:


df =pd.read_csv("C:/Users/vijay kumar jeekuru/Downloads/weatherHistory.csv")
df.head(10)


# In[3]:


df.tail(10)


# In[4]:


df.isnull().sum()


# In[5]:


df.dtypes


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df['Formatted Date'] =pd.to_datetime(df['Formatted Date'], utc=True)
df['Formatted Date']


# In[9]:


df=df.set_index('Formatted Date')
df.head()


# In[10]:


data_coloumn=['Apparent Temperature (C)','Humidity']
df_monthly_mean=df.resample('MS').mean()
df_monthly_mean.head()


# In[11]:


df=df.resample('MS').mean()


# In[12]:


apt=df['Apparent Temperature (C)']


# In[13]:


hmd=df['Humidity']


# In[14]:


import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

plt.figure(figsize=(12,9))
plt.title("Analysis of Humidity and Temperature")
plt.plot(hmd,label="Average Humidity")
plt.plot(apt,label="Average Apparent Temperature")
plt.xlabel("Years")
plt.legend(loc=(1.01,0.8))
plt.show()


# In[15]:


df1 = df[df.index.month==1]
hdt=df1['Humidity']
atm=df1['Apparent Temperature (C)']
plt.title("January Monthwise Stats",fontsize=20)
plt.plot(hdt,label="Average Humidity",marker=".")
plt.plot(atm,label="Average Apparent Temperature",marker='.')
plt.legend(loc=(1.02,0.8))
plt.show()


# In[16]:


df1 = df[df.index.month==2]
hdt=df1['Humidity']
atm=df1['Apparent Temperature (C)']
plt.title("feb Monthwise Stats",fontsize=20)
plt.plot(hdt,label="Average Humidity",marker=".")
plt.plot(atm,label="Average Apparent Temperature",marker='.')
plt.legend(loc=(1.02,0.8))
plt.show()


# In[17]:


df1 = df[df.index.month==3]
hdt=df1['Humidity']
atm=df1['Apparent Temperature (C)']
plt.title("march Monthwise Stats",fontsize=20)
plt.plot(hdt,label="Average Humidity",marker=".")
plt.plot(atm,label="Average Apparent Temperature",marker='.')
plt.legend(loc=(1.02,0.8))
plt.show()


# In[18]:


df1 = df[df.index.month==4]
hdt=df1['Humidity']
atm=df1['Apparent Temperature (C)']
plt.title("april Monthwise Stats",fontsize=17)
plt.plot(hdt,label="Average Humidity",marker=".")
plt.plot(atm,label="Average Apparent Temperature",marker=".")
plt.legend(loc=(1.02,0.8))
plt.show()


# In[19]:


df1 = df[df.index.month==5]
hdt=df1['Humidity']
atm=df1['Apparent Temperature (C)']
plt.title("May Monthwise Stats",fontsize=17)
plt.plot(hdt,label="Average Humidity",marker=".")
plt.plot(atm,label="Average Apparent Temperature",marker='.')
plt.legend(loc=(1.02,0.8))
plt.show()


# In[20]:


df1 = df[df.index.month==6]
hdt=df1['Humidity']
atm=df1['Apparent Temperature (C)']
plt.title("June Monthwise Stats",fontsize=17)
plt.plot(hdt,label="Average Humidity",marker=".")
plt.plot(atm,label="Average Apparent Temperature",marker='.')
plt.legend(loc=(1.02,0.8))
plt.show()


# In[21]:


df1 = df[df.index.month==7]
hdt=df1['Humidity']
atm=df1['Apparent Temperature (C)']
plt.title("July Monthwise Stats",fontsize=17)
plt.plot(hdt,label="Average Humidity",marker=".")
plt.plot(atm,label="Average Apparent Temperature",marker='.')
plt.legend(loc=(1.02,0.8))
plt.show()


# In[22]:


df1 = df[df.index.month==8]
hdt=df1['Humidity']
atm=df1['Apparent Temperature (C)']
plt.title("Aug Monthwise Stats",fontsize=17)
plt.plot(hdt,label="Average Humidity",marker=".")
plt.plot(atm,label="Average Apparent Temperature",marker='.')
plt.legend(loc=(1.02,0.8))
plt.show()


# In[23]:


df1 = df[df.index.month==9]
hdt=df1['Humidity']
atm=df1['Apparent Temperature (C)']
plt.title("sep Monthwise Stats",fontsize=17)
plt.plot(hdt,label="Average Humidity",marker=".")
plt.plot(atm,label="Average Apparent Temperature",marker='.')
plt.legend(loc=(1.02,0.8))
plt.show()


# In[24]:


df1 = df[df.index.month==10]
hdt=df1['Humidity']
atm=df1['Apparent Temperature (C)']
plt.title("oct Monthwise Stats",fontsize=17)
plt.plot(hdt,label="Average Humidity",marker=".")
plt.plot(atm,label="Average Apparent Temperature",marker='.')
plt.legend(loc=(1.02,0.8))
plt.show()


# In[25]:


df1 = df[df.index.month==11]
hdt=df1['Humidity']
atm=df1['Apparent Temperature (C)']
plt.title("November Monthwise Stats",fontsize=17)
plt.plot(hdt,label="Average Humidity",marker=".")
plt.plot(atm,label="Average Apparent Temperature",marker='.')
plt.legend(loc=(1.02,0.8))
plt.show()


# In[26]:


df1 = df[df.index.month==12]
hdt=df1['Humidity']
atm=df1['Apparent Temperature (C)']
plt.title("December Monthwise Stats",fontsize=17)
plt.plot(hdt,label="Average Humidity",marker=".")
plt.plot(atm,label="Average Apparent Temperature",marker='.')
plt.legend(loc=(1.02,0.8))
plt.show()


# In[27]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk


# In[28]:


sns.distplot(df.Humidity, color = 'red')


# In[29]:


sns.relplot(data = df, x = "Apparent Temperature (C)", y = "Humidity", color = 'purple')


# In[30]:


sns.jointplot(data = df, x = "Apparent Temperature (C)", y = "Humidity")


# In[31]:


sns.pairplot(df)


# In[32]:


#Conclusion:¶
#There is No change in average humidity. The year 2009 can see an increase in average apparent temperature, then a fall in 2010, then a slight increase in 2011, then a significant drop in 2015, and then an increase in 2016.


# In[ ]:




