
# coding: utf-8

# # Kris_tab_data 

# In[159]:


#importing required libraries
import pandas as pd
import time
from datetime import datetime, date,timedelta
import os
#from pandas.tseries.offsets import *
import matplotlib.pyplot as plt


# In[160]:


#Changing working directory
os.chdir('E:/VINOD KUMAR/Project_neospark_kristab')


# In[161]:


#reading excel data from local system to jupiter note book using pandas  
empl=pd.read_excel('Pulse.xlsx')


# In[162]:


#dimentions of data frame 
print(empl.shape)


# In[163]:


#information of data frame
empl.info()


# In[164]:


#view first 5 rows of data
empl.head()


# In[165]:


#date time column
empl["Created_Date"].head()


# # Extract freatures from date column 

# In[166]:


#extracting hours
empl["hours"]=empl.Created_Date.dt.hour


# In[167]:


#extracting weekday_names
empl["weekday_name"]=empl.Created_Date.dt.weekday_name


# In[168]:


#extracting months
empl["month"]=empl.Created_Date.dt.month


# In[169]:


#extracting minutes 
empl["minutes"]=empl.Created_Date.dt.minute


# # Grouping hours in to shift intervals

# In[170]:


def group_shifts(val):
    if 0 <= val <= 12:
        letter = 'Morning_shift'
    elif 12 <= val <= 16:
        letter = 'Afternoon_shift'
    else:
        letter = 'Evening_shift'
    return letter.strip(" ")


# In[171]:


group_shift=[]
for i in range (0,len(empl)):
    group_shift.append(group_shifts(empl.hours[i]))


# In[172]:


empl["group_shift"]=pd.DataFrame(group_shift)


# In[173]:


empl = pd.concat([empl, pd.get_dummies(empl['group_shift'])], axis=1)


# In[174]:


#groping teh data we required 


# In[175]:


group_by_table=(empl.groupby(["SL_Code","Customer_Code","Customer_Name","group_shift"]).group_shift.count()).unstack("group_shift")


# In[176]:


group_by_table.head()


# In[177]:


group_by_table.columns


# In[178]:


group_by_table.info()


# In[199]:


group_by_table[["Evening_shift","Morning_shift","Afternoon_shift"]].count().plot(kind="pie")
plt.xlabel('Frequency count of successfull visits')
plt.ylabel('All intervals of time')
plt.savefig('all_shifts_distribution.png')


# In[198]:


plt.rcParams["figure.figsize"] = (10,10)
print(group_by_table.groupby(["SL_Code"]).Afternoon_shift.count().plot(kind="pie"))
plt.xlabel('Contibution of successfull visits at each area')
plt.savefig('Afternoon_shift_area.png')


# In[197]:


plt.rcParams["figure.figsize"] = (10,10)
print(group_by_table.groupby(["SL_Code"]).Morning_shift.count().plot(kind="pie"))
plt.xlabel('Contibution of successfull visits at each area')
plt.savefig('Morning_shift_area.png')


# In[196]:


print(group_by_table.groupby(["SL_Code"]).Evening_shift.count().plot(kind="pie"))
plt.xlabel('Contibution of successfull visits at each area')
plt.savefig('Evening_shift_area.png')


# In[195]:


group_by_table[["Evening_shift","Morning_shift","Afternoon_shift"]].count().plot(kind="bar")
plt.ylabel('Total vists in each interval of time')
plt.savefig('group_shift.png')


# In[185]:


table1=pd.DataFrame(group_by_table)


# In[186]:


table1.columns


# In[187]:


table1.to_csv("table1.csv",index=True,header=True)


# In[188]:


table2=pd.read_csv("table1.csv")


# In[189]:


table2.head()


# In[191]:


#group_by_table[["Evening_shift","Morning_shift","Afternoon_shift"]]
#table2.groupby(["Customer_Code"]).Afternoon_shift.sum()
((empl.groupby(["SL_Code","Customer_Code","Customer_Name","group_shift"]).group_shift.count()).unstack("group_shift"))

