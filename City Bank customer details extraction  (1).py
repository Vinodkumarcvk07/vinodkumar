
# coding: utf-8

# # City Bank customer details extraction based on companies categories A,B,C

# In[1]:


import os
import pandas as pd
import numpy as np
from collections import Counter
os.chdir('E:\VINOD KUMAR\City_bank_table_creation')


# In[2]:


data = pd.read_csv('CAT_B.csv')
file1 = pd.read_excel('New - MDB - HANDOUT - MONSTER1 EDB-Dec232014.xlsx')
file2 = pd.read_excel('New - MDB - HANDOUT - Naukri0 EDB.xlsx')


# In[8]:


#print("\n",data.columns,"\n \n ",file1.columns,"\n \n",file2.columns,"\n \n")

#print("\n",data.info(),"\n \n ","\n \n",file1.info(),"\n \n","\n \n",file2.info())


# In[5]:


data.columns=["company"]
#Converted all company names to upper case 
data["company"]=data.company.str.upper()
file1["COMPANY"]=file1.COMPANY.str.upper()
file2["COMPANY"]=file2.COMPANY.str.upper()


# In[6]:


comp1=[]
comp2=[]
for i in range (0,len(data)): 
    comp1.append(file1[file1['COMPANY'].str.contains(" ".join(data["company"][i].split(" ")[:2]))==True])
    comp2.append(file2[file2['COMPANY'].str.contains(" ".join(data["company"][i].split(" ")[:2]))==True])


# In[7]:


comp3=[]
comp4=[]
for i in range (0,480):
    comp3.append(comp1[i])
    comp4.append(comp2[i])
    
company1=pd.concat(comp3)
company2=pd.concat(comp4)

company1.to_csv("CATB_New - MDB - HANDOUT - MONSTER1 EDB_13_June_2018.csv")
company2.to_csv("CATB_New - MDB - HANDOUT - Naukri0 EDB_13_June_2018.csv")

