
# coding: utf-8

# In[ ]:


#############################################################################################


# In[236]:


import os 
import nltk 
import re
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report
from sklearn import cross_validation,linear_model
from collections import Counter
import warnings # Stop deprecation warnings from being printed
warnings.filterwarnings("ignore", category=DeprecationWarning)
from nltk.corpus import stopwords 
os.chdir('E:/VINOD KUMAR/OMEGA_PROJECT_NLP')
from sklearn.svm import LinearSVC


# In[199]:


#############################################################################################


# In[200]:


os.getcwd()


# In[201]:


os.listdir('E:\\VINOD KUMAR\\OMEGA_PROJECT_NLP')


# In[202]:


os.listdir('E:/VINOD KUMAR/OMEGA_PROJECT_NLP/Train Data')


# In[203]:


# read the files if exists in working directory
if (os.path.exists("Train_data_IPM Jan,2017-May,2018.xlsx"))==True & (os.path.exists("IPM June1-17, 2018.xlsx"))==True:
    print("Train,Test data exists reading as dataframe")
    #print("Files in directory are:",os.listdir(cwd))
    #reading training data into a dataframe using pandas
    actual_data=pd.read_excel("Train_data_IPM Jan,2017-May,2018.xlsx")
    test_data=pd.read_excel("IPM June1-17, 2018.xlsx")
    print(actual_data.shape,test_data.shape)
else: 
    print("file not exists")
    print("Files in directory are:",os.listdir(cwd))


# In[204]:


##########################################################################################


# In[205]:


print(actual_data.info())
print(test_data.info())


# In[206]:


#################################################################################################


# In[207]:


s=actual_data.Categories1.value_counts()
ax=s.plot.bar(figsize=(10,5))
for i, v in s.reset_index().iterrows():
    ax.text(i, v.Categories1, v.Categories1, color='red')    


# In[208]:


#ax = actual_data.groupby("Sub_categories1").size().plot(kind="bar",figsize=(20,10))
#print(actual_data.Sub_categories1.value_counts(),ax)
s=actual_data.Sub_categories1.value_counts()
ax=s.plot.bar(width=.8,figsize=(20,10))
for i, v in s.reset_index().iterrows():
    ax.text(i, v.Sub_categories1, v.Sub_categories1, color='red')


# In[209]:


actual_data_ax = actual_data.groupby(["Categories1","Sub_categories1","Previous_Appointment"]).size().plot(kind="bar", figsize=(15, 5))


# In[210]:


#Each word frequency count in Whole Corpus
freq_count_data = pd.Series(' '.join(actual_data['schedule_note'].astype(str)).split()).value_counts()
print(len(freq_count_data),freq_count_data.sum())


# In[211]:


freq_count_data.describe().plot(kind="bar",figsize=(15, 5))


# In[212]:


freq_count_data.describe().plot(kind="hist",figsize=(15, 5))


# In[213]:


#############################################################################################################


# In[214]:


def process_text(input_string):
    input_string=re.sub(r'[^a-zA-Z0-9\s]', '',input_string)
    sentence = nltk.tokenize.sent_tokenize(input_string)
    out = []
    for sent in sentence:
        wordTokens = nltk.tokenize.word_tokenize(sent)
        lower_tokens = [token.lower() for token in wordTokens]
        stop = stopwords.words('english')
        tokens = [token for token in lower_tokens if token not in stop]
        lmtzr = nltk.stem.WordNetLemmatizer()
        tokens = [lmtzr.lemmatize(token) for token in tokens]
        out.append(" ".join(tokens))
    return out


# In[215]:


actual_data["schedule_note"]=actual_data["schedule_note"].astype("str")
test_data["schedule_note"]=test_data["schedule_note"].astype("str")


# In[216]:


sentence=[]
for i in range (0,len(actual_data)):
        sentence.append(process_text(actual_data["schedule_note"][i]))


# In[217]:


sentence_ts=[]
for i in range (0,len(test_data)):
        sentence_ts.append(process_text(test_data["schedule_note"][i]))


# In[218]:


actual_data["sentence"]=pd.DataFrame(sentence)
test_data["sentence"]=pd.DataFrame(sentence_ts)


# In[219]:


#Each word frequency count in Whole Corpus
freq_count_data = pd.Series(' '.join(actual_data['sentence'].astype(str)).split()).value_counts()
print(len(freq_count_data),freq_count_data.sum())


# In[220]:


actual_data["sentence"]=actual_data["sentence"].astype('str')
actual_data["Categories1"]=actual_data["Categories1"].astype("category")
actual_data["Sub_categories1"]=actual_data["Sub_categories1"].astype("category")
test_data["sentence"]=test_data["sentence"].astype("str")
test_data["Categories1"]=test_data["Categories1"].astype("str")
test_data["Sub_categories1"]=test_data["Sub_categories1"].astype("str")


# In[221]:


#########################################################################################################################


# In[222]:


X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(actual_data["sentence"], actual_data["Categories1"], test_size=0.2,random_state=42)


# In[223]:


X_test_1=test_data["sentence"]
y_test_1=test_data["Categories1"]


# In[224]:


tvec = TfidfVectorizer(analyzer='word', ngram_range=(1,3), max_features=200000)
x_train_tfidf = tvec.fit_transform(X_train_1)
x_validation_tfidf = tvec.transform(X_valid_1)
x_test_tfidf=tvec.transform(X_test_1)


# In[225]:


clf = LinearSVC(C=1.0,penalty='l2', max_iter=10000,multi_class='ovr',dual=True)
clf.fit(x_train_tfidf,y_train_1)
score1 = clf.score(x_train_tfidf,y_train_1)
score2 = clf.score(x_validation_tfidf, y_valid_1)
score3 = clf.score(x_test_tfidf,y_test_1)
print(clf.score(x_train_tfidf,y_train_1),clf.score(x_validation_tfidf, y_valid_1),clf.score(x_test_tfidf,y_test_1))


# In[226]:


ch2 = SelectKBest(chi2, k=120000)
x_train_chi2_selected = ch2.fit_transform(x_train_tfidf,y_train_1)
x_validation_chi2_selected = ch2.transform(x_validation_tfidf)
x_test_chi2_selected=ch2.transform(x_test_tfidf)
clf2 = LinearSVC(C=1.0,penalty='l2', max_iter=10000,multi_class='ovr',dual=True)
clf2.fit(x_train_chi2_selected, y_train_1)
score1_2 = clf2.score(x_train_chi2_selected,y_train_1)
score2_2 = clf2.score(x_validation_chi2_selected, y_valid_1)
score3_2 = clf2.score(x_test_chi2_selected,y_test_1)
print(score1_2,score2_2,score3_2)


# In[ ]:


##########################################################################################################################


# In[227]:


pd.DataFrame(confusion_matrix(y_train_1,clf.predict(x_train_tfidf)),["APPOINTMENTS","ASK_A_DOCTOR ","LAB","MISCELLANEOUS","PRESCRIPTION"],["APPOINTMENTS","ASK_A_DOCTOR ","LAB","MISCELLANEOUS","PRESCRIPTION"])


# In[228]:


print(classification_report(y_train_1,clf.predict(x_train_tfidf)))


# In[239]:


print(classification_report(y_train_1,clf.predict(x_train_tfidf)))


# In[229]:


pd.DataFrame(confusion_matrix(y_train_1,clf2.predict(x_train_chi2_selected)),["APPOINTMENTS","ASK_A_DOCTOR ","LAB","MISCELLANEOUS","PRESCRIPTION"],["APPOINTMENTS","ASK_A_DOCTOR ","LAB","MISCELLANEOUS","PRESCRIPTION"])


# In[186]:


print(classification_report(y_train_1,clf2.predict(x_train_chi2_selected)))


# In[ ]:


#######################################################################################################################


# In[230]:


X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(actual_data["sentence"], actual_data["Sub_categories1"], test_size=0.2,random_state=42)


# In[231]:


X_test_1=test_data["sentence"]
y_test_1=test_data["Sub_categories1"]


# In[232]:


tvec = TfidfVectorizer(analyzer='word', ngram_range=(1,3), max_features=200000)
x_train_tfidf = tvec.fit_transform(X_train_1)
x_validation_tfidf = tvec.transform(X_valid_1)
x_test_tfidf=tvec.transform(X_test_1)


# In[238]:


clf = LinearSVC(C=1.0,penalty='l2', max_iter=10000,multi_class='ovr',dual=True)
#clf = linear_model.LogisticRegression(n_jobs=-1,C=4.0,multi_class='ovr',max_iter=3000, penalty='l2')
clf.fit(x_train_tfidf,y_train_1)
score1 = clf.score(x_train_tfidf,y_train_1)
score2 = clf.score(x_validation_tfidf, y_valid_1)
score3 = clf.score(x_test_tfidf,y_test_1)
print(clf.score(x_train_tfidf,y_train_1),clf.score(x_validation_tfidf, y_valid_1),clf.score(x_test_tfidf,y_test_1))


# In[234]:


print(classification_report(y_train_1,clf.predict(x_train_tfidf)))
pd.DataFrame(confusion_matrix(y_train_1,clf.predict(x_train_tfidf)),["CANCELLATION","CHANGE OF HOSPITAL","CHANGE OF PHARMACY ","CHANGE OF PROVIDER","FOLLOW UP ON PREVIOUS REQUEST","LAB RESULTS","MEDICATION RELATED","NEW APPOINTMENT","OTHERS","PRIOR AUTHORIZATION","PROVIDER","QUERIES FROM PHARMACY","QUERY ON CURRENT APPOINTMENT","REFILL","RESCHEDULING","RUNNING LATE TO APPOINTMENT","SHARING OF HEALTH RECORDS","SHARING OF LAB RECORDS","SYMPTOMS"],["CANCELLATION","CHANGE OF HOSPITAL","CHANGE OF PHARMACY ","CHANGE OF PROVIDER","FOLLOW UP ON PREVIOUS REQUEST","LAB RESULTS","MEDICATION RELATED","NEW APPOINTMENT","OTHERS","PRIOR AUTHORIZATION","PROVIDER","QUERIES FROM PHARMACY","QUERY ON CURRENT APPOINTMENT","REFILL","RESCHEDULING","RUNNING LATE TO APPOINTMENT","SHARING OF HEALTH RECORDS","SHARING OF LAB RECORDS","SYMPTOMS"])

