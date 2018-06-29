
# coding: utf-8

# In[1]:


#############################################################################################


# In[66]:


import os 
import nltk 
import re
import string
import numpy as np
import pandas as pd
import rtfConverter as rtf
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
# spacy
import spacy
import en_core_web_sm 
nlp2= en_core_web_sm.load()


# In[3]:


#############################################################################################


# In[4]:


os.getcwd()


# In[5]:


os.listdir('E:\\VINOD KUMAR\\OMEGA_PROJECT_NLP')


# In[6]:


os.listdir('E:/VINOD KUMAR/OMEGA_PROJECT_NLP/Train Data')


# In[7]:


# read the files if exists in working directory
if (os.path.exists("Train_data_Guliford Jan,2017 -May,2018.xlsx"))==True & (os.path.exists('Guliford June 1-25, 2018.xlsx'))==True:
    print("Train,Test data exists reading as dataframe")
    #print("Files in directory are:",os.listdir(cwd))
    #reading training data into a dataframe using pandas
    actual_data=pd.read_excel("Train_data_Guliford Jan,2017 -May,2018.xlsx")
    test_data=pd.read_excel('Guliford June 1-25, 2018.xlsx')
    print(actual_data.shape,test_data.shape)
else: 
    print("file not exists")
    print("Files in directory are:",os.listdir(cwd))


# In[8]:


##########################################################################################


# In[9]:


print(actual_data.info())
print(test_data.info())


# In[10]:


#################################################################################################


# In[11]:


s=actual_data.Categories1.value_counts()
ax=s.plot.bar(figsize=(10,5))
for i, v in s.reset_index().iterrows():
    ax.text(i, v.Categories1, v.Categories1, color='red')    


# In[12]:


#ax = actual_data.groupby("Sub_categories1").size().plot(kind="bar",figsize=(20,10))
#print(actual_data.Sub_categories1.value_counts(),ax)
s=actual_data.Sub_categories1.value_counts()
ax=s.plot.bar(width=.8,figsize=(20,10))
for i, v in s.reset_index().iterrows():
    ax.text(i, v.Sub_categories1, v.Sub_categories1, color='red')


# In[18]:


s=actual_data.Previous_Appointment.value_counts()
ax=s.plot.bar(width=.3,figsize=(5,5))
for i, v in s.reset_index().iterrows():
    ax.text(i, v.Previous_Appointment, v.Previous_Appointment, color='red')


# In[19]:


actual_data_ax = actual_data.groupby(["Categories1","Sub_categories1","Previous_Appointment"]).size().plot(kind="bar", figsize=(15, 5))


# In[20]:


#Each word frequency count in Whole Corpus
freq_count_data = pd.Series(' '.join(actual_data['DATA'].astype(str)).split()).value_counts()
print(len(freq_count_data),freq_count_data.sum())


# In[21]:


freq_count_data.describe(),freq_count_data.describe().plot(kind="bar",figsize=(15, 5))


# In[22]:


freq_count_data.describe(),freq_count_data.describe().plot(kind="hist",figsize=(15, 5))


# In[ ]:


#############################################################################################################


# In[23]:


actual_data["DATA"]=actual_data["DATA"].astype("str")
actual_data["SUMMARY"]=actual_data["SUMMARY"].astype("str")
test_data["DATA"]=test_data["DATA"].astype("str")
test_data["SUMMARY"]=test_data["SUMMARY"].astype("str")


# In[24]:


#Combining SUMMARY column with Spacy_sentences
actual_data['combined_sent'] = actual_data['SUMMARY'].map(str) +" " +actual_data['DATA']
test_data['combined_sent'] = test_data['SUMMARY'].map(str) +" " +test_data['DATA']


# In[126]:


def process_text(input_string):
    input_string = rtf.striprtf(input_string)
    string = re.sub(r"Phone Note", " ", input_string)
    string = re.sub(r"Summary of Call", " ", string)
    string = re.sub(r"New Medications", " ", string)
    string = re.sub(r"New Allergies", " ", string)
    string = re.sub(r"Call from Patient", " ", string)
    string = re.sub(r"Clinical Followup Details", " ", string)
    string = re.sub(r"Caller"," ", string)
    string = re.sub(r"Initial call taken by", " ", string)
    string = re.sub(r"Followup Details", " ", string)
    string = re.sub(r"Follow-up by", " ", string)
    string = re.sub(r"Call back at", " ", string)
    string = re.sub(r"Call from Other Clinic", " ", string)
    string = re.sub(r"Home Phone", " ", string)
    string = re.sub(r"Call from Pharmacy", " ", string)
    string = re.sub(r"Follow up for Phone Call ", " ", string)
    string = re.sub(r"Phone Call Completed ", " ", string)
    string = re.sub(r"Follow up by", " ", string)
      
    #d=(" ".join(map(str,(nlp2(string).ents))).split(" "))
    #word_tokens=nltk.tokenize.word_tokenize(string)
    #string=(" ".join([w for w in word_tokens if not w in d]))
    
    string = string.lower()
    string = re.sub(r"appt "," appointment ",string)
    string = re.sub(r"schd "," scheduled ",string)
    string = re.sub(r"rs "," rescheduled ",string)
    string = re.sub(r"med "," medication ",string)
    string = re.sub(r"pa "," prior authorization ",string)
    string = re.sub(r"auth "," authorization ",string)
    string = re.sub(r"preauth "," authorization ",string)
    string = re.sub(r"sx "," scheduled ",string)
    string = re.sub(r"cx "," cancel ",string)
    string = re.sub(r"re "," cancel ",string)
     
    input_string=re.sub(r'[^a-zA-Z\s]',' ',string)
    sentence = nltk.tokenize.sent_tokenize(input_string)
    out = []
    for sent in sentence:
        wordTokens = nltk.tokenize.word_tokenize(sent)
        lower_tokens = [token.lower() for token in wordTokens]
        stop = stopwords.words('english')+["nan","none","pm","am","2017","2016","january",
                                           "february","march","april","may","june","july","august","september",
                                          "october","november","december"]
        tokens = [token for token in lower_tokens if token not in stop]
        lmtzr = nltk.stem.WordNetLemmatizer()
        tokens = [lmtzr.lemmatize(token) for token in tokens]
        out.append(" ".join(tokens))
    return out


# In[127]:


sentence=[]
for i in range (0,len(actual_data)):
        sentence.append(process_text(actual_data["combined_sent"][i]))


# In[128]:


sentence_ts=[]
for i in range (0,len(test_data)):
        sentence_ts.append(process_text(test_data["combined_sent"][i]))


# In[129]:


actual_data["sentence"]=pd.DataFrame(sentence)
test_data["sentence"]=pd.DataFrame(sentence_ts)


# In[130]:


a=44555
actual_data["sentence"][a],"\n\n",actual_data["combined_sent"][a]


# In[132]:


#Each word frequency count in Whole Corpus
freq_count_data = pd.Series(' '.join(actual_data['sentence'].astype(str)).split()).value_counts()
print(len(freq_count_data),freq_count_data.sum())


# In[29]:


#Each word frequency count in Whole Corpus
freq_count_data = pd.Series(' '.join(actual_data['sentence'].astype(str)).split()).value_counts()
print(len(freq_count_data),freq_count_data.sum())


# In[133]:


freq_count_data.describe()


# In[134]:


import seaborn as sns
p=sns.distplot(freq_count_data,bins=1)


# In[186]:


#freq = pd.Series(' '.join(actual_data["sentence"]).split()).value_counts()[-2000:]
freq2 = pd.Series(' '.join(actual_data["sentence"]).split()).value_counts()[:10]
freq2


# In[187]:


freq2 = list(freq2.index)
actual_data['sentence'] = actual_data['sentence'].apply(lambda x: " ".join(x for x in x.split() if x not in freq2))


# In[176]:


freq = list(freq.index)
actual_data['sentence'] = actual_data['sentence'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
actual_data['sentence'].head()


# In[194]:


freq2 = pd.Series(' '.join(actual_data["sentence"]).split()).value_counts()[:10]
freq2


# In[195]:


actual_data["sentence"]=actual_data["sentence"].astype('str')
actual_data["Categories1"]=actual_data["Categories1"].astype("category")
actual_data["Sub_categories1"]=actual_data["Sub_categories1"].astype("category")
test_data["sentence"]=test_data["sentence"].astype("str")
test_data["Categories1"]=test_data["Categories1"].astype("str")
test_data["Sub_categories1"]=test_data["Sub_categories1"].astype("str")


# In[239]:


#actual_data.to_csv("train_data_June.csv",index=False)
#test_data.to_csv("test_data_June.csv",index=False)


# In[196]:


#########################################################################################################################


# In[197]:


X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(actual_data["sentence"], actual_data["Categories1"], test_size=0.2)


# In[198]:


X_test_1=test_data["sentence"]
y_test_1=test_data["Categories1"]


# In[199]:


tvec = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=200000)
x_train_tfidf = tvec.fit_transform(X_train_1)
x_validation_tfidf = tvec.transform(X_valid_1)
x_test_tfidf=tvec.transform(X_test_1)


# In[200]:


clf = LinearSVC(C=1.0,penalty='l2', max_iter=10000,multi_class='ovr',dual=True)
clf.fit(x_train_tfidf,y_train_1)
score1 = clf.score(x_train_tfidf,y_train_1)
score2 = clf.score(x_validation_tfidf, y_valid_1)
score3 = clf.score(x_test_tfidf,y_test_1)
print(clf.score(x_train_tfidf,y_train_1),clf.score(x_validation_tfidf, y_valid_1),clf.score(x_test_tfidf,y_test_1))


# In[184]:


clf = linear_model.LogisticRegression(C=10.0,multi_class='ovr',max_iter=3000, penalty='l2')
clf.fit(x_train_tfidf,y_train_1)
score1 = clf.score(x_train_tfidf,y_train_1)
score2 = clf.score(x_validation_tfidf, y_valid_1)
score3 = clf.score(x_test_tfidf,y_test_1)
print(clf.score(x_train_tfidf,y_train_1),clf.score(x_validation_tfidf, y_valid_1),clf.score(x_test_tfidf,y_test_1))


# In[185]:


ch2 = SelectKBest(chi2, k=100000)
x_train_chi2_selected = ch2.fit_transform(x_train_tfidf,y_train_1)
x_validation_chi2_selected = ch2.transform(x_validation_tfidf)
x_test_chi2_selected=ch2.transform(x_test_tfidf)
clf2 = LinearSVC(C=1.0,penalty='l2', max_iter=10000,multi_class='ovr',dual=True)
clf2.fit(x_train_chi2_selected, y_train_1)
score1_2 = clf2.score(x_train_chi2_selected,y_train_1)
score2_2 = clf2.score(x_validation_chi2_selected, y_valid_1)
score3_2 = clf2.score(x_test_chi2_selected,y_test_1)
print(score1_2,score2_2,score3_2)


# In[146]:


##########################################################################################################################


# In[201]:


print(classification_report(y_train_1,clf.predict(x_train_tfidf)))
pd.DataFrame(confusion_matrix(y_train_1,clf.predict(x_train_tfidf)),["APPOINTMENTS","ASK_A_DOCTOR ","LAB","MISCELLANEOUS","PRESCRIPTION"],["APPOINTMENTS","ASK_A_DOCTOR ","LAB","MISCELLANEOUS","PRESCRIPTION"])


# In[202]:


print(classification_report(y_valid_1,clf.predict(x_validation_tfidf)))
pd.DataFrame(confusion_matrix(y_valid_1,clf.predict(x_validation_tfidf)),["APPOINTMENTS","ASK_A_DOCTOR ","LAB","MISCELLANEOUS","PRESCRIPTION"],["APPOINTMENTS","ASK_A_DOCTOR ","LAB","MISCELLANEOUS","PRESCRIPTION"])


# In[203]:


print(classification_report(y_train_1,clf.predict(x_train_tfidf)))


# In[204]:


pd.DataFrame(confusion_matrix(y_train_1,clf2.predict(x_train_chi2_selected)),["APPOINTMENTS","ASK_A_DOCTOR ","LAB","MISCELLANEOUS","PRESCRIPTION"],["APPOINTMENTS","ASK_A_DOCTOR ","LAB","MISCELLANEOUS","PRESCRIPTION"])


# In[205]:


print(classification_report(y_train_1,clf2.predict(x_train_chi2_selected)))


# In[165]:


#######################################################################################################################


# In[166]:


X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(actual_data["sentence"], actual_data["Sub_categories1"], test_size=0.2,random_state=42)


# In[167]:


X_test_1=test_data["sentence"]
y_test_1=test_data["Sub_categories1"]


# In[168]:


tvec = TfidfVectorizer(analyzer='word', ngram_range=(1,3), max_features=200000)
x_train_tfidf = tvec.fit_transform(X_train_1)
x_validation_tfidf = tvec.transform(X_valid_1)
x_test_tfidf=tvec.transform(X_test_1)


# In[169]:


clf = LinearSVC(C=1.0,penalty='l2', max_iter=10000,multi_class='ovr',dual=True)
#clf = linear_model.LogisticRegression(n_jobs=-1,C=4.0,multi_class='ovr',max_iter=3000, penalty='l2')
clf.fit(x_train_tfidf,y_train_1)
score1 = clf.score(x_train_tfidf,y_train_1)
score2 = clf.score(x_validation_tfidf, y_valid_1)
score3 = clf.score(x_test_tfidf,y_test_1)
print(clf.score(x_train_tfidf,y_train_1),clf.score(x_validation_tfidf, y_valid_1),clf.score(x_test_tfidf,y_test_1))


# In[174]:


print(clf.score(x_train_tfidf,y_train_1),clf.score(x_validation_tfidf, y_valid_1),clf.score(x_test_tfidf,y_test_1))
print(classification_report(y_train_1,clf.predict(x_train_tfidf)))
pd.DataFrame(confusion_matrix(y_train_1,clf.predict(x_train_tfidf)),["CANCELLATION","CHANGE OF HOSPITAL","CHANGE OF PHARMACY ",
                                                                     "CHANGE OF PROVIDER","FOLLOW UP ON PREVIOUS REQUEST","LAB RESULTS",
                                                                     "MEDICATION RELATED","NEW APPOINTMENT","OTHERS","PRIOR AUTHORIZATION",
                                                                     "PROVIDER","QUERIES FROM INSURANCE FIRM","QUERIES FROM PHARMACY",
                                                                     "QUERY ON CURRENT APPOINTMENT","REFILL","RESCHEDULING",
                                                                     "RUNNING LATE TO APPOINTMENT","SHARING OF HEALTH RECORDS",
                                                                     "SHARING OF LAB RECORDS","SYMPTOMS"],["CANCELLATION","CHANGE OF HOSPITAL","CHANGE OF PHARMACY ",
                                                                     "CHANGE OF PROVIDER","FOLLOW UP ON PREVIOUS REQUEST","LAB RESULTS",
                                                                     "MEDICATION RELATED","NEW APPOINTMENT","OTHERS","PRIOR AUTHORIZATION",
                                                                     "PROVIDER","QUERIES FROM INSURANCE FIRM","QUERIES FROM PHARMACY",
                                                                     "QUERY ON CURRENT APPOINTMENT","REFILL","RESCHEDULING",
                                                                     "RUNNING LATE TO APPOINTMENT","SHARING OF HEALTH RECORDS",
                                                                     "SHARING OF LAB RECORDS","SYMPTOMS"])


# In[ ]:


######################################################################################################


# In[208]:


from gensim.models import Word2Vec


# In[206]:


#to create a list of words in a list
list_of_words= []
for i in range (0,len(actual_data["sentence"])):
    list_of_words.append(nltk.word_tokenize(actual_data["sentence"][i]))


# In[209]:


wordtovec = Word2Vec(list_of_words,min_count =2)


# In[210]:


wc_words = list(wordtovec.wv.vocab)
print(wc_words)


# In[238]:


wc_words


# In[228]:


wordtovec.most_similar('refill')," ", wordtovec.most_similar('treat')," ",wordtovec.most_similar('sch'),"  ",wordtovec.most_similar('others')," ",wordtovec.most_similar('pharmacy')


# In[229]:


wordtovec.most_similar('appointment')," ",wordtovec.most_similar('rx')


# In[230]:


wordtovec.most_similar('sch'),"  ",wordtovec.most_similar('others')," ",wordtovec.most_similar('pharmacy')


# In[235]:


wordtovec

