
# coding: utf-8

# In[4]:


from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import os,nltk,re,string,numpy as np,pandas as pd,pickle
from sklearn.externals import joblib
os.chdir('E:\VINOD KUMAR\OMEGA_PROJECT_NLP\IPM_AUTOMATION')


# In[5]:


# Load categories model 
with open("IPM_pickle_model_categories.pkl", 'rb') as file:  
    IPM_pickle_model_categories = pickle.load(file)


# In[6]:


# Load sub categories model 
with open("IPM_pickle_model_sub_categories.pkl", 'rb') as file:  
    IPM_pickle_model_sub_categories = pickle.load(file)


# In[7]:


# Load previous appointments model 
with open("IPM_pickle_model_Previous_Appointment.pkl", 'rb') as file:  
    IPM_pickle_model_Previous_Appointment = pickle.load(file)


# In[8]:


# Load vectorizer for categories 
with open("tfidf_categories.pkl", 'rb') as file:
    tfidf_categories = joblib.load(file)


# In[9]:


# Load vectorizer for categories 
with open("tfidf_sub_categories.pkl", 'rb') as file:
    tfidf_sub_categories = joblib.load(file)


# In[10]:


# Load vectorizer for categories 
with open("tfidf_previous_appointment.pkl", 'rb') as file:
    tfidf_previous_appointment = joblib.load(file)


# In[11]:


test_data=pd.read_excel("IPM to be processed 27 to 30 June 2018.xlsx")


# In[15]:


test_data["schedule_note"]=test_data["schedule_note"].astype("str")


# In[16]:


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


# In[18]:


sentence_ts=[]
for i in range (0,len(test_data)):
        sentence_ts.append(process_text(test_data["schedule_note"][i]))


# In[20]:


test_data["sentence"]=(pd.DataFrame(sentence_ts)).astype("str")


# In[21]:


X_test_1=test_data["sentence"]


# In[22]:


x_test_tfidf1=tfidf_categories.transform(X_test_1)
x_test_tfidf2=tfidf_sub_categories.transform(X_test_1)
x_test_tfidf3=tfidf_previous_appointment.transform(X_test_1)


# In[23]:


test_data["Categories"]=IPM_pickle_model_categories.predict(x_test_tfidf1)
test_data["Sub_Categories"]=IPM_pickle_model_sub_categories.predict(x_test_tfidf2)
test_data["Previous_appointments"]=IPM_pickle_model_Previous_Appointment.predict(x_test_tfidf3)


# In[24]:


test_data.to_excel("IPM after processed.xlsx",index=False)


# In[25]:


print("IPM PREDICTIONS SUCESSFULL")

