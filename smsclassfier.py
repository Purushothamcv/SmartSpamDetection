import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv("C:\\Users\\purus\\OneDrive\\New folder\\Desktop\\ml datasets\\spam.csv",encoding="latin1")
# print(df.info())
# data cleaning
# drop last 3 columns
# df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
# print(df.sample(5))
# renaming columns
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")

df.rename(columns={'Category':'target','Message':'text'},inplace=True)
encoder=LabelEncoder()
df['target']=encoder.fit_transform(df['target'])#This converts the categorical labels into numeric values
df.isnull().sum()
df.duplicated().sum()#403 duplicates
df=df.drop_duplicates(keep='first')
df.duplicated().sum() #0 duplicates
# EDA Exploitary Data Analysis 
plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct="%0.2f")#autopct="%0.2f" → Displays the percentage values with two decimal places.
plt.show()
# nltk.download('punkt', download_dir='C:/nltk_data')
df['num_characters']=df['text'].apply(len)
df['num_word'] = df['text'].apply(lambda x: len(x.split()))
#this breaks down the the sentence into separate words
# sent_tokenize (This converts the text into sentence format)
nltk.download('punkt',download_dir='C:/nltk_data')
nltk.data.path.append('C:/nltk_data')
print(nltk.data.path)
df['sentence']=df['text'].apply(lambda x:nltk.sent_tokenize(str(x)) if pd.notnull(x) else [])
# print(df[['text','sentence']].head())
print(df.columns)
tfidf = TfidfVectorizer()
df['sentence'] = df['sentence'].apply(lambda sents: " ".join(sents))
X_text = tfidf.fit_transform(df['sentence']) 
sc=StandardScaler()
x_num=sc.fit_transform(df[['num_characters','num_word']])
import scipy.sparse
# this is used to combine two sparse matrices
x = scipy.sparse.hstack((X_text, x_num))
y=df['target'].values
from sklearn.model_selection import train_test_split    
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
# model=LogisticRegression()
model1=RandomForestClassifier()  
# model.fit(X_train,y_train)
model1.fit(X_train,y_train)
y_pred1=model1.predict(X_test)
print(accuracy_score(y_test,y_pred1))

y_pred=model1.predict(X_test)
print(accuracy_score(y_test,y_pred)) 
# y_pre=model.predict(X_test[0])
# print(X_test[0])
# print(y_pre)
# print(encoder.inverse_transform(y_pre))   
X_train_text = tfidf.fit_transform(df['text'])
model1.fit(X_train_text, y)
email = "Congratulations! You've won a free ticket. Reply now!"
X_new_text = tfidf.transform([email])
pred=model1.predict(X_new_text)
print(pred)
print(encoder.inverse_transform(pred))
import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(model1,open('model.pkl','wb'))   
vectorizer=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

