#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd  
import numpy as np
import seaborn as sns
#from sklearn.svm import SVC
#from sklearn.model_selection import KFold
from sklearn import preprocessing
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv('android.csv')


# In[3]:


data


# In[4]:


data.shape


# In[5]:


data = data.sample(frac=1).reset_index(drop=True)


# In[6]:


data.head()


# In[7]:


import seaborn as sns


# In[8]:


sns.countplot(x='malware',data=data)


# In[9]:


target_count = data.malware.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])


# In[10]:


count_class_0, count_class_1 = data.malware.value_counts()


# In[11]:


df_class_0 = data[data['malware'] == 0]
df_class_1 = data[data['malware'] == 1]


# In[12]:


df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)


# In[13]:


df_test_over.shape


# In[14]:


sns.countplot(x='malware',data=df_test_over)


# In[15]:


X=df_test_over.iloc[:,df_test_over.columns !='malware']
Y=df_test_over.iloc[:,df_test_over.columns =="malware"]


# In[16]:


X.head()


# In[17]:


Y.head()


# In[18]:


from sklearn.utils import shuffle


# In[19]:


X, Y=shuffle(X, Y)


# In[20]:


X.head()


# In[21]:


X=X.drop(columns='name')
X.head()


# In[22]:


Y.head()


# In[23]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[24]:


bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)


# In[25]:


featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  
featureScores.nlargest(10,'Score')  


# In[26]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# In[27]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state=0)


# In[28]:


X_train.shape


# In[29]:


X_train.head()


# In[30]:


y_train.head()


# In[31]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix


# # DecisionTreeClassifier 

# In[32]:


from sklearn.tree import DecisionTreeClassifier 


# In[33]:


tree = DecisionTreeClassifier() 


# In[34]:


tree.fit(X_train,y_train)


# In[35]:


y_pred = tree.predict(X_test)
y_pred


# In[36]:


model2=metrics.accuracy_score(y_test,y_pred)
print(model2)


# In[37]:


cnf_matrix = confusion_matrix(y_test,y_pred)


# In[38]:


labels = [0,1]
sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
plt.show()


# In[ ]:




