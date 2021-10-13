#!/usr/bin/env python
# coding: utf-8

# # Importing Essential Libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # Importing Dataset

# In[7]:


df = pd.read_csv('in-vehicle-coupon-recommendation.csv')
df.head()


# # Defining Supervisor

# In[8]:


y = df['Y']
y


# # Redefining Dataset

# In[9]:


df = df.drop(['Y'] , axis=1 )
df


# In[10]:


X = df
X


# # Label Encoding

# In[12]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
y


# In[13]:


X = X.apply(LabelEncoder().fit_transform)
X


# # Defining Training and Testing Data

# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)

print ( X_train )
print ( X_test )
print ( y_train )
print ( y_test )


# # Data Normalization

# In[16]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train)
print(X_test)


# # Training Model

# In[17]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# # Prediction

# In[18]:


y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# # Confusion Matrix and Accuracy 

# In[20]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy is : ",  accuracy_score(y_test, y_pred))
print("Precision is : ", precision_score(y_test, y_pred))
print("Recall Score is : ", recall_score(y_test, y_pred))


# In[21]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# # Decision Tree

# In[27]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


# # Graph

# In[28]:


from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus


# In[29]:


feature_cols = list(df.columns)

dot_data = StringIO()

export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Coupon.png')
Image(graph.create_png())


# In[ ]:




