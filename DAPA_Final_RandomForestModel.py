#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import DataFrame as DF 


# In[2]:


#!pip install imblearn


# In[3]:


import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set_theme()


# In[4]:



import warnings
warnings.filterwarnings("ignore")


# In[5]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[6]:


rawdf = pd.read_csv ('/Users/prashantwaghela/Documents/DAPA Project Docs/AppleStore.csv')

#Compute a categorical column 'is_priced' using 'price'
rawdf.loc[rawdf['price'] == 0, 'is_priced'] = 0  
rawdf.loc[rawdf['price'] > 0, 'is_priced'] = 1 
rawdf['is_priced'] = rawdf['is_priced'].astype(int)

#Compute app size in megabytes using 'size_bytes'
rawdf['size_mb'] = round(rawdf['size_bytes']/10**6, 2)

#Compute minimum age required based on 'cont_rating'
rawdf.loc[rawdf['cont_rating'] == '4+', 'min_age_req'] = 4
rawdf.loc[rawdf['cont_rating'] == '9+', 'min_age_req'] = 9
rawdf.loc[rawdf['cont_rating'] == '12+', 'min_age_req'] = 12
rawdf.loc[rawdf['cont_rating'] == '17+', 'min_age_req'] = 17
rawdf['min_age_req'] = rawdf['min_age_req'].astype('int')

#Additional column based on prime_genre to compute broader categories
cat_main=['Book','Business','Catalogs','Education','Finance','News','Productivity','Reference','Weather','Entertainment','Food & Drink','Games','Music','Photo & Video','Shopping','Social Networking','Health & Fitness','Lifestyle','Medical','Navigation','Sports','Travel','Utilities']
cat_1=['Book','Business','Catalogs','Education','Finance','News','Productivity','Reference','Weather']
cat_2=['Entertainment','Food & Drink','Games','Music','Photo & Video','Shopping','Social Networking']
cat_3=['Health & Fitness','Lifestyle','Medical','Navigation','Sports','Travel','Utilities']
a=len(cat_1)
b=len(cat_1) + len(cat_2)
c=len(cat_1) + len(cat_2) + len(cat_3)
print(len(cat_main))
for i in range(0,23):
    if(i in range(0,a)):
        print('in cond 1:'+ str(i))
        rawdf.loc[rawdf['prime_genre'] == cat_main[i], 'broad_category'] = 1
    elif(i in range(a,b)):
        print('in cond 2:'+ str(i))
        rawdf.loc[rawdf['prime_genre'] == cat_main[i], 'broad_category'] = 2
    elif(i in range(b,c)):
        print('in cond 3:'+ str(i))
        rawdf.loc[rawdf['prime_genre'] == cat_main[i], 'broad_category'] = 3
        
rawdf['broad_category'] = rawdf['broad_category'].astype('int')

rating_main=[0,1,1.5,2,2.5,3,3.5,4,4.5,5]
rating_1=[0,1,1.5,2,2.5]
rating_2=[3,3.5,4]
rating_3=[4.5,5]
d=len(rating_1)
e=len(rating_1) + len(rating_2)
f=len(rating_1) + len(rating_2) + len(rating_3)
print(len(rating_main))
for i in range(0,len(rating_main)):
    if(i in range(0,d)):
        print('in cond 1:'+ str(i))
        rawdf.loc[rawdf['user_rating'] == rating_main[i], 'rating_category'] = 1
    elif(i in range(d,e)):
        print('in cond 2:'+ str(i))
        rawdf.loc[rawdf['user_rating'] == rating_main[i], 'rating_category'] = 2
    elif(i in range(e,f)):
        print('in cond 3:'+ str(i))
        rawdf.loc[rawdf['user_rating'] == rating_main[i], 'rating_category'] = 3
        
rawdf['rating_category'] = rawdf['rating_category'].astype('int')



from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
#label=le.fit_transform(rawdf['prime_genre'])
#label2=le.fit_transform(rawdf['user_rating'])
#Drop unnecessary columns, rename columns and remove null values
rawdf=rawdf.drop(['id','track_name','currency','ver','cont_rating','size_bytes', 'prime_genre','Unnamed: 0','price','user_rating'],axis=1)
#rawdf['prime_genre']=label
#rawdf['user_rating']=label2
rawdf.rename(columns = {'sup_devices.num':'supported_devices','ipadSc_urls.num':'thumbnails','lang.num':'supported_languages'}, inplace=True)
rawdf.isnull()
final_df=rawdf
print (final_df)
final_df.to_csv('/Users/prashantwaghela/Desktop/file_name.csv', index=False)
final_df2=final_df


# In[7]:


final_df.corr()


# count_3, count_2, count_1=final_df['rating_category'].value_counts()
# 
# rating_3=final_df[final_df['rating_category']==3]
# rating_2=final_df[final_df['rating_category']==2]
# rating_1=final_df[final_df['rating_category']==1]
# #print(rating_3.shape)
# #print(rating_2.shape)
# #print(rating_1.shape)
# 
# np.random.seed(1)
# class_3=rating_3.sample(count_1)
# class_2=rating_2.sample(count_1)
# 
# final_df2 = pd.concat([class_3, class_2, rating_1], axis=0)
# final_df2['rating_category'].value_counts()
# final_df2.head()
# 
# 
# 

# In[25]:


#Data Splitting
count_3, count_2, count_1=final_df['rating_category'].value_counts()

from sklearn.model_selection import train_test_split
train, test = train_test_split(final_df2, stratify=final_df2['rating_category'], test_size=0.25)
train.head()

from collections import Counter
from imblearn.over_sampling import SMOTE
smote=SMOTE(random_state=42)
X,Y=smote.fit_resample(train[["rating_count_tot","rating_count_ver","min_age_req","broad_category","vpp_lic","size_mb","supported_devices","user_rating_ver","thumbnails","supported_languages","is_priced"]], train["rating_category"])
print(sorted(Counter(Y).items()))


# In[ ]:


#Random Forest Model

from sklearn.ensemble import RandomForestClassifier

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 50)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [2,4]
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True, False]

param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(param_grid)




# In[ ]:


model_RF = RandomForestClassifier()
#model_.fit(train[["user_rating_ver","thumbnails","supported_languages","is_priced"]], train["rating_category"])
from sklearn.model_selection import GridSearchCV
rf_Grid = GridSearchCV(estimator = model_RF, param_grid = param_grid, cv = 6, verbose=2, n_jobs = 4)
rf_Grid.fit(train[["user_rating_ver","thumbnails","supported_languages","is_priced"]], train["rating_category"])


# In[9]:


from sklearn.ensemble import RandomForestClassifier
model_RF = RandomForestClassifier(n_estimators=30)


# In[ ]:



print (f'Train Accuracy - : {rf_Grid.score(train[["user_rating_ver","thumbnails","supported_languages","is_priced"]], train["rating_category"]):.3f}')
print (f'Test Accuracy - : {rf_Grid.score(test[["user_rating_ver","thumbnails","supported_languages","is_priced"]], test["rating_category"]):.3f}')
rf_Grid.best_params_
#score_RF = classification_report(test["rating_category"], test["predicted"], labels=[1,2,3])
#print(score_RF)
#cm_RF=rf_Grid.confusion_matrix(test["rating_category"], test["predicted"])
#print(cm_RF)


# In[15]:


model_finalRF = RandomForestClassifier(bootstrap= False, max_depth=4, max_features='auto', min_samples_leaf= 1, min_samples_split= 2, n_estimators= 42)

model_finalRF.fit(train[["rating_count_tot","rating_count_ver","min_age_req","broad_category","vpp_lic","size_mb","supported_devices","user_rating_ver","thumbnails","supported_languages","is_priced"]], train["rating_category"])
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
test.copy()
test["predicted"]=model_finalRF.predict(test[["rating_count_tot","rating_count_ver","min_age_req","broad_category","vpp_lic","size_mb","supported_devices","user_rating_ver","thumbnails","supported_languages","is_priced"]])
test.head()
score_RF = classification_report(test["rating_category"], test["predicted"], labels=[1,2,3])
print(score_RF)
cm_RF=confusion_matrix(test["rating_category"], test["predicted"])
print(cm_RF)

import seaborn as sns
import matplotlib.pyplot as plt
#Plotting the confusion matrix
plt.figure(figsize=(8,7))
sns.heatmap(cm_RF, annot=True,cmap="Blues")
plt.title('Confusion Matrix Before Pruning')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


# In[17]:


sorted_idx = model_RF.feature_importances_.argsort()
plt.barh(train.columns[sorted_idx], model_RF.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.title("Variable of Importance")
print(sorted_idx)


# In[65]:


#plt.bar(train['rating_category'], train['rating_category'].value_counts())
#plt.show()

bar1 = train['rating_category'].value_counts().plot(kind='bar',
                                    figsize=(12,8),
                                    legend=True)
bar1.set_xlabel("Rating Cataegory", fontweight='bold', fontsize=14)
bar1.set_ylabel("Frequency", fontweight='bold', fontsize=14)
bar1.set_title("Initial Rating category Distribution",fontweight='bold', fontsize=16)
plt.xticks((0, 1, 2), ('High Rating', 'Average Rating','Poor Rating'))
plt.xticks(rotation=360)


# In[63]:


bar1 = Y.value_counts().plot(kind='bar', figsize=(12,8), legend=True)
bar1.set_xlabel("Rating Cataegory", fontweight='bold', fontsize=14)
bar1.set_ylabel("Frequency", fontweight='bold', fontsize=14)
bar1.set_title("Balanced Rating category Distribution",fontweight='bold', fontsize=16)
plt.xticks((0, 1, 2), ('High Rating', 'Average Rating','Poor Rating'))
plt.xticks(rotation=360)


# In[ ]:




