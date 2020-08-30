#!/usr/bin/env python
# coding: utf-8

# # Predicting Term Sheet Purchase

# #### Model Buidling Steps

# - 
# - 
# - 
# - 
# - 
# - 
# - 

# #### References

# - https://stackoverflow.com/questions/46062679/right-order-of-doing-feature-selection-pca-and-normalization
# 
# - https://towardsdatascience.com/preventing-data-leakage-in-your-machine-learning-model-9ae54b3cd1fb
# 
# - https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/
# 
# - https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
# 
# - https://stats.stackexchange.com/questions/363548/use-of-smote-with-training-test-and-dev-sets
# 
# - https://datascience.stackexchange.com/questions/27615/should-we-apply-normalization-to-test-data-as-well

# > Import analysis and visualization libraires

# In[449]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype


# > Import dataset
# 
# - drop duration column (directly impacts the target varible and not good for modelling)

# In[450]:


path= '../../datasets/main_data/bank-additional-full.csv'
full_bank = pd.read_csv(path, sep=';')


# In[451]:


full_bank.shape


# In[452]:


full_bank.drop('duration', axis=1, inplace=True)


# In[453]:


full_bank.shape


# > import project-defined modules

# In[454]:


from plot import plot_univariate, plot_bivariate
from data import check_outliers, treat_outliers


# > Import preprocessing libraries

# In[455]:


from sklearn.preprocessing import StandardScaler,RobustScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from imblearn.over_sampling import SMOTE, _random_over_sampler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, from_model, SelectKBest,chi2, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest # outlier detection and re,oval
from collections import Counter


# > Import estimator libraries

# In[456]:


from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


# > Import libraries for measuring model perofrmance

# In[457]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate


# In[ ]:





# > Import production libraries

# In[458]:


from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
import joblib


# In[ ]:





# #### Data Preprocessing

# In[459]:


full_bank


# In[460]:


full_bank.dtypes


# In[461]:


full_bank.isna().any()


# #### Outlier Detection and Treatment

# > check for outliers

# In[462]:


check_outliers(full_bank)


# > treat outliers

# In[463]:


iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(full_bank.select_dtypes(exclude='object'))
# select all rows that are not outliers
mask = yhat != -1 
clean_df1 = full_bank[mask]
# summarize the shape of the updated training dataset
print(clean_df1.shape)


# In[464]:


clean_df1


# #### Now let's check for class imbalance

# In[465]:


plt.subplots(figsize=(10,8))
clean_df1['y'].value_counts().plot(kind='bar')
plt.text(x=0.7, y=30000, s=f'Class Imbalance Count:\n\n{clean_df1.y.value_counts().to_dict()}', fontsize=15)


# > We have a very high class imbalance
# 
# > This would be dealt with after splitting out data to train and test and applied only to train to avoid data leakage

# #### split data
# 
# > Beofore further preprocessing, it is important we split the data into train and test set to avoid `data leakage`

# In[466]:


features, target = clean_df1.drop('y', axis=1), pd.DataFrame(clean_df1['y'])


# #### encode categorical variables

# In[467]:


def encode (data):
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore', )
    to_encode = data.select_dtypes(exclude='number')
    if data.shape[1] > 1:
        #ohe = MultiLabelBinarizer()
        data.drop(to_encode.columns.tolist(), axis=1, inplace = True)
        features_cat_encode = pd.DataFrame(ohe.fit_transform(to_encode))
        data = data.merge(features_cat_encode, left_index=True, right_index=True)
        #print(ohe.classes_) 
    else:
        data = pd.DataFrame(ohe.fit_transform(to_encode))
        print(ohe.categories_) 
    return data 


# In[468]:


features = encode(features)
features


# In[469]:


target = encode(target)


# In[470]:


target


# > make shape equal

# In[471]:


target=target.iloc[0:features.shape[0]]


# In[472]:


target


# > trim target varibles `purchase` to a single column of `yes`

# In[473]:


target.drop(0, axis=1, inplace=True)


# In[474]:


target


# In[475]:


X_train, X_test, y_train, y_test = train_test_split(features, target,
                                               test_size=.10, random_state=1234)


# In[476]:


X_train


# In[477]:


y_train


# #### Deal with class imbalance

# In[478]:


y_train[1].value_counts()


# > Since we have an imbalance case of very high majority vs very low minority,
# 
# > a good decision would be to use the SMOTE technique of oversampling the minority to mathc the majority class

# In[479]:


oversample = SMOTE(random_state=1234)
X_train, y_train = oversample.fit_sample(X_train, y_train)


# In[480]:


X_train


# In[481]:


y_train


# > compare the class distribution

# In[482]:


trains = pd.merge(X_train, y_train, left_index=True, right_index=True)


# In[483]:


trains


# In[484]:


trains_class = trains['1_y'].value_counts().to_dict() 
trains_class


# In[485]:


plt.subplots(figsize=(10,8))
trains['1_y'].value_counts().plot(kind='bar')
plt.text(x=0.2, y=30000, s=f'Class Imbalance Count:\n\n{trains_class}', fontsize=15)


# > We now have a balanced class and can go on to further preprocess

# #### Normalize X_train and X_test datasets
# 
# > I would be using the RobustScaler which is less prone to outliers

# In[486]:


scaler = RobustScaler()
scaler


# In[487]:


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


# In[488]:


X_train_scaled


# In[489]:


X_test_scaled


# In[490]:


Y_train = scaler.fit_transform(Y_train)


# In[491]:


Y_train


# In[492]:


pd.DataFrame(Y_train)[0].value_counts()


# #### Dimensionality Reduction

# > RIght now, our input variables are quite large, which has potential of affecting our prediciton
# 
# > To optimise the predictive features of our variables and save memory space on the model,
# 
# > we can reduce the number of features using PCA

# First let's plot the number of components we need to get the most explained variance of our data

# - Xtrain

# In[493]:


pca = PCA().fit(X_train_scaled)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# From the plot above, we can see that we need up to 50 components to get an high explained variance

# > This means whar we need for our prediction is feature extraction  and not feature reduction

# In[401]:


pca = PCA(50, random_state=1234)  # project from 64 to 2 dimensions
X_train_reduced = pca.fit_transform(X_train_scaled)
X_test_reduced = pca.fit_transform(X_test_scaled)
print(X_train_reduced.shape) 
print(x_test_reduced.shape)


# In[402]:


X_train_reduced


# In[403]:


pca.explained_variance_ratio_


# In[ ]:





# #### Perform feature selection on numerical features
# 
# > At first I will plot the feature importacne for all featurs then decide on the number to use depending on the number of features

# In[167]:


features_num = features.select_dtypes(include='number')
features_num.shape


# In[179]:


scaler = StandardScaler()


# In[ ]:





# In[ ]:





# In[168]:


# define feature selection, 
fs = SelectKBest(score_func=f_classif, k='all')
fs.fit(features_num, target)


# > PLot Feature Importance

# In[169]:


for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()


# In[170]:


num_kbest = pd.DataFrame(fs.scores_, columns=['scores']).sort_values(
    ascending=False, by='scores').reset_index().rename(columns={'index':'features'})
num_kbest


# In[171]:


plt.subplots(figsize=(12,10))
sns.barplot(data=num_kbest, x='features', y='scores', order=num_kbest.features)
plt.title('Feature Importance for numerical variables')
plt.show()


# > I would go with features that equals or above the average score

# In[172]:


num_kbest_avg = num_kbest.scores.mean()
num_kbest_avg


# In[173]:


num_kbest=num_kbest.query('scores >= @num_kbest_avg')
num_kbest


# > We have only 4 of such features

# > about four features are most useful for our prediction, hence I will go for 4

# In[175]:


# define feature selection, 
fs = SelectKBest(score_func=f_classif, k=4)
# apply feature selection
num_selected_kbest = fs.fit_transform(features_num, target)
print(num_selected_kbest.shape)


# In[176]:


num_selected_kbest


# > 

# #### Perform feature selection on categorical features
# 
# > I would take the same approach as the numerical vars on the cat vars
# 
# > But first, we have to encode our categorical variables using OneHotEncoding

# In[130]:


fs_1 = SelectKBest(score_func=chi2, k='all')
fs_1.fit(features_cat_encode, target)


# In[144]:


for i in range(len(fs_1.scores_)):
	print('Feature %d: %f' % (i, fs_1.scores_[i]))


# In[152]:


cat_kbest = pd.DataFrame(fs_1.scores_, columns=['scores']).sort_values(
    ascending=False, by='scores').reset_index().rename(columns={'index':'features'})
cat_kbest.head(10)


# In[158]:


plt.subplots(figsize=(12,10))
sns.barplot(data=cat_kbest, x='features', y='scores', order=cat_kbest.features)
plt.title('Feature Importance for categorical variables')
plt.show()


# > I would go with features that equals or above the average score

# In[160]:


cat_kbest_avg = cat_kbest.scores.mean()
cat_kbest_avg


# In[164]:


cat_kbest=cat_kbest.query('scores >= @cat_kbest_avg')
cat_kbest


# > We have only 12 of such features

# In[ ]:


> I would concatenate 


# In[ ]:





# In[ ]:





# In[ ]:





# > encode categorical variables in features

# > merge with train data

# In[ ]:


for col in to


# In[ ]:




