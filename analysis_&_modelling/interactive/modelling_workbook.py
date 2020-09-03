#!/usr/bin/env python
# coding: utf-8

# # Predicting Term Sheet Purchase

# #### Model Buidling Steps

# -  Load Dataset and Clean Data
# -  Check for Outlers and Treat Outliers
# -  Check for Class Imbalances 
# -  Split Data into Train & Test Split
# -  Build Preprocessing and Estimation Pipeline
# - - Using Imblearn pipeline, OverSample minority class, apply PCA, and estimator (Logistic Regression (L1 regularization - lasso ) 
#        & RandomForest Classifier)
# - - Use GridSearch to serach for best parameters and estimators as well as PCA components

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
# 
# - https://ro-che.info/articles/2017-12-11-pca-explained-variance
# 
# - https://www.researchgate.net/deref/http%3A%2F%2Fwww.marcoaltini.com%2Fblog%2Fdealing-with-imbalanced-data-undersampling-oversampling-and-proper-cross-validation

# > Import analysis and visualization libraires

# In[2]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns 
from pandas.api.types import is_numeric_dtype 
import os


# > import from project-defined modules

# In[3]:


import bi_plot
import uni_plot
from data import WrangleData
from model import Preprocessor, plot_pca_components, plot_confusion_matrix, check_imbalance
from model import x_y_split, gridSearch, plot_grid_search
from model_metrics import metrics 


# > Import preprocessing libraries

# In[4]:


from sklearn.preprocessing import StandardScaler,RobustScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from imblearn.over_sampling import SMOTE, _random_over_sampler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, from_model, SelectKBest,chi2, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest # outlier detection and re,oval
from collections import Counter


# > Import estimator libraries

# In[5]:


from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
import xgboost
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, TransformerMixin


# > Import libraries for measuring model perofrmance

# In[6]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate


# > Import production libraries

# In[7]:


from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from imblearn.pipeline import Pipeline as ImbPipe
import joblib


# ## Data Preprocessing

# > Import dataset
# 
# - drop duration column (directly impacts the target varible and not good for modelling)

# In[ ]:





# In[4]:


wrangle = WrangleData()


# In[5]:


wrangle.load_data(path, sep=';')


# In[6]:


wrangle.data


# In[7]:


wrangle.check_outliers()


# In[8]:


wrangle.treat_outliers(type_="isf")


# In[9]:


wrangle.split_data_single(target_cols=['duration']);


# In[10]:


wrangle.split2.head(2)


# In[11]:


wrangle.split1.head(2)


# In[12]:


wrangle.encode(use_split1 = True)


# In[13]:


wrangle.split1


# In[14]:


wrangle.scale_data(use_split2=True)


# In[15]:


wrangle.split2


# In[1]:


"Works Fine"


# In[ ]:





#  

# ### Modelling

# In[20]:


p = Preprocessor() 


# In[21]:


data = wrangle.data 


# In[22]:


data.head(2)


# In[23]:


p.fit(data)


# In[24]:


data.head(2)


# In[25]:


p.check_outliers()


# In[26]:


treated_data = p.treat_outliers(type_='isf')


# In[27]:


treated_data


# In[28]:


p.map_col_values(col_name='y', values_dict={'yes':1, 'no':0})


# In[29]:


p.data


# In[30]:


p.data.rename(columns={'y':'purchases'}, inplace=True)


# In[31]:


p.data


# In[32]:


p.split_data_single(target_cols=['purchases'])


# In[33]:


p.features


# In[34]:


p.target


# In[35]:


p.encode()


# In[36]:


p.features


# In[37]:


p.split_data_double() 


# In[38]:


p.X_train


# #### Train Models

# #### Models to be used
# 
# - LogisticRegression (CV)
# 
# - RandomForest Classifier
# 
# - SVM
# 
# - XGboost
# 
# - MLP (Multi-Layer Perceptron Network)

# In[39]:


from sklearn.pipeline import FeatureUnion, Pipeline  
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline, make_pipeline
from sklearn.decomposition import PCA
from imblearn.metrics import make_index_balanced_accuracy
from sklearn.metrics import balanced_accuracy_score

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.tree import ExtraTreeClassifier
import xgboost
from sklearn.neural_network import MLPClassifier


# ##### model1 - Logistic Regression CV

# In[40]:


model1 = make_pipeline(SMOTE(sampling_strategy=.60), 
                       PCA(n_components=45), LogisticRegressionCV(cv=5,
                                                                  random_state=123, max_iter=1000));


# In[41]:


model1.fit(p.X_train,p.y_train.purchases.ravel());


# ##### Test model1

# In[42]:


model1_pred = model1.predict(p.X_test)


# In[43]:


model1_pred


# In[44]:


balanced_accuracy_score(p.y_test, model1_pred)


# ##### Train model2 - Logistic Regression

# In[45]:


model2 = make_pipeline(SMOTE(sampling_strategy=.60), PCA(n_components=45), LogisticRegression(random_state=124
                                                                                             ,max_iter=1000))


# In[46]:


model2.fit(p.X_train,p.y_train.purchases.ravel());


# ##### Test Model2

# In[47]:


model2_pred = model2.predict(p.X_test)
model2_pred


# In[48]:


balanced_accuracy_score(p.y_test, model2_pred) 


# ##### Train Model3 - RandomForest Classifier (SkLearn Implementation)

# In[49]:


from sklearn.ensemble import RandomForestClassifier


# In[50]:


model3 = make_pipeline(SMOTE(sampling_strategy=.60), PCA(n_components=45), RandomForestClassifier())
model3.fit(p.X_train,p.y_train.purchases.ravel()); 


# ##### Test Model3

# In[51]:


model3_pred = model3.predict(p.X_test)
balanced_accuracy_score(p.y_test, model3_pred) 


#  

#  

# ##### Train Model4 - RandomForest Classifier (Imblearn Implementation)

# In[52]:


from imblearn.ensemble import BalancedRandomForestClassifier


# In[53]:


model4 = make_pipeline(SMOTE(sampling_strategy=.60), PCA(n_components=45), BalancedRandomForestClassifier())
model4.fit(p.X_train,p.y_train.purchases.ravel());


# ##### Test Model3

# In[54]:


model4_pred = model4.predict(p.X_test)
model4_pred_proba = model4.predict_proba(p.X_test)


# > Probability of each input belonging to a particular class (purchase and no-purchase)

# In[55]:


model4_pred_proba


# > model score

# In[56]:


balanced_accuracy_score(p.y_test, model4_pred)  


# > The BalancedRandomForestClassifier Implementation performs better than  RandomForestClassifier

# ##### Train Model5 - SVM Classification

# In[57]:


from sklearn.svm import SVC


# In[58]:


model5 = make_pipeline(SMOTE(sampling_strategy=.60), PCA(n_components=45), SVC())
model5.fit(p.X_train,p.y_train.purchases.ravel());


# Test Model5

# In[59]:


model5_pred = model5.predict(p.X_test)


# In[60]:


balanced_accuracy_score(p.y_test, model5_pred)  


#  

# ### Measuring Model Performance

# In[134]:


y_train = p.y_train.purchases.ravel()
y_test = p.y_test.purchases.ravel()


# ##### 1. Evaluating Model Performances
# > put all models in a list and create Metrics object

# In[135]:


models = [model1, model2, model3, model4, model5]
model_evaluations = Metrics(p.X_train, y_train, p.X_test, y_test)


# > Create an empty Dataframe for models evaluations

# In[136]:


models_scores = pd.DataFrame()


# > Loop through models and evaluate

# In[65]:


for model in models:
    
    models_scores = model_evaluations.evaluate_classifier(clf=model, models_eval_scores = models_scores);


# ##### Comparison of Model Performances (Acuuracy Vs Balanced Accuracy)

# In[68]:


models_scores.T


#  

# #### 2. Test conf_matrix, accuracy_score, classification_error, specif_sensitiv

# In[91]:


predictions = [model1_pred, model2_pred, model3_pred, model4_pred, model5_pred]


# confusion matrix

# In[101]:


count = 0
for pred in predictions:
    count+=1
    print(f'model {count} confusion matrix -->\n{model_evaluations.conf_matrix(pred)}\n')


# balanced accuracy score

# In[102]:


count = 0
for pred in predictions:
    count+=1
    print(f'model {count} confusion matrix -->\n{model_evaluations.balanced_accuracy_score(pred)}\n')


# balanced classification error

# In[112]:


count = 0
for pred in predictions:
    count+=1
    print(f'model {count} balanced classfication error -->\n{model_evaluations.balanced_classification_error(pred)}\n')


# In[137]:


count = 0
for pred in predictions:
    count+=1
    print(f'model {count} specificity Vs. sensitivuty -->\n{model_evaluations.specif_sensitiv(pred)}\n')


# In[ ]:





# In[191]:


x = model1_pred[[np.where(model1_pred == 0)]], model1_pred[[np.where(model1_pred == 1)]]


# In[228]:


y = [[1,1,1,1,1], [0,0,0,0,0]]


# In[ ]:





# In[223]:


pd.DataFrame(x[0].astype(list)).append(pd.DataFrame(x[1].astype(list))).T


# ###  Building Model Pipelines

# In[ ]:


#%%writefile pipeline.py
#%%writefile ../scripts/project_package/model_package/pipeline.py 
from sklearn.pipeline import FeatureUnion, Pipeline  
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline, make_pipeline

