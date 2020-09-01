
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE, _random_over_sampler
from sklearn.preprocessing import OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipe
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate

def plot_pca_components(data):
    pca = PCA().fit(data)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    
def check_imbalance(data,label='', x=0.7, y=30000):
    plt.subplots(figsize=(10,8))
    data[label].value_counts().plot(kind='bar')
    text = f'Class Imbalance Count:\n\n{data[label].value_counts().to_dict()}'
    plt.text(x=x, y=y, s = text ,  fontsize=15)
    
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


def x_y_split(data, x=None, y=None, type_="single", test_size=.10):
    
    """
    Single type divides into just x and y
    Double type divides into train and test for each of x and y
    """
    
    X, y = data.drop(columns=y, axis=1), data[y]
    
    if type_ == "single":
        
        return X, y
    
    if type == "double":
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                               test_size=test_size, random_state=123)
        
        return X_train, X_test, y_train, y_test
    
    
    
def model_pipeline(X_train=None, y_train=None, X_test=None, pca=PCA(), cv=StratifiedKFold(), imb_sample=SMOTE(random_state=123),
                  estimator=LogisticRegressionCV()):
    
    """
    Trains a model for an imbalanced class using the specified estimator
    The training is done in K-folds or its nuances as specified folds 
    applying the specified sampling strategy
    """
    
    model = ImbPipe([('imb_sample', imb_sample), ('pca', pca), ('estimator', estimator)])
    model.fit(X_train, y_train) 
    y_hat = model.predict(X_test) 
    return model, y_hat
    
    
def gridSearch(model, hyper_params={},cv=StratifiedKFold(), x_train=None, y_train=None):
    
    """ 
    Performs GridSeach of the best hyperparmaters for the passed model
    """
    
    search = GridSearchCV(estimator=model, param_grid = hyper_params, n_jobs=-1, cv=cv)
    search.fit(X=x_train, y=y_train)
    print("Best parameter (CV score=%0.3f):\n" % search.best_score_)
    print(search.best_params_)
    print(search.score)  
    return search


def plot_grid_search(search_obj, pca_obj, X_train):
    
    """
    Prints the best (optimised) hyperparmatersfor the grid search object
    and plots the optimised pca components
    """
    
    print("Best parameter (CV score=%0.3f):\n" % search_obj.best_score_)
    print("Best Params:",search_obj.best_params_)
    pca_obj.fit(X_train)

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(8, 8))
    ax0.plot(np.arange(1, pca_obj.n_components_ + 1),
             pca_obj.explained_variance_ratio_, '+', linewidth=2)
    ax0.set_ylabel('PCA explained variance ratio')

    ax0.axvline(search_obj.best_estimator_.named_steps['pca'].n_components,
                linestyle=':', label='n_components chosen')
    ax0.legend(prop=dict(size=12))

    # For each number of components, find the best classifier results
    results = pd.DataFrame(search_obj.cv_results_)
    components_col = 'param_pca__n_components'
    best_clfs = results.groupby(components_col).apply(
        lambda g: g.nlargest(1, 'mean_test_score'))

    best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
                   legend=False, ax=ax1)
    ax1.set_ylabel('Classification accuracy (val)')
    ax1.set_xlabel('n_components')

    plt.xlim(-1, 70)

    plt.tight_layout()
    plt.show() 
    

class metrics ():
    
    def __init__(self, y_test, y_hat):
        pass
        self.y_test = y_test
        self.y_hat =  y_hat
        
    
    def class_report(self):
        
        full_report = classification_report(self.y_test, self.y_hat)
        
        print(full_report)
        
    def conf_matrix(self):
        
        conf_matrix = confusion_matrix(self.y_test, self.y_hat)
        
        conf_matrix_df = pd.DataFrame(conf_matrix, columns=['Actual_+ve', 'Actual_-ve'],
                               index=['predicted_+ve', 'predicted_-ve'])
        
        return conf_matrix_df
    
    def accuracy_score(self):
        return  accuracy_score(self.y_test, self.y_hat)
    
    def classification_error(self):
        
        return 1 - accuracy_score() 
        
    def specif_sensitiv(self):
        
        """
        Sensitivity: When the actual value is positive, how often is the prediction correct?
        
        Specificity: When the actual value is negative, how often is the prediction correct?
        """
        
        conf_matrix = confusion_matrix(self.y_test, self.y_hat)
        
        TP = conf_matrix[1, 1]
        TN = conf_matrix[0, 0]
        FP = conf_matrix[0, 1]
        FN = conf_matrix[1, 0]
        
        sensitivity = TP / float(FN + TP)
        specificity = TN / (TN + FP)
        
        sensitiv_specific_table = pd.DataFrame([[sensitivity, specificity]],
                                               columns=['sensitivity', 'specificity'])
        
        return sensitiv_specific_table
