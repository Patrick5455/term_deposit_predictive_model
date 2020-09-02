

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
import os


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate 


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin

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
    
    
def gridSearch(model,hyper_params={},cv=StratifiedKFold(), x_train=None, y_train=None):
    
    """
    Performs GridSeach of the best hyperparmaters for the passed model
    """
    
    search = GridSearchCV(model=model, param_grid = hyper_params, n_jobs=-1, cv=cv)
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
    
    print("Best parameter (CV score=%0.3f):\n" % search.best_score_)
    print("Best Params:",search.best_params_)
    pca.fit(X_train_scaled)

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(8, 8))
    ax0.plot(np.arange(1, pca.n_components_ + 1),
             pca.explained_variance_ratio_, '+', linewidth=2)
    ax0.set_ylabel('PCA explained variance ratio')

    ax0.axvline(search.best_estimator_.named_steps['pca'].n_components,
                linestyle=':', label='n_components chosen')
    ax0.legend(prop=dict(size=12))

    # For each number of components, find the best classifier results
    results = pd.DataFrame(search.cv_results_)
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
    
    
class Preprocessor(BaseEstimator, TransformerMixin):
    
    def __repr__(self):
        
        return "Used to prepare data for modelling"
    
    def  __init__(self):
        
        pass
        
    
    def fit(self, data, y=None):
        
        assert(type(data) is pandas.core.frame.DataFrame), "data must be of type pandas.DataFrame"
        
        self.data = data 
        
        print("Fitted")
        
        return self 
        


    def check_outliers(self, show_plot=False, save_img=os.getcwd()+'/outliers.png'):
            
        """
        This functions checks for columns with outlers using the IQR method

        It accespts as argmuent a dataset. 
        show_plot can be set to True to output pairplots of outlier columns    
        """

        self.outliers = [] 
        Q1 = self.data.quantile(0.25)  
        Q3 = self.data.quantile(0.75)
        IQR = Q3 - Q1
        num_data = self.data.select_dtypes(include='number')
        result = dict ((((num_data < (Q1 - 1.5 * IQR)) | (num_data > (Q3 + 1.5 * IQR)))==True).any())
        #data[(data[col] >= high)|(data[col] <= low)].index
        index = self.data[(num_data < Q1 - 1.5 * IQR) | (num_data > Q3 + 1.5 * IQR)].index
        for k,v in result.items():
            if v == True:  
                self.outliers.append(k)
        if show_plot:
            self.outlier_pair_plot = sns.pairplot(self.data[self.outliers]);
            print(f'{result},\n\n Visualization of outlier columns')
            plt.savefig(fname=save_img, format='png')
            return  self.outlier_pair_plot
        else:
            return self.data.loc[index, self.outliers] 
        
        
    def treat_outliers(self, type_='median_replace'):
            
        """
        This treat outliers using any ofthses 3 methods as specified by user

            1. median_replace -  median replacement

            2. quant_floor - quantile flooring

            3. trim - trimming 

            4. log_transform - log transformations

        The methods are some of the commont statistical methods in treating outler
        columns

        By default treatment type is set to median replacement

        """

        if type_ == "median_replace":

            for col in self.data.columns.tolist():
                if is_numeric_dtype(self.data[col]):
                    median = (self.data[col].quantile(0.50))
                    q1 = self.data[col].quantile(0.25)
                    q3 = self.data[col].quantile(0.75)
                    iqr = q3 - q1
                    high = int(q3 + 1.5 * iqr) 
                    low = int(q1 - 1.5 * iqr)
                    self.data[col] = np.where(self.data[col] > high, median, self.data[col])
                    self.data[col] = np.where(self.data[col] > high, median, self.data[col])        

        if type_ == "quant_floor":

            for col in self.data.columns.tolist():
                if is_numeric_dtype(data[col]):
                    q_10 = self.data[col].quantile(0.5)
                    q_90 = self.data[col].quantile(0.95)
                    self.data[col] =  self.data[col] = np.where(self.data[col] < q_10, q_10 , self.data[col])
                    self.data[col] =  self.data[col] = np.where(self.data[col] > q_90, q_90 , self.data[col])

        if type_ == "trim": 

            for col in self.data.columns.tolist():
                low = .05
                high = .95
                quant_df = self.data.quantile([low, high])
                for name in list(self.data.columns):
                    if is_numeric_dtype(self.data[name]):
                        self.data = self.data[(self.data[name] >= quant_df.loc[low, name]) 
                            & (self.data[name] <= quant_df.loc[high, name])]

        if type_ == "log_transform":  
            for col in self.data.columns.tolist():
                if is_numeric_dtype(self.data[col]):
                    self.data[col] = self.data[col].map(lambda i: np.log(i) if i > 0 else 0)

        if type_ == "isf":
            iso = IsolationForest(contamination=0.1)
            yhat = iso.fit_predict(self.data.select_dtypes(exclude='object'))
            #select all rows that are not outliers
            mask = yhat != -1 
            self.data = self.data[mask]


        return self.data 
    
    
    def map_col_values(self, col_name="", values_dict={}):

        self.data[col_name] = self.data[col_name].map(values_dict)

        return self.data
    
    
    def split_data_single(self, target_cols=[]):
            
        self.features = self.data.drop(columns=target_cols, axis=1) 

        self.target   = pd.DataFrame(self.data[target_cols])

        return self.features, self.target
    
    
    def encode (self, data_obj=None, use_features=True, use_target=False): 
        
        if data_obj is None and use_features == False and use_target == False:
        
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore', )
            to_encode = self.data.select_dtypes(exclude='number')
            if self.data.shape[1] > 1:
                #ohe = MultiLabelBinarizer()
                self.data.drop(to_encode.columns.tolist(), axis=1, inplace = True)
                features_cat_encode = pd.DataFrame(ohe.fit_transform(to_encode))
                self.data = self.data.merge(features_cat_encode, left_index=True, right_index=True)
               # print(ohe.classes_) 
            else: 
                self.data = pd.DataFrame(ohe.fit_transform(to_encode))
                print(ohe.categories_) 
            return self.data
        
        if data_obj is not None:
        
            self.data_obj = data_obj
            print("Not None")
            
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore', )
            to_encode = self.data_obj.select_dtypes(exclude='number')
            if self.data_obj.shape[1] > 1:
                #ohe = MultiLabelBinarizer()
                self.data_obj.drop(to_encode.columns.tolist(), axis=1, inplace = True)
                features_cat_encode = pd.DataFrame(ohe.fit_transform(to_encode))
                self.data_obj = self.data_obj.merge(features_cat_encode, left_index=True, right_index=True)
               # print(ohe.classes_) 
            else:
                self.data_obj = pd.DataFrame(ohe.fit_transform(to_encode))
                print(ohe.categories_) 
            return self.data_obj
        
        if use_features:
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore', )
            to_encode = self.features.select_dtypes(exclude='number')
            if self.features.shape[1] > 1:
                #ohe = MultiLabelBinarizer()
                self.features.drop(to_encode.columns.tolist(), axis=1, inplace = True)
                features_cat_encode = pd.DataFrame(ohe.fit_transform(to_encode))
                self.features = self.features.merge(features_cat_encode, left_index=True, right_index=True)
               # print(ohe.classes_) 
            else: 
                self.features = pd.DataFrame(ohe.fit_transform(to_encode))
                print(ohe.categories_) 
            return self.features
        
        if use_target:
            
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore', )
            to_encode = self.target.select_dtypes(exclude='number')
            if self.target.shape[1] > 1:
                #ohe = MultiLabelBinarizer()
                self.target.drop(to_encode.columns.tolist(), axis=1, inplace = True)
                features_cat_encode = pd.DataFrame(ohe.fit_transform(to_encode))
                self.target = self.target.merge(features_cat_encode, left_index=True, right_index=True)
               # print(ohe.classes_) 
            else: 
                self.target = pd.DataFrame(ohe.fit_transform(to_encode))
                print(ohe.categories_) 
            return self.target
            
    
    def split_data_double(self, features_=pd.DataFrame([[]]), target_=pd.DataFrame([[]]), 
                          test_size=.10, use_native=True):
        
        if use_native == False:
        
            if features.shape[0] != target.shape[0]:
                
                raise Exception("Wrong, you are trying to pass unequal shapes\n\
                Shapes of dataframes must be equal\n\
                Try target = target.iloc[0:features.shape[0]]")

            self.features_ = features_
            self.target_ = target_

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features_, 
                                                                                    self.target_,
                                           test_size= test_size, random_state=24)

            return self.X_train, self.X_test, self.y_train, self.y_test
        
        if use_native:
            
            self.target = self.target.iloc[0:self.features.shape[0]]
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.target,
                                           test_size= test_size, random_state=24)

            return self.X_train, self.X_test, self.y_train, self.y_test
        
    
    
    


    def scale_data(self, scale_data=pd.DataFrame([[]]),
                   scaler=RobustScaler(), use_features=True,
                  use_target=False, use_data=False):
        
        """
            Specify scaler type, scaler type must have fit_transform as a method
        """
        
        if use_features:
    
            self.features = scaler.fit_transform(self.features)

            return self.features
        
        if use_target:
            
            self.target = scaler.fit_transform(self.target)

            return self.target
        
        if use_data:
            
            self.data = scaler.fit_transform(self.data)

            return self.data
        
        if use_data == False and use_features == False and use_target == False:
            
            self.scale_data = scale_data
            
            self.scale_data = scaler.fit_transform(self.scale_data)

            return self.scale_data
            
    def transform(self, X):
        
        """
        Ideally, a preapred trainX data ought to be passed to in case of passing into a pipeline
        """
        
        self.data = X
                
        self.data = self.treat_outliers(type_="isf") 
        
        #self.data = self.map_col_values(col_name="y", values_dict={"no":0, "yes":1})
        
       # self.features, self.target = self.split_data_single(target_cols=["y"])
        #print(self.features)
                
        self.features = self.encode(self.features)
      #  self.target = self.target.iloc[0:self.features.shape[0], 0:]
        #print(self.target)
       # self.X_train, self.X_test, self.y_train, self.y_test = self.split_data_double(
        #    self.features, self.target, test_size=.10)
        
        scaler=RobustScaler() 
            
        X = scaler.fit_transform(self.X_train)
        
        return X
    
    
    def fit_transform(self, X, y=None):
        
        self.X = X
        
        return self.transform(self.X) 
