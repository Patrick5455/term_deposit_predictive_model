# %%writefile ../scripts/data.py

import seaborn as sns
import matplotlib.pyplot as plt 
import os
import numpy as  np
import pandas as pd
import pandas
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import IsolationForest
import os
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin


def load_data(path="", sep=",", cols_to_drop=[]):
            
        try :
            data = pd.read_csv(path, sep)
            
            if len(cols_to_drop) > 0:
                for col in cols_to_drop:
                    data.drop(col, axis=1, inplace=True)

            return data 
        
        except:
            
            "No data path was passed upon inastantiation of object"
    


# define class Preprocess to preprocess data
# class Preprocess inherits from BaseEstimator & TransformerMixin
# the idea behind the Preprocess class is to preprocess our data ready for modelling

class Preprocessor(BaseEstimator, TransformerMixin):
    
    def __repr__(self):
        
        return "Used to prepare data for modelling"
    
    def  __init__(self):
        
        pass
        
#         self.path = path
#         self.cols_to_drop = cols_to_drop 
#         self.sep = sep 
        
        
    
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
            return self.data.loc[index, outliers] 
        
        
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
    
    
    def encode (self, data_obj=None): 
        
        if data_obj is None:
            print("Not None")
        
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
        
        else:
            
            self.data_obj = data_obj
            
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
            
    
    def split_data_double(self, features, target, test_size=.10):
        
        if features.shape[0] != target.shape[0]:
            raise Exception("Wrong, you are trying to pass unequal shapes\n\
            Shapes of dataframes must be equal\n\
            Try target = target.iloc[0:features.shape[0]]")
        
        self.features = features
        self.target = target
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.target,
                                               test_size= test_size, random_state=24)
        
        return self.X_train, self.X_test, self.y_train, self.y_test


    def scale_data(self, scaler=RobustScaler()):
    
        """
        Specify scaler type, scaler type must have fit_transform as a method

        """
        self.data_scaled = scaler.fit_transform(self.data)
        
        return self.data_scaled
    
    
    def transform(self, X):
        
        self.data = X
                
        self.data = self.treat_outliers(type_="isf") 
        
        self.data = self.map_col_values(col_name="y", values_dict={"no":0, "yes":1})
        
        self.features, self.target = self.split_data_single(target_cols=["y"])
        #print(self.features)
                
        self.features = self.encode(self.features)
        self.target = self.target.iloc[0:self.features.shape[0], 0:]
        #print(self.target)
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data_double(
            self.features, self.target, test_size=.10)
        
        scaler=RobustScaler() 
            
        X = scaler.fit_transform(self.X_train)
        
        return X
    
    
    def fit_transform(self, X, y=None):
        
        self.X = X
        
        return self.transform(self.X) 
