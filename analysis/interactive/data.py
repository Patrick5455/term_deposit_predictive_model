# %%writefile ../scripts/data.py

import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as  np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import IsolationForest
import os
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split


class PrepareData():
    
    def __repr__(self):
        
        return "Used to prepare data for modelling"
    
    def  __init__(self, path=""):
        self.path = path
    
    def load_data(self, path="", sep=",", cols_to_drop=[]):
        self.data = pd.read_csv(self.path, sep)
        for col in cols_to_drop:
            self.data.drop(col, axis=1, inplace=True)

        return self.data 


    def check_outliers(self, data, show_plot=False, save_img=os.getcwd()+'/outliers.png'):
        
        self.data = data
    
        """
        This functions checks for columns with outlers using the IQR method

        It accespts as argmuent a dataset. 
        show_plot can be set to True to output pairplots of outlier columns    
        """

        outliers = [] 
        Q1 = self.data.quantile(0.25)  
        Q3 = self.data.quantile(0.75)
        IQR = Q3 - Q1
        num_data = self.data.select_dtypes(include='number')
        result = dict ((((num_data < (Q1 - 1.5 * IQR)) | (num_data > (Q3 + 1.5 * IQR)))==True).any())
        #data[(data[col] >= high)|(data[col] <= low)].index
        index = self.data[(num_data < Q1 - 1.5 * IQR) | (num_data > Q3 + 1.5 * IQR)].index
        for k,v in result.items():
            if v == True:  
                outliers.append(k)
        if show_plot:
            pair_plot = sns.pairplot(self.data[outliers]);
            print(f'{result},\n\n Visualization of outlier columns')
            plt.savefig(fname=save_img, format='png')
            return pair_plot
        else:
            return self.data.loc[index, outliers] 
        
        
    def treat_outliers(self, data, type='median_replace'):
        
        self.data = data
    
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

        if type == "median_replace":

            for col in self.data.columns.tolist():
                if is_numeric_dtype(self.data[col]):
                    median = (self.data[col].quantile(0.50))
                    print(median)
                    q1 = self.data[col].quantile(0.25)
                    q3 = self.data[col].quantile(0.75)
                    iqr = q3 - q1
                    high = int(q3 + 1.5 * iqr) 
                    low = int(q1 - 1.5 * iqr)
                    print(high, low, iqr)
                    print(col)
                    self.data[col] = np.where(self.data[col] > high, median, self.data[col])
                    self.data[col] = np.where(self.data[col] > high, median, self.data[col])        

        if type == "quant_floor":

            for col in self.data.columns.tolist():
                if is_numeric_dtype(data[col]):
                    q_10 = self.data[col].quantile(0.5)
                    q_90 = self.data[col].quantile(0.95)
                    self.data[col] =  self.data[col] = np.where(self.data[col] < q_10, q_10 , self.data[col])
                    self.data[col] =  self.data[col] = np.where(self.data[col] > q_90, q_90 , self.data[col])

        if type == "trim":

            for col in self.data.columns.tolist():
                low = .05
                high = .95
                quant_df = self.data.quantile([low, high])
                for name in list(self.data.columns):
                    if is_numeric_dtype(self.data[name]):
                        self.data = self.data[(self.data[name] >= quant_df.loc[low, name]) 
                            & (self.data[name] <= quant_df.loc[high, name])]

        if type == "log_transform":  
            for col in self.data.columns.tolist():
                if is_numeric_dtype(self.data[col]):
                    self.data[col] = self.data[col].map(lambda i: np.log(i) if i > 0 else 0)

        if type == "isf":
            iso = IsolationForest(contamination=0.1)
            yhat = iso.fit_predict(self.data.select_dtypes(exclude='object'))
            #select all rows that are not outliers
            mask = yhat != -1 
            self.data = self.data[mask]


        return self.data 
    
    
    def map_col_values(self, data, col_name="", values_dict={}):

        self.data = data

        self.data[col_name] = self.data[col_name].map(values_dict)

        return self.data
    
    
    def split_data_single(self, data, target_cols=[]):
        
        self.data = data
    
        self.features = self.data.drop(columns=target_cols, axis=1) 

        self.target   = pd.DataFrame(self.data[target_cols])

        return self.features, self.target
    
    
    def encode (self, data):
        
        self.data = data
        
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore', )
        to_encode = self.data.select_dtypes(exclude='number')
        if self.data.shape[1] > 1:
            #ohe = MultiLabelBinarizer()
            self.data.drop(to_encode.columns.tolist(), axis=1, inplace = True)
            features_cat_encode = pd.DataFrame(ohe.fit_transform(to_encode))
            self.data = self.data.merge(features_cat_encode, left_index=True, right_index=True)
            #print(ohe.classes_) 
        else:
            self.data = pd.DataFrame(ohe.fit_transform(to_encode))
            print(ohe.categories_) 
        return self.data
    
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


    def scale_data(self, data,scaler=RobustScaler()):
        
        self.data = data

        """
        Specify scaler type, scaler type must have fit_transform as a method

        """
        self.data_scaled = scaler.fit_transform(self.data)
        
        return self.data_scaled
