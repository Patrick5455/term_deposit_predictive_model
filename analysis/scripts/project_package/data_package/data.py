
import seaborn as sns
import matplotlib.pyplot as plt 
import os
import numpy as  np
import pandas as pd
import pandas
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.base import *


class WrangleData():
    
    def __repr__(self):
        
        return "Used to prepare data for wrangling"
    
    def  __init__(self):
        
        pass
        
    
    
    def load_data(self, path="", sep=",", cols_to_drop=[]):
        
        import pandas as pd
        
        self.path = path
        self.cols_to_drop = cols_to_drop 
        self.sep = sep 
            
        try :
            self.data = pd.read_csv(path, sep)
            
            if len(self.cols_to_drop) > 0:
                for col in self.cols_to_drop:
                    self.data.drop(col, axis=1, inplace=True)
                    
            self.fit()
            print("You are now fit to use this object for wrangling")
            
            return self.data 
        
        except:
            
            "No data path was passed upon call of method load_data"
    
    def _fit(self):
        
        try:
            assert(type(self.data) is pandas.core.frame.DataFrame), "data must be of type pandas.DataFrame"
            
            print("You are now fit to use this object for wrangling")
        
        except AttributeError:
            
            print("Hey Buddy you need to load a data first !!! ")
        


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
            
            5. isf    -       IsolationForest (also like trimming)

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
        
        """
        replace values in a series (values_dict.keys) with specified values from (values_dict.values)
        """

        self.data[col_name] = self.data[col_name].map(values_dict)

        return self.data
    
    
    def split_data_single(self, target_cols=[]):
            
        self.split1 = self.data.drop(columns=target_cols, axis=1) 

        self.split2   = pd.DataFrame(self.data[target_cols])

        return self.split1, self.split2
    
    
    def encode (self, use_split1=False, use_split2 = False, use_data=False): 
        
        if use_data:
      
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
        
        
        if use_split1:
      
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore', )
            to_encode = self.split1.select_dtypes(exclude='number')
            if self.split1.shape[1] > 1:
                #ohe = MultiLabelBinarizer()
                self.split1.drop(to_encode.columns.tolist(), axis=1, inplace = True)
                features_cat_encode = pd.DataFrame(ohe.fit_transform(to_encode))
                self.split1 = self.split1.merge(features_cat_encode, left_index=True, right_index=True)
               # print(ohe.classes_) 
            else: 
                self.split1 = pd.DataFrame(ohe.fit_transform(to_encode))
                print(ohe.categories_) 

            return self.split1
        
        
        if use_split2:
      
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore', )
            to_encode = self.split2.select_dtypes(exclude='number')
            if self.split2.shape[1] > 1:
                #ohe = MultiLabelBinarizer()
                self.split2.drop(to_encode.columns.tolist(), axis=1, inplace = True)
                features_cat_encode = pd.DataFrame(ohe.fit_transform(to_encode))
                self.split2 = self.split2.merge(features_cat_encode, left_index=True, right_index=True)
               # print(ohe.classes_) 
            else: 
                self.split2 = pd.DataFrame(ohe.fit_transform(to_encode))
                print(ohe.categories_) 

            return self.split2
            

    def scale_data(self, scaler=RobustScaler(),
                  use_data=False, use_split1= False, use_split2 = False):
        
        """
            Specify scaler type, scaler type must have fit_transform as a method
        """
        
        if use_data:
            self.data = scaler.fit_transform(self.data)
            return self.data 
        
        if use_split1:
            self.split1 = scaler.fit_transform(self.split1)
            return self.split1
        
        if use_split2:
            self.split2 = scaler.fit_transform(self.split2)
            return self.split2
            
