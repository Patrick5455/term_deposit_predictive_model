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

def load_data(path="", sep=",", cols_to_drop=[]):
    data = pd.read_csv(path, sep)
    for col in cols_to_drop:
        data.drop(col, axis=1, inplace=True)
    return data

def check_outliers(data, show_plot=False, save_img=os.getcwd()+'/outliers.png'):
    
    """
    This functions checks for columns with outlers using the IQR method
    
    It accespts as argmuent a dataset. 
    show_plot can be set to True to output pairplots of outlier columns    
    """
    
    outliers = [] 
    Q1 = data.quantile(0.25)  
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    num_data = data.select_dtypes(include='number')
    result = dict ((((num_data < (Q1 - 1.5 * IQR)) | (num_data > (Q3 + 1.5 * IQR)))==True).any())
    #data[(data[col] >= high)|(data[col] <= low)].index
    index = data[(num_data < Q1 - 1.5 * IQR) | (num_data > Q3 + 1.5 * IQR)].index
    for k,v in result.items():
        if v == True:  
            outliers.append(k)
    if show_plot:
        pair_plot = sns.pairplot(data[outliers]);
        print(f'{result},\n\n Visualization of outlier columns')
        plt.savefig(fname=save_img, format='png')
        return pair_plot
    else:
        return data.loc[index, outliers]
    
    

def treat_outliers(data, type='median_replace'):
    
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
        
        for col in data.columns.tolist():
            if is_numeric_dtype(data[col]):
                median = (data[col].quantile(0.50))
                print(median)
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                high = int(q3 + 1.5 * iqr) 
                low = int(q1 - 1.5 * iqr)
                print(high, low, iqr)
                print(col)
                data[col] = np.where(data[col] > high, median, data[col])
                data[col] = np.where(data[col] > high, median, data[col])        
    
    if type == "quant_floor":
        
        for col in data.columns.tolist():
            if is_numeric_dtype(data[col]):
                q_10 = data[col].quantile(0.5)
                q_90 = data[col].quantile(0.95)
                data[col] =  data[col] = np.where(data[col] < q_10, q_10 , data[col])
                data[col] =  data[col] = np.where(data[col] > q_90, q_90 , data[col])
            
    if type == "trim":
        
        for col in data.columns.tolist():
            low = .05
            high = .95
            quant_df = data.quantile([low, high])
            for name in list(data.columns):
                if is_numeric_dtype(data[name]):
                    data = data[(data[name] >= quant_df.loc[low, name]) 
                        & (data[name] <= quant_df.loc[high, name])]
            
    if type == "log_transform":  
        for col in data.columns.tolist():
            if is_numeric_dtype(data[col]):
                data[col] = data[col].map(lambda i: np.log(i) if i > 0 else 0)
                
    if type == "isf":
        iso = IsolationForest(contamination=0.1)
        yhat = iso.fit_predict(data.select_dtypes(exclude='object'))
        #select all rows that are not outliers
        mask = yhat != -1 
        data = data[mask]
        

    return data 


def scale_data(data,scaler=RobustScaler()):
    
    """
    Specify scaler type, scaler type must have fit_transform as a method
    
    """
    data_scaled = scaler.fit_transform(data)
    return data_scaled
