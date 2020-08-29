import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def outlier_vars(data, show_plot=False, save_img=os.getcwd()+'/outliers.png'):
    
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
    result = dict ((((num_data < (Q1 - 1.5 * IQR)) | (num_data > (Q3 + 1.5
                  *IQR)))==True).any())
    for k,v in result.items():
        if v == True:  
            outliers.append(k)
    if show_plot:
        pair_plot = sns.pairplot(data[outliers]);
        print(f'{result},\n\n Visualization of outlier columns')
        plt.savefig(fname=save_img, format='png')
        return pair_plot
    else:
        return data[outliers]

#_________________________________________________________________________________
      
def outlier_treatment(data, col_list, type='median_replacement'):
    
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
        
        for col in col_list:
            median = data[col].quantile(0.50)
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            high = q3 + 1.5 * iqr
            low = q1 - 1.5 * iqr
            #print(q3 + 1.5 * iqr)
            data = np.where(data[col] > high, median, data[col])
            data = np.where(data[col] < low, median, data[col])        
    
    if type == "quant_floor":
        
        for col in col_list:
            q_10 = data[col].quantile(0.10)
            q_90 = data[col].quantile(0.90)
            data =  data[col] = np.where(data[col] < q_10, q_10 , data[col])
            data =  data[col] = np.where(data[col] > q_90, q_90 , data[col])
            
    if type == "trim":
        
        for col in col_list:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            high = (q3 + 1.5) * iqr 
            low = (q1 - 1.5) * iqr
            index = data[(data[col] >= high)|(data[col] <= low)].index
            print(col,'\n', index)
            data.drop(index, inplace=True)
            
    if type == "log_transform":
        for col in outlier_cols:
            data = data[col].map(lambda i: np.log(i) if i > 0 else 0)
      
    return data
        
        
