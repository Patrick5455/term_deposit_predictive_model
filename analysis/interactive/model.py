from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

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
