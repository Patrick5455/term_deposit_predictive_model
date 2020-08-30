from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import numpy as np

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
