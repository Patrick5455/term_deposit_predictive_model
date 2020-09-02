
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import os

def plot_univariate (data, x=None, y=None, color='r',save=False,
                title='New Chart', chart_type='hist', xlabel='', ylabel='',
                    save_to=os.getcwd(), log_normalise=False):
    
    
    """
    Make a univariate plot of any of these selcted types:
    
    1. bar - barchart
    
    2. hist - Histogram
    
    3. pie - Piechart
    
    4. count - Countplot
    
    
    """
    
    plt.subplots(figsize=(10,7))
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    
    if chart_type == 'hist':
        if log_normalise:
            data = np.log(data)
        plot = sns.distplot(a=data, color=color)
        if save:
            plt.savefig(fname=save_to+f'/{title}.png', format='png')
        
    return plot
