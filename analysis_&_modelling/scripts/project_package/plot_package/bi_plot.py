
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn import *
import os
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def plot_bivariate(data, x=None, y=None, hue=None, 
                  color='r',save=False,
                title='New Chart', chart_type='hist',
                   xlabel='', ylabel='',
                    save_to=os.getcwd(), img_name = " ", 
                   palette={'use':False, "size":1}, log_normalise=False,
                  kind_joint_plot = 'scatter', kind_pair_plot="scatter", figsize=(10,7)):
    
    """
    Make a bivariate plot of any of the selcted types:
    
    1. bar - barchart
    
    2. scatter  - scatter plot
    
    3. cat  - catplot
    
    4. count - countplot
    
    5 joint - jointplot 
    
    6  pair - pairplot
    
    7  corr - corr_plot
    
    When calling joint_plot:
        
        kind_joint_plot is default to `scatter`
        other types include "reg", "reside", "kde", "hex"
        
    When calling pair_plot:
        
        kind_pair_plot is default to `scatter`
        other types include 'reg'
    """
    def plt_tweaks():
        plt.subplots(figsize= figsize)
        plt.title(title, fontsize=18)
        plt.xlabel(xlabel, fontsize=15)
        plt.ylabel(ylabel, fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
    
    
    # define helper functions
    
    def use_palette():
        palettes = []
#        palette_to_use=[]
        if palette['use'] == True:
            palette_to_use = [palettes[i] for i in range(palette['size'])]
            
            return palette_to_use

    def log_norm():
        if log_normalise and y != None:
            y = np.log(y)
        elif log_normalise and y == None:
            data = np.log(data)
            
    def save_image():
        if save:
            if img_name != " ":
                plt.savefig(fname=save_to+"/"+img_name+'.png', format='png')
            else:
                plt.savefig(fname=save_to+f'/{title}.png', format='png')
                
        
    # make plots
    
    if chart_type == "joint":
        log_norm()
        plot = sns.jointplot(x=x, y=y, data=data,
                            height=6, ratio=5, space=0.2, kind=kind_joint_plot)
        
        save_image()
        
    if chart_type == "pair":
       # try:
        log_norm()
        if palette['use'] == True:
            palette_to_use = use_palette()
            plot = sns.pairplot(data, palette=palette_to_use, 
                            kind= kind_pair_plot,height=3, aspect=1, hue=hue)
        else:
             plot = sns.pairplot(data, 
                            kind= kind_pair_plot,height=2.5, aspect=1, hue=hue, )
        save_image()
        
    if chart_type  == "corr":
        plt_tweaks()
        corr_data = data.corr()
        corr_plot = sns.heatmap(corr_data,annot=True, fmt='.2g', center=0) 
        
    return plot
