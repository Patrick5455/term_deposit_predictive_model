import seaborn as sns
def outlier_vars(data, show_plot=False):
    import seaborn as sns

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
    for k,v in result.items():
        if v == True: 
            outliers.append(k) 
    if show_plot:
        pair_plot = sns.pairplot(data[outliers]);
        print(f'{result},\n\n Visualization of outlier columns')
        return pair_plot
    else:
        return data[outliers]

