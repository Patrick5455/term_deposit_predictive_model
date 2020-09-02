def evaluate_classifier(clf, df_scores, X_train, y_train, X_test, y_test, clf_name=None):
    
    """
    Returns a dataframe of unbalanced and balanced acuuracy score of estimators used
    Run for the first time, pass an empty dataframe of df_scorees 
    and when running on more estimators, pass the previous df_scores dataframe for a 
    single table of evaluation scores
    
    Example given below:
    
                    LogisticRegressionCV
            Accuracy 	        0.908
            Balanced accuracy 	0.842
    
    """
    from imblearn.pipeline import Pipeline as ImbPipe
    from sklearn.pipeline import Pipeline as Pipe
    if clf_name is None:
        if isinstance(clf, ImbPipe) or isinstance(clf, Pipe):
            clf_name = clf[-1].__class__.__name__
        else:
            clf_name = clf.__class__.__name__
    acc = clf.fit(X_train, y_train).score(X_test, y_test)
    y_pred = clf.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    clf_score = pd.DataFrame(
        {clf_name: [acc, bal_acc]},
        index=['Accuracy', 'Balanced accuracy']
    )
    df_scores = pd.concat([df_scores, clf_score], axis=1).round(decimals=3)
    return df_scores