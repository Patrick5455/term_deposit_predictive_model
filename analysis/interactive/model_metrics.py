
from imblearn.metrics import *
from sklearn.metrics import *
import pandas as pd

class metrics ():
    
    def __init__(self, y_test, y_hat):
        pass
        self.y_test = y_test
        self.y_hat =  y_hat
        
    
    def class_report(self):
        
        full_report = classification_report(self.y_test, self.y_hat)
        
        print(full_report)
        
    def conf_matrix(self):
        
        conf_matrix = confusion_matrix(self.y_test, self.y_hat)
        
        conf_matrix_df = pd.DataFrame(conf_matrix, columns=['Actual_+ve', 'Actual_-ve'],
                               index=['predicted_+ve', 'predicted_-ve'])
        
        return conf_matrix_df
    
    def accuracy_score(self):
        return  accuracy_score(self.y_test, self.y_hat)
    
    def classification_error(self):
        
        return 1 - accuracy_score() 
        
    def specif_sensitiv(self):
        
        """
        Sensitivity: When the actual value is positive, how often is the prediction correct?
        
        Specificity: When the actual value is negative, how often is the prediction correct?
        """
        
        conf_matrix = confusion_matrix(self.y_test, self.y_hat)
        
        TP = conf_matrix[1, 1]
        TN = conf_matrix[0, 0]
        FP = conf_matrix[0, 1]
        FN = conf_matrix[1, 0]
        
        sensitivity = TP / float(FN + TP)
        specificity = TN / (TN + FP)
        
        sensitiv_specific_table = pd.DataFrame([[sensitivity, specificity]],
                                               columns=['sensitivity', 'specificity'])
        
        return sensitiv_specific_table
    
    
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
