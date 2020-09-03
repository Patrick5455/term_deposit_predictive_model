
from imblearn.metrics import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.decomposition import *
from sklearn.base import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc 
import matplotlib.pyplot as plt
import itertools

import pandas as pd

class Metrics ():
    
    def __init__(self, X_train, y_train, X_test, y_test, y_hat=np.array([])):
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_hat =  y_hat
        
    
    def class_report(self, y_hat):
        
        self.y_hat = y_hat
        
        full_report = classification_report_imbalanced(self.y_test, self.y_hat)
        
        print(full_report)
        
    def conf_matrix(self, y_hat):
        self.y_hat = y_hat
        
        conf_matrix = confusion_matrix(self.y_test, self.y_hat)
        
        conf_matrix_df = pd.DataFrame(conf_matrix, columns=['Actual_+ve', 'Actual_-ve'],
                               index=['predicted_+ve', 'predicted_-ve'])
        
        return conf_matrix_df
    
    def balanced_accuracy_score(self, y_hat):
        
        self.y_hat = y_hat
        return  balanced_accuracy_score(self.y_test, self.y_hat)
    
    def balanced_classification_error(self, y_hat):
        self.y_hat = y_hat
        return 1 - balanced_accuracy_score(self.y_test, self.y_hat) 
        
    def specif_sensitiv(self, y_hat):
        
        
        """
        Sensitivity: When the actual value is positive, how often is the prediction correct?
        
        Specificity: When the actual value is negative, how often is the prediction correct?
        """
        
        self.y_hat = y_hat
        conf_matrix = confusion_matrix(self.y_test, self.y_hat)
        
        #vals = self.y_test, self.y_hat
        
        TP = conf_matrix[1, 1]
        TN = conf_matrix[0, 0]
        FP = conf_matrix[0, 1]
        FN = conf_matrix[1, 0]
        
        sensitiv = TP / float(FN + TP)
        specific = TN / (TN + FP)
        
        # this functions was updated to use the provided speciictiy/sensitivity score provided by Imblearn.metrics
        
        # but the manual calculation was retained for record purposes
        
        sensitiv_specific_table = pd.DataFrame([[sensitivity_score(self.y_test, self.y_hat),
                                                 specificity_score(self.y_test, self.y_hat)]],
                                               columns=['sensitivity', 'specificity'])
        
        return sensitiv_specific_table
    
    
    def evaluate_classifier(self, clf, models_eval_scores, clf_name=None): # X_train, y_train, X_test, y_test, ):
    
        self.clf = clf
        self.models_eval_scores = models_eval_scores
        self.clf_name = clf_name
        
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
        if self.clf_name is None:
            if isinstance(self.clf, ImbPipe) or isinstance(self.clf, Pipe):
                self.clf_name = self.clf[-1].__class__.__name__
                print(self.clf_name)
            else:
                self.clf_name = self.clf.__class__.__name__
                print(self.clf_name)
        acc = self.clf.fit(self.X_train, self.y_train).score(self.X_test, self.y_test)
        y_pred = self.clf.predict(self.X_test)
        bal_acc = balanced_accuracy_score(self.y_test, y_pred)
        print(f"acc->{acc}, bal_acc->{bal_acc}")
        clf_score = pd.DataFrame(
            {self.clf_name: [acc, bal_acc]}, 
            index=['Accuracy', 'Balanced accuracy']
        )
        self.models_eval_scores = pd.concat([self.models_eval_scores, clf_score], axis=1).round(decimals=3)
        
        return self.models_eval_scores
    
    def plot_confusion_matrix(self, y_hat,ncols=1,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        
        self.y_hat = y_hat 
        
        fig, ax = plt.subplots()
        
        cm = confusion_matrix(self.y_test, self.y_hat)
        
        classes = self.y_train
        
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
       
        """ 
        print(cm)
        print('')

        ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title(title)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.sca(ax)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd' 
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        
        
    
    def plot_roc(self, y_hat, n_classes=2):
        
        self.y_hat = y_hat
        
       # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(self.y_test[:, i],  self.y_hat[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(self.y_test, self.y_hat)
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        

                # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()


