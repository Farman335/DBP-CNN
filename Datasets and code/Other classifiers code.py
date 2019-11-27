# Avoiding warning
import warnings

def warn(*args, **kwargs): pass
warnings.warn = warn
# _______________________________


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# _____________________________
import pandas as pd
import numpy as np
import numpy
import matplotlib.pyplot as plt
import warnings
seed = 123
numpy.random.seed(seed)
# Essential Library
#from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, \
    RandomForestClassifier,  \
    AdaBoostClassifier,    \
    GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, \
    confusion_matrix, \
    roc_auc_score, \
    average_precision_score, \
    roc_curve, \
    f1_score, \
    recall_score, \
    matthews_corrcoef, \
    auc,cohen_kappa_score


# Step 01 : Load the dataset :
iRec = 'BiPSSM_14189.csv'

D = pd.read_csv(iRec, header=None)  # Using pandas
# ___________________________________________________________________________


# Step 02 : Divide features (X) and classes (y) :
# ___________________________________________________________________________
X = D.iloc[:, :-1].values
y = D.iloc[:, -1].values

# scikit-learn :
#from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from xgboost import XGBClassifier

#from neupy import algorithms
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
#from neupy.algorithms import GRNN as grnn
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier,  AdaBoostClassifier,    GradientBoostingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier, \
    RandomForestClassifier, \
    AdaBoostClassifier, \
    GradientBoostingClassifier, \
    ExtraTreesClassifier

Names = ['RF']

Classifiers = [

   #lgb.LGBMClassifier(n_estimators=500),
  
  #MLPClassifier(hidden_layer_sizes=500, solver='sgd', alpha=0.01),
  #XGBClassifier(),
  #RandomForestClassifier(n_estimators=30),
  ExtraTreesClassifier(n_estimators=5),
   #grnn(std = x, verbose = False),
   # GaussianNB(), #4
   # BaggingClassifier(), #5

   # AdaBoostClassifier(), #7
   # GradientBoostingClassifier(), #8
   #SVC(probability=True),
   #9
   # LinearDiscriminantAnalysis(), #10
   
    
]


def runClassifiers():
    i=0
    Results = []  # compare algorithms
    #cv = StratifiedKFold(n_splits=8, shuffle=True)
    from sklearn.model_selection import StratifiedKFold,KFold
    cv = KFold(n_splits=5, random_state=seed, shuffle=True)
    for classifier, name in zip(Classifiers, Names):
        accuray = []
        auROC= []
        avePrecision = []
        F1_Score = []
        AUC = []
        MCC = []
        Recall = []
        mean_TPR = 0.0
        mean_FPR = np.linspace(0, 1, 100)
        CM = np.array([
            [0, 0],
            [0, 0],
        ], dtype=int)
        print(classifier.__class__.__name__)
        model = classifier
        #counter=41
        #counter = 51
        yProCounter = 0
        yThreshCounter = 0
        yDecisionfunCounter = 0
        for (train_index, test_index) in cv.split(X, y):
            X_train = X[train_index]
            X_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]
            model.fit(X_train, y_train)
            # Calculate ROC Curve and Area the Curve
            y_artificial = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            y_proba_threshold = (model.predict_proba(X_test)[:, 1] >= 0.50).astype('float64')

            FPR, TPR, _ = roc_curve(y_test, y_proba)
            mean_TPR += np.interp(mean_FPR, FPR, TPR)
            mean_TPR[0] = 0.0
            roc_auc = auc(FPR, TPR)
            CM = confusion_matrix(y_true=y_test, y_pred=y_artificial)
            TN, FP, FN, TP = CM.ravel()
            
            print('Accuracy: {0:.2f}%\n'.format(((TP + TN) / (TP + FP + TN + FN)) * 100.0))
            auROC.append(roc_auc_score(y_test, y_proba))
            accuray.append(accuracy_score(y_pred=y_artificial, y_true=y_test))
            avePrecision.append(average_precision_score(y_test, y_proba)) # auPR
            F1_Score.append(f1_score(y_true=y_test, y_pred=y_artificial))
            MCC.append(matthews_corrcoef(y_true=y_test, y_pred=y_artificial))
            Recall.append(recall_score(y_true=y_test, y_pred=y_artificial))
            AUC.append(roc_auc)
            CM += confusion_matrix(y_pred=y_artificial, y_true=y_test)

            np.savetxt(str(yProCounter) + '-Prob' + '.csv', np.asarray(y_proba.round(3)))
            yProCounter += 1
            # ProbabilityScore = ProbabilityScore + 1
            np.savetxt(str(yThreshCounter) + '-Thresh' + '.csv', np.asarray(y_proba_threshold.round(3)))
            yThreshCounter += 1

        accuray = [_*100.0 for _ in accuray]
        Results.append(accuray)

        TN, FP, FN, TP = CM.ravel()
        print('Accuracy: {0:.2f}%'.format(np.mean(accuray)))
        print('Sensitivity (+): {0:.2f}%'.format( float( (TP) / (TP + FN) )*100.0))

        print('Specificity (-): {0:.2f}%'.format( float( (TN) / (TN + FP) )*100.0))
        print('Precision: {0:.2f}%'.format(float( TP / (TP + FP)*100.0)))
        # print('auROC: {0:.6f}'.format(np.mean(auROC)))
        #print('auROC: {0:.6f}'.format(mean_auc))
        print('AUC: {0:.2f}%'.format( np.mean(AUC)*100.0))
        print('F1-score: {0:.2f}%'.format(np.mean(F1_Score)*100.0))
        print('MCC: {0:.4f}'.format(np.mean(MCC)))
        print('Confusion Matrix:')
        print(CM)
        print("Training_dataset TP:" + str(TP))
        print("Training_dataset FP:" + str(FP))
        print("Training_dataset FN:" + str(FN))
        print("Training_dataset TN:" + str(TN))         
        #print("The predicted score for:" + np.asarray(y_proba))
        print('_______________________________________')

        mean_TPR /= cv.get_n_splits(X, y)
        mean_TPR[-1] = 1.0
        mean_auc = auc(mean_FPR, mean_TPR)
        plt.plot(
            mean_FPR,
            mean_TPR,
            linestyle='-',
            label='{} ({:0.3f})'.format(name, mean_auc), lw=2.0)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Random')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        plt.title('ROC curve', fontweight='bold')
        plt.legend(loc='lower right')
        plt.savefig('ROC_Sg3_CS_PSSM_1056_Org1.png', dpi=300)
        #plt.show()
        
        import pickle
        pkl_filename = "pickle_model_14189.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model, file)        



'''def boxPlot(Results, Names):
    ### Algoritms Comparison ###
    # boxplot algorithm comparison
    fig = plt.figure()
    # fig.suptitle('Classifier Comparison')
    ax = fig.add_subplot(111)
    ax.yaxis.grid(True)
    plt.boxplot(Results, patch_artist=True, vert = True, whis=True, showbox=True)
    ax.set_xticklabels(Names)
    plt.xlabel('\nName of Classifiers')
    plt.ylabel('\nAccuracy (%)')

    plt.savefig('AccuracyBoxPlot.png', dpi=100)
    plt.show()'''
    ### --- ###
'''def auROCplot():
    ### auROC ###
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Random')
    plt.xlim([0.0, 1.00])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    # plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.savefig('auROC.png', dpi=300)
    plt.show()'''
    ### --- ###
if __name__ == '__main__':
    runClassifiers()
   # auROCplot()


