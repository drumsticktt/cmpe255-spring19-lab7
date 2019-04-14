import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
from mlxtend.plotting import plot_decision_regions
   

plot_num = 1
tf = open("KernelResults.txt", "a+")


def create_svm(kernel='linear', degree=8):
    # download data set: https://drive.google.com/file/d/13nw-uRXPY8XIZQxKRNZ3yYlho-CYm_Qt/view
    # info: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

    # load data
    bankdata = pd.read_csv("bill_authentication.csv")  

    # see the data
    bankdata.shape  

    # see head
    bankdata.head()  

    # data processing
    X = bankdata.drop('Class', axis=1)  
    y = bankdata['Class']  

    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  


    from sklearn.svm import SVC  

    # train the SVM
    if kernel == 'linear':
        svclassifier = SVC(kernel='linear', gamma='scale')  

    elif kernel == 'poly':
        svclassifier = SVC(kernel='poly', degree = degree, gamma='scale')  

    elif kernel == 'rbf':
        svclassifier = SVC(kernel='rbf', gamma='scale')  

    elif kernel == 'sigmoid':
        svclassifier = SVC(kernel='sigmoid', gamma='scale')  

    svclassifier.fit(X_train, y_train)  

    # predictions
    y_pred = svclassifier.predict(X_test)  

    global tf

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix 
    tf.write(kernel + " Kernel=================================================================\n")
    np.savetxt(tf, confusion_matrix(y_test,y_pred))
    tf.write(classification_report(y_test,y_pred))
    
    global plot_num

    value = 1.5
    width = 0.75
    plt.subplot(2,2, plot_num)
    plot_decision_regions(X=X[['Variance', 'Skewness', 'Curtosis', 'Entropy']].values, y=y.values, clf=svclassifier,feature_index=[0,1],  
                         filler_feature_values={2: value, 3:value}, filler_feature_ranges={2: width, 3:width}, legend=2)
    plt.xlabel("Variance", size=12)
    plt.ylabel("Skewness", size=12)
    plt.title(kernel + " SVM Decision Boundary")
    plot_num += 1


# Iris dataset  https://archive.ics.uci.edu/ml/datasets/iris4
def import_iris():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # Assign colum names to the dataset
    colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

    # Read dataset to pandas dataframe
    irisdata = pd.read_csv(url, names=colnames) 

    # process
    X = irisdata.drop('Class', axis=1)  
    y = irisdata['Class']  

    # train
    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

def polynomial_kernel():
    create_svm('poly')

def gaussian_kernel():
    create_svm('rbf')

def sigmoid_kernel():
    create_svm('sigmoid')

def test():
    import_iris()
    create_svm()
    polynomial_kernel()
    gaussian_kernel()
    sigmoid_kernel()
    # NOTE: 3-point extra credit for plotting three kernel models.
    plt.savefig("Kernal_Comparison.jpg")
    plt.waitforbuttonpress()
    tf.close()
test()
