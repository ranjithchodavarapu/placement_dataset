import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
import sklearn
import numpy as np
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt
import seaborn as sns


data=pd.read_csv('datasets_596958_1073629_Placement_Data_Full_Class.csv')
print(data.head())


print(data.info())
print(data.describe())
print(data.corr())

data=data[['ssc_p','mba_p','degree_p','status','etest_p','hsc_p']]
print(data.head())
print(data.info())
print(data.describe())
print(data.corr())


x=data.drop(['status'],axis=1)
y=data['status']

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
print(x_train,y_train)

cls=RandomForestClassifier()

cls.fit(x_train,y_train)
pre=cls.predict(x_test)
print(pre)
print(y_test)


print(metrics.accuracy_score(y_test,pre))
print(metrics.confusion_matrix(y_test,pre))
print(metrics.classification_report(y_test,pre))
