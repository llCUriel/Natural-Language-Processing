import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import mord as m
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from prepareData import prepareData_lemmas

def testModel(sampleTexts, y):
    y=np.asarray(y)


    count_vect = CountVectorizer()
    X_counts = count_vect.fit_transform(sampleTexts)
    X=X_counts

    X_train, X_test, y_train, y_test = train_test_split(
                                   X, y, test_size=0.2, random_state=42)
    clf = m.LogisticIT()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print('Accuracy of prediction is', clf.score(X_test, y_test))
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))

if __name__=='__main__':

    dir = '../corpusCriticasCine/'

    tabla = prepareData_lemmas(dir)
    print(tabla)
