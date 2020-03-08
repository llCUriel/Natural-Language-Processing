from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from session import readMessages


def build_and_evaluate_sklearn(sampleTexts, y):

    '''Build vector of token counts'''
    count_vect = CountVectorizer()
    X_counts = count_vect.fit_transform(sampleTexts)
    # list of texts, each text is a string
    X = X_counts

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    from sklearn import metrics
    from sklearn.metrics import confusion_matrix
    print('Accuracy of prediction is', clf.score(X_test, y_test))
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))


if __name__ == '__main__':
    sampleTexts, y = readMessages()
    print(len(sampleTexts), "messages in corpus")
    print(y.count(0), " spam messages in corpus")
    print(y.count(1), " ham messages in corpus")
    build_and_evaluate_sklearn(sampleTexts, y)
