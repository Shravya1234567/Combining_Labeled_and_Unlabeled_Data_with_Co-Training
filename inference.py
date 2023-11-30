from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import sys

class MultinomialLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=100, tol=1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.loss = []
        self.weights = None
        self.bias = None
        
    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        self.loss = []
        self.weights = np.zeros((x.shape[1], len(np.unique(y))))
        self.bias = np.zeros(len(np.unique(y)))
        y = self.one_hot_encode(y)
        for i in range(self.max_iter):

            loss = self.cross_entropy_loss(x, y)
            self.loss.append(loss)

            dw, db = self.gradient(x, y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if loss < self.tol:
                break
        return self
    
    def predict(self, x):
        probs = self.softmax(x)
        return np.argmax(probs, axis=1)
    
    def predict_proba(self, x):
        return self.softmax(x)
    
    def softmax(self, x):
        scores = np.dot(x, self.weights) + self.bias
        probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
        return probs
    
    def cross_entropy_loss(self, x, y):
        probs = self.softmax(x)
        loss = -np.sum(y * np.log(probs)) / len(x)
        return loss
    
    def gradient(self, x, y):
        probs = self.softmax(x)
        dw = np.dot(x.T, (probs - y)) / len(x)
        db = np.sum(probs - y) / len(x)
        return dw, db
    
    def one_hot_encode(self, y):
        y = np.eye(len(np.unique(y)))[y]
        return y
    
class MultinomialNaiveBayes:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.prior = None
        self.likelihood = None
        self.classes = None

    def fit(self, X, y):
        y = np.array(y)
        X = np.array(X)
        self.classes = np.unique(y)
        self.prior = np.zeros(len(self.classes))
        self.likelihood = np.zeros((len(self.classes), X.shape[1]))
        for i, c in enumerate(self.classes):
            self.prior[i] = np.sum(y == c) / len(y)
            self.likelihood[i] = (np.sum(X[y == c], axis=0) + self.alpha) / (np.sum(X[y == c]) + self.alpha * X.shape[1])

    def predict(self, X):
        X = np.array(X)
        return self.classes[np.argmax(np.log(self.prior) + X @ np.log(self.likelihood).T, axis=1)]

    def predict_proba(self, X):
        X = np.array(X)
        probs = np.zeros((X.shape[0], len(self.classes)))
        for i, c in enumerate(self.classes):
            probs[:, i] = np.log(self.prior[i]) + X @ np.log(self.likelihood[i]).T
        probs = np.exp(probs)
        probs = probs / np.sum(probs, axis=1).reshape(-1, 1)
        return probs

def accuracy(X_test, y_test, clf_fulltext,clf_inlinks,flag=0):
    X_test_fulltext = []
    X_test_inlinks = []

    for i in range(len(X_test)):
        X_test_fulltext.append(X_test[i]['view1'])
        X_test_inlinks.append(X_test[i]['view2'])

    y_pred_fulltext = clf_fulltext.predict_proba(X_test_fulltext)
    y_pred_inlinks = clf_inlinks.predict_proba(X_test_inlinks)

    y_pred = []
    for i in range(len(y_pred_fulltext)):
        max_prob1 = max(y_pred_fulltext[i])
        index1 = np.argmax(y_pred_fulltext[i])
        max_prob2 = max(y_pred_inlinks[i])
        index2 = np.argmax(y_pred_inlinks[i])
        if max_prob1 > max_prob2:
            y_pred.append(index1)
        else:
            if flag==1:
                y_pred.append(1-index2)
            else:
                y_pred.append(index2)

    # calculating the accuracy
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            count += 1
    
    acc = count/len(y_pred)
    pre = precision_score(y_test, y_pred, average='macro',zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro',zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro',zero_division=0)
    
    return acc,pre,rec,f1

# given clf_fulltext, clf_inlinks, X_test, y_test as arguments, find the accuracy
if __name__ == "__main__":
    import pickle
    args = sys.argv
    # first argument is the path to the pickle file of clf_fulltext, second argument is the path to the pickle file of clf_inlinks
    file_path1 = args[1]
    file_path2 = args[2]
    with open(file_path1, 'rb') as f:
        clf_fulltext = pickle.load(f)
    with open(file_path2, 'rb') as f:
        clf_inlinks = pickle.load(f)

    # second argument is the path to the pickle file of X_test, third argument is the path to the pickle file of y_test
    file_path = args[3]
    with open(file_path, 'rb') as f:
        X_test = pickle.load(f)

    X_test = np.array(X_test)
    print(X_test[0]['view1'].shape)
    file_path = args[4]
    with open(file_path, 'rb') as f:
        y_test = pickle.load(f)
        
    acc,pre,rec,f1 = accuracy(X_test, y_test, clf_fulltext,clf_inlinks)
    print("Accuracy: ", acc)
    print("Precision: ", pre)
    print("Recall: ", rec)
    print("F1: ", f1)