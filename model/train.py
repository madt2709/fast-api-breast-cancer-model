from httpx import main
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump
import pathlib

def train():
    # load data
    df, y = load_breast_cancer(return_X_y=True, as_frame=True)

    # splitting data
    X_train, X_test, y_train, y_test = train_test_split(
                    df,
                    y,
                    test_size=0.2,
                    random_state=42)

    # standardize features by removing the mean and scaling to unit variance.
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.fit_transform(X_test)

    # train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions1 = model.predict(X_test)

    # work out accuracy
    logreg_acc = accuracy_score(y_test, predictions1)
    print("Accuracy of the Logistic Regression Model is: ", logreg_acc)

    # save model
    dump(model, pathlib.Path('model/breast-cancer-v1.joblib')) 

if __name__ == "__main__":
    train()
