from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
from .logistic_regression import LogisticRegression
from .linear_regression import LinearRegression


from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score


def home(request):
    return render(request,'home.html')
def predict(request):
    return render(request,'predict.html')

def result(request):
    #load Dataset
    dataset = pd.read_csv("E:\diabetes.csv")

    # Feature selection
    selected_features = [0, 1, 2, 3, 4, 5, 6, 7]
    X = dataset.iloc[:, selected_features].values
    y = dataset.iloc[:, 8].values

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Normalize the input features
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # define the logistic regression model
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def predict(X, theta):
        z = np.dot(X, theta)
        Y_pred = sigmoid(z)
        return Y_pred

    def cost_function(X, Y, theta):
        m = len(Y)
        h = predict(X, theta)
        cost = (-1 / m) * np.sum(Y * np.log(h) + (1 - Y) * np.log(1 - h))
        return cost

    def gradient_descent(X, Y, theta, alpha, num_iterations):
        m = len(Y)
        J_history = []
        for i in range(num_iterations):
            h = predict(X, theta)
            gradient = np.dot(X.T, (h - Y)) / m
            theta = theta - alpha * gradient
            J_history.append(cost_function(X, Y, theta))
        return theta, J_history

    # train the logistic regression model
    theta = np.zeros(len(selected_features) + 1)
    X_train_selected = X_train[:, selected_features]
    X_train_selected = np.hstack((np.ones((len(X_train_selected), 1)), X_train_selected))
    theta, J_history = gradient_descent(X_train_selected, y_train, theta, alpha=0.1, num_iterations=1000)

    # plot the cost function over the iterations

    plt.plot(J_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    input_data = np.array([val1, val2, val3, val4, val5, val6, val7, val8])
    input_data = input_data.reshape(1, -1)
    input_data = sc_X.transform(input_data)
    pred = classifier.predict(input_data)
# calculate accuracy
    y_pred_test = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)

    #return the prediction to the result page


    result1 = ""

    if pred == [1]:
        result1 = "Congratulations,You have diabetes"
    else:
        result1 = "Oh no You did not have any diabetes"

    return render(request, "predict.html", {"result2": result1})

