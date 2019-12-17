"""
Data: Wine Quality Data (data file)
Technique: Supervised, Classification
Algorithm: Multinomial Naive Bayes Classifier
"""

#importing built-in wine dataset
from sklearn import datasets
wine_data = datasets.load_wine()
print("Features:",wine_data.feature_names)
print("Labels:",wine_data.target_names)

#creating train and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(wine_data.data, wine_data.target, test_size=0.3, random_state=0)

#training and testing the model
from sklearn.naive_bayes import GaussianNB
gnb_model = GaussianNB()
gnb_model.fit(x_train,y_train)
y_predicted = gnb_model.predict(x_test)

#evaluating the model
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test,y_predicted))

