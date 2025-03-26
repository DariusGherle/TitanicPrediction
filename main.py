import pandas as pd
import matplotlib.pyplot as plt
import pydotplus
from IPython.display import Image
import os
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from yellowbrick.model_selection import LearningCurve
from sklearn.tree import plot_tree

#Load training data
train_data = pd.read_csv('train.csv')

# Drop cabin because it is mostly empty
train_data.drop('Cabin', axis=1, inplace=True)

#dro rows with missing embarked
train_data.dropna(subset=['Embarked'], inplace = True)

#Fill missing values with median
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())

#Convert Sex and embarked to numeric
train_data['Sex'] = train_data['Sex'].map({'male':0, 'female':1})
train_data['Embarked'] = train_data['Embarked'].map({'S':0, 'C':1, 'Q':2})

#choosing features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X = train_data[features]
y = train_data['Survived']

#splitting the data 80% trianing and 20% test but with no randomization of the test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

#initialize the scaler
scaler = StandardScaler()

#fit the scaler on the training data and transform both training and validation sets
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

#Train model Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=2, max_features='sqrt', random_state=1)
model.fit(X_train, y_train)

#evaluate the training set
predictions = model.predict(X_val)
accuracy = accuracy_score(y_val, predictions)

print(f"Accuracy for Random Forest: {accuracy * 100:.2f}%")


#Train model Logistic Regression
log_model = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_preds = log_model.predict(X_val_scaled)

log_accuracy = accuracy_score(y_val, log_preds)
print(f"Logistic Regression accuracy: {log_accuracy * 100:.2f}%")



# Train model SVM
svm_model = SVC(C=1.0, kernel='linear', gamma='scale')
svm_model.fit(X_train_scaled, y_train)
svm_preds = svm_model.predict(X_val_scaled)

svm_accuracy = accuracy_score(y_val, svm_preds)
print(f"SVM accuracy: {svm_accuracy * 100:.2f}%")



#Train model K-nearest neighbours (KNN)
knn_model = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn_model.fit(X_train_scaled, y_train)
knn_preds = knn_model.predict(X_val_scaled)

knn_accuracy = accuracy_score(y_val, knn_preds)
print(f"KNN accuracy: {knn_accuracy * 100:.2f}%")

''' Way to find the best parameters for SVM
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)'''