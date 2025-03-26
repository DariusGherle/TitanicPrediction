import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignoră avertismentele de convergență
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Încarcă datele
train_data = pd.read_csv('train.csv')
train_data.drop('Cabin', axis=1, inplace=True)
train_data.dropna(subset=['Embarked'], inplace=True)
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
train_data['Sex'] = train_data['Sex'].map({'male':0, 'female':1})
train_data['Embarked'] = train_data['Embarked'].map({'S':0, 'C':1, 'Q':2})

# Selectează trăsături
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_data[features]
y = train_data['Survived']

# Împarte setul
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

# Scalează
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Model
model = SGDClassifier(loss='log_loss', max_iter=1, warm_start=True, learning_rate='constant', eta0=0.01, random_state=1)

# Inițializăm graficele
train_losses = []
val_losses = []

plt.ion()  # Interactive mode ON
fig, ax = plt.subplots()
line1, = ax.plot([], [], label="Train Loss")
line2, = ax.plot([], [], label="Validation Loss")
ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
ax.legend()
ax.set_xlabel("Epoch")
ax.set_ylabel("Log Loss")

# Antrenare și afișare în timp real
for epoch in range(100):
    model.fit(X_train_scaled, y_train)

    train_proba = model.predict_proba(X_train_scaled)
    val_proba = model.predict_proba(X_val_scaled)

    train_loss = log_loss(y_train, train_proba)
    val_loss = log_loss(y_val, val_proba)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Actualizează datele
    line1.set_data(range(epoch+1), train_losses)
    line2.set_data(range(epoch+1), val_losses)

    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.05)  # Pausă scurtă pt update UI

# Oprește modul interactiv la final
plt.ioff()
plt.show()
