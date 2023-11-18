from gettext import install

import learn
import numpy
import pandas as pd
import pickle

import pip

df = pd.read_csv("Dataset.csv")

X = df.drop(["class_level"], axis=1)
y = df.class_level.values

#pip install numpy print-learn

from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler

X, y = make_classification(n_classes=2, weights=[0.9, 0.1], n_samples=1000, random_state=42,n_features=24)
over_sampler = RandomOverSampler(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = over_sampler.fit_resample(X, y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)

model_score = model.score(X_test, y_test)


print("random forest result: ", model_score)
pickle.dump(model, open('model1.pkl', 'wb'))
model = pickle.load(open('model1.pkl', 'rb'))
print("Success loaded")
