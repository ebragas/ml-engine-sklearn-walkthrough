"""Setup pipeline, train model, and pickle pipeline object"""

from sklearn import datasets
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import pickle

# Load Iris dataset
iris = datasets.load_iris()

# Setup pipeline
pipeline = Pipeline([
    ('feature_selection', SelectKBest(chi2, k=2)),
    ('classification', RandomForestClassifier(n_estimators=1000))
])

pipeline.fit(iris.data, iris.target)

# Export classifier to file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)
