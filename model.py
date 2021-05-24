import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import joblib

df = pd.read_csv("file_new.csv")

Y=df[["Survived"]]

df=df.drop(["Survived"], axis=1)

X_train,X_test, y_train, y_test = train_test_split(df,Y , stratify=Y, test_size=0.25)



clfGB = GradientBoostingClassifier(n_estimators=200, learning_rate=.01, max_depth=1, random_state=0)
clfGB.fit(X_train, y_train)
joblib.dump(clfGB, 'model.pkl')