import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import math
from rdkit import Chem
from rdkit.Chem import AllChem
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

###
### Read in train and test as Pandas DataFrames
###
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

#store gap values
Y_train = df_train.gap.values
#row where testing examples start
test_idx = df_train.shape[0]
#delete 'Id' column
df_test = df_test.drop(['Id'], axis=1)
#delete 'gap' column
df_train = df_train.drop(['gap'], axis=1)

NA_train = []
NA_test = []
GNHA_train = []
GNHA_test = []
NB_train = []
NB_test = []

for smile in df_train['smiles']:
    mol = Chem.MolFromSmiles(smile)
    NA_train.append(mol.GetNumAtoms())
    GNHA_train.append(mol.GetNumHeavyAtoms())
    NB_train.append(mol.GetNumBonds())


for smile in df_test['smiles']:
    mol = Chem.MolFromSmiles(smile)
    NA_test.append(mol.GetNumAtoms())
    GNHA_test.append(mol.GetNumHeavyAtoms())
    NB_test.append(mol.GetNumBonds())

df_train['NumAtms'] = NA_train
df_train['HvyAtms'] = GNHA_train
df_train['NumBnds'] = NB_train
df_test['NumAtms'] = NA_test
df_test['HvyAtms'] = GNHA_test
df_test['NumBnds'] = NB_test

train_smiles = df_train['smiles']
test_smiles = df_train['smiles']

df_train = df_train.drop(['smiles'], axis=1)
df_test = df_test.drop(['smiles'], axis=1)

print("features created")

# x_train, x_test, y_train, y_test = train_test_split(df_train, Y_train, test_size = .33)

###
### Testing only on train to find best fit
###

# # Lasso Regression
# Lasso = LassoCV()
# Lasso.fit(x_train, y_train)
# Lasso_pred = Lasso.predict(x_test)
# Lasso_error = mean_squared_error(y_test, Lasso_pred)
#
# # Ridge Regression
# Ridge = RidgeCV()
# Ridge.fit(x_train, y_train)
# Ridge_pred = Ridge.predict(x_test)
# Ridge_error = mean_squared_error(y_test, Ridge_pred)
#
# # ElasticNet regression
# l1_rtio = Lasso_error / (Lasso_error + Ridge_error)
# EN = ElasticNetCV(l1_ratio = [l1_rtio, .1, .9, .95, .99, 1])
# EN.fit(x_train, y_train)
# EN_pred = EN.predict(x_test)
# EN_error = mean_squared_error(y_test, EN_pred)
#
# # AdaBoost regression
# Ada = AdaBoostRegressor(DecisionTreeRegressor(), learning_rate=0.05)
# params = {'base_estimator__max_depth':list(range(1,6))}
# ada_cv = GridSearchCV(Ada, params, cv = 5)
# ada_cv.fit(x_train, y_train)
# ada_pred = ada_cv.predict(x_test)
#
# AB_error = mean_squared_error(y_test, ada_pred)
#
# # Linear Regression
# LR = LinearRegression()
# LR.fit(x_train, y_train)
# LR_pred = LR.predict(x_test)
#
# LR_error = mean_squared_error(y_test, LR_pred)

###
### Regressing all data to predict test
###

Ada = AdaBoostRegressor(DecisionTreeRegressor(), learning_rate=0.05)
params = {'base_estimator__max_depth':list(range(1,6))}
ada_cv = GridSearchCV(Ada, params, cv = 5)
ada_cv.fit(df_train, Y_train)
Ada_final_pred = ada_cv.predict(df_test)

print("Regression Completed!")

def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")

write_to_file("AdaBoostRegressor.csv", Ada_final_pred)
