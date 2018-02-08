import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from sklearn.decomposition import PCA
import concurrent.futures
import csv

"""
Read in train and test as Pandas DataFrames
"""
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


i = 0
def getBin(smile):
    global i
    i += 1
    print(i)
    mol = Chem.MolFromSmiles(smile)
    fprint = Chem.rdmolops.RDKFingerprint(mol)
    return list(map(int, fprint.ToBitString()))

with concurrent.futures.ProcessPoolExecutor() as executor:
    Bits = executor.map(getBin, df_test['smiles'].iloc[1:100000], chunksize=100)

print("Done with feature eng")

with open("FingerPrintFeatures.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(Bits)
