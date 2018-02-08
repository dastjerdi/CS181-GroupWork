import pandas as pd
import numpy as np
import math
from rdkit import Chem
from rdkit.Chem import AllChem
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