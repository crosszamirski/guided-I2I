#create active master sheets
from numpy import reshape
import seaborn as sns
import pandas as pd  
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
sys.path.append('/projects/')


plate = "SET_B"
split = "test"

df_master = pd.read_csv(f'/projects/img/GAN_CP/PAPER_3/src/target2_{plate}_{split}.csv')



actives = [
"CSK", "OPRL1", "FOXM1", "CATSPER4", "TGM2", "S1PR1", "GHSR", "PTPN2", "HCK","KDR", "TGFBR1",
"HIF1A", "LCK", "HSP90AB1", "LYN", "BRD4", "CDK9", "CSF1R", "FLT3", "PPARD", "CDC25A","CDK2",
"CACNA2D3", "CCND1", "CYP3A4", "CDK7", "PLK1", "TUBB3", "CHEK2", "HSP90AA1", "PIK3CG", "USP1",
"NAMPT", "IMPDH1", "RPL3", "TNF", "PAK1", "AGER", "CTSG", "EZH2", "TUBB", "TUBB4B", "MAPK8",
"BAX", "CHRM2", "PTK2B", "PDPK1", "IGF1R", "PAK4", "ABL1", "FFAR4", "PRKCE", "KRAS", "ICAM1",
"RET", "DYRK1B", "ATM", "ALK", "MAPK14", "SIRT2", "HDAC3", "RPL23A", "AURKB", "OPRM1", "DDR2",
"ITGB2", "NTRK1", "BTK", "HSD11B1", "CACNG1"
]



# Select active subset
df_master_actives = df_master[df_master.target.isin(actives)]
df_master_actives = df_master_actives.reset_index(drop=True)
df_master_actives.to_csv(f'/projects/img/GAN_CP/PAPER_3/src/target2_{plate}_{split}_actives.csv')