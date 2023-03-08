import os
import sys
sys.path.append('/projects/')
import pandas as pd

plate_list = 1086293492#1086292884#1086293492#{"1053597936","1053599503","1053600674", "1086289686", "1086292037", 
              #"1086292389", "1086292884", "1086293133", "1086293492", "1086293911"}

code = "Pred_AdaGN_None_CG_None_1086293492" #"1086293492" #1086292884


plate = "Actives_Plate_A"
model = "AdaGN_None_CG_None"

#Plate A = 1086293492
#Plate B = 1086292884


df = pd.read_csv('/projects/img/GAN_CP/PAPER_3/Palette-Image-to-Image-Diffusion-Models-main/target2_metadata.csv')
df = df[df.PLATE == plate_list]
df = df.rename(columns={"SAMPLEIDDISPLAY":"Metadata_broad_sample", "Well Id":"Metadata_Well"})


df_final = pd.read_csv(f'/projects/img/GAN_CP/PAPER_3/Palette-Image-to-Image-Diffusion-Models-main/only_active_normalization/plateA_all_combined_normalization_featureSelect.csv')

df_final = df_final[(df_final.Metadata_Plate == code) | (df_final.Metadata_Plate == "1086293492")]

print(df_final)

df_master = pd.merge(df, df_final, on=['Metadata_broad_sample', 'Metadata_Well'])

df_master = df_master.drop(["Metadata_concentration","PLATE","CONCENTRATION","InChIKey","pubchem_cid","pert_type","control_type","smiles"], axis=1)

print(df_master)
    
# "Metadata_Plate", "pert_iname", "target"

## Remove samples where target only has one pert (poscon_diverse)
#poscon_diverse = ["BRD-K25412176-001-04-9", "BRD-K64800655-001-11-9", "BRD-K86525559-001-07-8", 
#                  "BRD-K03406345-001-18-7", "BRD-K33882852-003-04-9", "BRD-K38852836-001-04-9",
#                  "BRD-K49350383-001-15-2", "BRD-K41599323-001-03-1", "BRD-K58550667-001-12-9",
#                  "BRD-K37764012-001-06-9", "BRD-K42191735-001-08-7", "BRD-K64890080-001-14-9",
#                  "BRD-K07881437-001-03-8", "BRD-K23363278-001-02-1",
#                  "BRD-K06426971-001-04-9", "BRD-K62949423-001-04-9"]# CDK4  not poscon_diverse
#df_master = df_master[~df_master.Metadata_broad_sample.isin(poscon_diverse)]

print(df_master)

actives = [
"CSK", "OPRL1", "FOXM1", "CATSPER4", "TGM2", "S1PR1", "GHSR", "PTPN2", "HCK","KDR", "TGFBR1",
"HIF1A", "LCK", "HSP90AB1", "LYN", "BRD4", "CDK9", "CSF1R", "FLT3", "PPARD", "CDC25A","CDK2",
"CACNA2D3", "CCND1", "CYP3A4", "CDK7", "PLK1", "TUBB3", "CHEK2", "HSP90AA1", "PIK3CG", "USP1",
"NAMPT", "IMPDH1", "RPL3", "TNF", "PAK1", "AGER", "CTSG", "EZH2", "TUBB", "TUBB4B", "MAPK8",
"BAX", "CHRM2", "PTK2B", "PDPK1", "IGF1R", "PAK4", "ABL1", "FFAR4", "PRKCE", "KRAS", "ICAM1",
"RET", "DYRK1B", "ATM", "ALK", "MAPK14", "SIRT2", "HDAC3", "RPL23A", "AURKB", "OPRM1", "DDR2",
"ITGB2", "NTRK1", "BTK", "HSD11B1", "CACNG1"
]


df_master.to_csv(f'{plate}_annotated_features_{model}_all.csv')

## Select active subset
#df_master_actives = df_master[df_master.target.isin(actives)]
#df_master_actives = df_master_actives.reset_index(drop=True)
#df_master_actives.to_csv(f'{plate}_annotated_features_{model}_actives.csv')
#
## Remove DMSO
#df_master_no_control = df_master[df_master['pert_iname'] != 'DMSO']
#df_master_no_control = df_master_no_control.reset_index(drop=True)
#df_master_no_control.to_csv(f'{plate}_annotated_features_{model}_no_controls.csv')
#df_master_controls = df_master[df_master['pert_iname'] == 'DMSO']
#df_master_controls = df_master_controls.reset_index(drop=True)
#df_master_controls.to_csv(f'{plate}_annotated_features_{model}_controls.csv')


##df_master_no_control_actives.target = pd.Categorical
#
#df_master_no_control = df_master
#
#
## Add Unique pert, plate and target columns and labels
#df_master_no_control.target = pd.Categorical(df_master_no_control.target)
#df_master_no_control.SAMPLEIDDISPLAY = pd.Categorical(df_master_no_control.SAMPLEIDDISPLAY)
#df_master_no_control.PLATE = pd.Categorical(df_master_no_control.PLATE)
#df_master_no_control['Unique_Target'] = df_master_no_control.target.cat.codes
#df_master_no_control['Unique_Pert'] = df_master_no_control.SAMPLEIDDISPLAY.cat.codes
#df_master_no_control['Unique_Plate'] = df_master_no_control.PLATE.cat.codes
##df_master_no_control.to_csv('target2_BOTH_SETS.csv')
##df_master_no_control.to_csv('target2_BOTH_SETS_ACTIVE.csv')
#
#



