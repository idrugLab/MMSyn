import sys
sys.path.append('D:\PANGYU\DDS-DFFDDS_ECFP')
import csv
from smiles2graph import smile_to_graph
import torch
# from dataset import TestbedDataset
from dataset import TestbedDataset
import numpy as np
import random
import pickle
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
from pubchemfp import GetPubChemFPs
from ESPF.gen_fp import drug2espf

def fpn(mol):
    fp = []
    fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
    fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
    fp_pubcfp = GetPubChemFPs(mol)
    fp.extend(fp_maccs)
    fp.extend(fp_phaErGfp)
    fp.extend(fp_pubcfp)
    return fp

def get_drug_espf(drug_smile_dict):
    espf = {}
    for drug, smile in drug_smile_dict.items():
        espf[drug] = drug2espf(smile)
    return(espf)

def get_drug_erg(drug_smile_dict):
    erg = {}
    for drug, smile in drug_smile_dict.items():
        mol = Chem.MolFromSmiles(smile)
        erg[drug] = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
    return erg

def get_drug_mixfp(drug_smile_dict):
    mixfp = {}
    for drug, smile in drug_smile_dict.items():
        mol = Chem.MolFromSmiles(smile)
        mixfp[drug] = fpn(mol)
    return mixfp

def get_drug_pubchem(filename):
    pubchemfp = {}
    with open(filename) as f:
        reader = csv.reader(f)
        reader.__next__()
        for line in reader:
            pubchemfp[line[0]] = list(map(int, line[1:]))
    return pubchemfp

def get_drug_descriptor(filename):
    desc = {}
    with open(filename) as f:
        reader = csv.reader(f)
        reader.__next__()
        for line in reader:
            desc[line[0]] = list(map(float, line[1:]))
    return desc

nbits = 1024
def get_drug_ecfp6(drug_smile_dict):
    fp = {}
    for drug, smile in drug_smile_dict.items():
        mol = Chem.MolFromSmiles(smile)
        fp[drug] = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=nbits)
    return fp

def get_cell_fe(filename, cell_line_dict):
    feature_dict = {}
    cell_feature = torch.load(filename)
    for cell in cell_line_dict:
        feature_dict[cell] = cell_feature[cell_line_dict[cell]]
    return feature_dict

def get_cell_pw(filename):
    pw_dict = {}
    df = pd.read_csv(filename)
    cell_list = list(df.columns)[1:]
    for cell in cell_list:
        pw_dict[cell] = df[cell].tolist()
    return pw_dict
        
def get_all_graph(drug_smile_dict, drug_index_dict):
    smile_graph = {}
    for drug in drug_smile_dict:
            graph = smile_to_graph(drug_smile_dict[drug])
            smile_graph[drug_index_dict[drug]] = graph
    return smile_graph


def get_drug_bert(filename, drug_smile_dict, drug_index_dict):
    bert_dict = pickle.load(open(filename, 'rb'))
    drug_bert = {}
    for drug, smile in drug_smile_dict.items():
        drug_bert[drug_index_dict[drug]] = bert_dict[smile]
    return drug_bert

random.seed(42)
def read_response_data_and_process(filename):
    # load features
    drug_index_dict = dict(zip(pd.read_csv('D:/PANGYU/DDS-DFFDDS_ECFP/ONeil/id_38_drugs.csv')['drug'], pd.read_csv('D:/PANGYU/DDS-DFFDDS_ECFP/ONeil/id_38_drugs.csv')['ID']))
    drug_smile_file = 'D:\PANGYU\DDS-DFFDDS_ECFP\ONeil\smiles_title.csv'
    drug_smile_dict = dict(zip(pd.read_csv(drug_smile_file)['DRUG'], pd.read_csv(drug_smile_file)['SMILE']))
    drug_ecfp6 = get_drug_ecfp6(drug_smile_dict)
    drug_espf = get_drug_espf(drug_smile_dict)
    drug_pubchem = get_drug_pubchem("D:/PANGYU/DDS-DFFDDS_ECFP/ONeil/drug39_881_dim_fp.csv")
    drug_desc = get_drug_descriptor("D:/PANGYU/DDS-DFFDDS_ECFP/ONeil/drug39_269_dim_descriptor.csv")
    cell_index_file = 'D:/PANGYU/DDS-DFFDDS_ECFP/ONeil/cell_30_ge_cn_id.csv'
    cell_line_dict =  dict(zip(pd.read_csv(cell_index_file)['cell_line'], pd.read_csv(cell_index_file)['ID']))
    cell_ge = get_cell_fe('D:\PANGYU\DDS-DFFDDS_ECFP\ONeil\ONeil_30cell_ge_512dim.pt', cell_line_dict)
    cell_cn = get_cell_fe('D:\PANGYU\DDS-DFFDDS_ECFP\ONeil\ONeil_30cell_cn_512dim.pt', cell_line_dict)
    cell_pw = get_cell_pw('D:\PANGYU\DDS-DFFDDS_ECFP\ONeil\ONeil_31_gsva_1329_dim.csv')
    smile_graph = get_all_graph(drug_smile_dict, drug_index_dict)
    drug_bert = get_drug_bert('D:\PANGYU\DDS-DFFDDS_ECFP\ONeil\FG_BERT_embedding_256_oneil_smiles.pkl', drug_smile_dict, drug_index_dict)
    drug_mixfp = get_drug_mixfp(drug_smile_dict)
    

    # read response data
    with open(filename, 'r') as f:
        # f.__next__()
        reader = csv.reader(f)
        reader.__next__()
        reader.__next__()
        data = []
        for line in reader:
            drugA = line[0]
            drugB = line[1]
            cell = line[2]
            synergy = float(line[3])
            data.append((drugA, drugB, cell, synergy))
    random.shuffle(data)

    # match features and labels
    drugA_smile = []
    drugA_ecfp6 = []
    drugB_smile = []
    drugB_ecfp6 = []
    drugA_desc = []
    drugB_desc = []
    drugA_mixfp = []
    drugB_mixfp = []
    drugA_espf = []
    drugB_espf = []
    drugA_pub = []
    drugB_pub = []

    ge = []
    cn = []
    pw = []
    label = []
    for item in data:
        drugA, drugB, cell, synergy = item
        drugA_smile.append(drug_index_dict[drugA])
        drugB_smile.append(drug_index_dict[drugB])

        drugA_ecfp6.append(drug_ecfp6[drugA])
        drugB_ecfp6.append(drug_ecfp6[drugB])

        drugA_espf.append(drug_espf[drugA])
        drugB_espf.append(drug_espf[drugB])

        drugA_desc.append(drug_desc[drugA])
        drugB_desc.append(drug_desc[drugB])

        drugA_mixfp.append(drug_mixfp[drugA])
        drugB_mixfp.append(drug_mixfp[drugB])

        drugA_pub.append(drug_pubchem[drugA])
        drugB_pub.append(drug_pubchem[drugB])

        ge.append(cell_ge[cell].cpu().detach().numpy())
        cn.append(cell_cn[cell].cpu().detach().numpy())
        pw.append(cell_pw[cell])
        if synergy > 10:
            label.append(int(1))
        else:
            label.append(int(0))


    # split data
    drugA_smile = np.asarray(drugA_smile)
    drugB_smile = np.asarray(drugB_smile)

    drugA_espf = np.asarray(drugA_espf)
    drugB_espf = np.asarray(drugB_espf)

    drugA_ecfp6 = np.asarray(drugA_ecfp6)
    drugB_ecfp6 = np.asarray(drugB_ecfp6)

    drugA_desc = np.asarray(drugA_desc)
    drugB_desc = np.asarray(drugB_desc)

    drugA_mixfp = np.asarray(drugA_mixfp)
    drugB_mixfp = np.asarray(drugB_mixfp)

    drugA_pub = np.asarray(drugA_pub)
    drugB_pub = np.asarray(drugB_pub)

    ge = np.asarray(ge)
    cn = np.asarray(cn)
    pw = np.asarray(pw)
    label = np.asarray(label)
   

    for i in range(0,5):
        total_size = drugA_smile.shape[0]
        size_0 = int(total_size * 0.2 * i)
        size_1 = size_0 + int(total_size * 0.1)
        size_2 = int(total_size * 0.2 * (i + 1))
        
        drugAsm_test = drugA_smile[size_0:size_1]
        drugAsm_val = drugA_smile[size_1:size_2]
        drugAsm_train = np.concatenate((drugA_smile[:size_0], drugA_smile[size_2:]), axis=0)

        drugBsm_test = drugB_smile[size_0:size_1]
        drugBsm_val = drugB_smile[size_1:size_2]
        drugBsm_train = np.concatenate((drugB_smile[:size_0], drugB_smile[size_2:]), axis=0)

        drugAespf_test = drugA_espf[size_0:size_1]
        drugAespf_val = drugA_espf[size_1:size_2]
        drugAespf_train = np.concatenate((drugA_espf[:size_0], drugA_espf[size_2:]), axis=0)

        drugBespf_test = drugB_espf[size_0:size_1]
        drugBespf_val = drugB_espf[size_1:size_2]
        drugBespf_train = np.concatenate((drugB_espf[:size_0], drugB_espf[size_2:]), axis=0)

        drugAecfp6_test = drugA_ecfp6[size_0:size_1]
        drugAecfp6_val = drugA_ecfp6[size_1:size_2]
        drugAecfp6_train = np.concatenate((drugA_ecfp6[:size_0], drugA_ecfp6[size_2:]), axis=0)

        drugBecfp6_test = drugB_ecfp6[size_0:size_1]
        drugBecfp6_val = drugB_ecfp6[size_1:size_2]
        drugBecfp6_train = np.concatenate((drugB_ecfp6[:size_0], drugB_ecfp6[size_2:]), axis=0)

        drugAdesc_test = drugA_desc[size_0:size_1]
        drugAdesc_val = drugA_desc[size_1:size_2]
        drugAdesc_train = np.concatenate((drugA_desc[:size_0], drugA_desc[size_2:]), axis=0)

        drugBdesc_test = drugB_desc[size_0:size_1]
        drugBdesc_val = drugB_desc[size_1:size_2]
        drugBdesc_train = np.concatenate((drugB_desc[:size_0], drugB_desc[size_2:]), axis=0)

        drugAmixfp_test = drugA_mixfp[size_0:size_1]
        drugAmixfp_val = drugA_mixfp[size_1:size_2]
        drugAmixfp_train = np.concatenate((drugA_mixfp[:size_0], drugA_mixfp[size_2:]), axis=0)

        drugBmixfp_test = drugB_mixfp[size_0:size_1]
        drugBmixfp_val = drugB_mixfp[size_1:size_2]
        drugBmixfp_train = np.concatenate((drugB_mixfp[:size_0], drugB_mixfp[size_2:]), axis=0)

        drugApub_test = drugA_pub[size_0:size_1]
        drugApub_val = drugA_pub[size_1:size_2]
        drugApub_train = np.concatenate((drugA_pub[:size_0], drugA_pub[size_2:]), axis=0)

        drugBpub_test = drugB_pub[size_0:size_1]
        drugBpub_val = drugB_pub[size_1:size_2]
        drugBpub_train = np.concatenate((drugB_pub[:size_0], drugB_pub[size_2:]), axis=0)

        ge_test = ge[size_0:size_1]
        ge_val = ge[size_1:size_2]
        ge_train = np.concatenate((ge[:size_0], ge[size_2:]), axis=0)

        cn_test = cn[size_0:size_1]
        cn_val = cn[size_1:size_2]
        cn_train = np.concatenate((cn[:size_0], cn[size_2:]), axis=0)

        pw_test = pw[size_0:size_1]
        pw_val = pw[size_1:size_2]
        pw_train = np.concatenate((pw[:size_0], pw[size_2:]), axis=0)


        label_test = label[size_0:size_1]
        label_val = label[size_1:size_2]
        label_train = np.concatenate((label[:size_0], label[size_2:]), axis=0)
        # xds, xespf, xecfp6, xdesc, xmixfp, xge, xcn, y, smile_graph

        TestbedDataset(root='data', dataset='train_setA{num}'.format(num=i), xds=drugAsm_train, xespf= drugAespf_train, xecfp6=drugAecfp6_train,xdesc=drugAdesc_train, xmixfp=drugAmixfp_train,
                       xpub = drugApub_train, xge=ge_train, xcn=cn_train, xpw=pw_train, y=label_train, smile_graph=smile_graph)
        TestbedDataset(root='data', dataset='test_setA{num}'.format(num=i), xds=drugAsm_test, xespf= drugAespf_test, xecfp6=drugAecfp6_test,xdesc=drugAdesc_test, xmixfp=drugAmixfp_test,
                       xpub = drugApub_test,xge=ge_test,xcn=cn_test, xpw=pw_test, y=label_test, smile_graph=smile_graph)
        TestbedDataset(root='data', dataset='val_setA{num}'.format(num=i), xds=drugAsm_val, xespf= drugAespf_val, xecfp6=drugAecfp6_val,xdesc=drugAdesc_val, xmixfp=drugAmixfp_val,
                       xpub = drugApub_val,xge=ge_val, xcn=cn_val,xpw=pw_val, y=label_val, smile_graph=smile_graph)
        
        TestbedDataset(root='data', dataset='train_setB{num}'.format(num=i), xds=drugBsm_train, xespf= drugBespf_train, xecfp6=drugBecfp6_train,xdesc=drugBdesc_train, xmixfp=drugBmixfp_train,
                       xpub = drugBpub_train, xge=ge_train,xcn=cn_train,xpw=pw_train, y=label_train, smile_graph=smile_graph)
        TestbedDataset(root='data', dataset='test_setB{num}'.format(num=i), xds=drugBsm_test, xespf= drugBespf_test, xecfp6=drugBecfp6_test,xdesc=drugBdesc_test, xmixfp=drugBmixfp_test,
                       xpub = drugBpub_test, xge=ge_test,xcn=cn_test, xpw=pw_test, y=label_test, smile_graph=smile_graph)
        TestbedDataset(root='data', dataset='val_setB{num}'.format(num=i), xds=drugBsm_val, xespf= drugBespf_val, xecfp6=drugBecfp6_val,xdesc=drugBdesc_val, xmixfp=drugBmixfp_val,
                       xpub = drugBpub_val,xge=ge_val, xcn=cn_val,xpw=pw_val, y=label_val, smile_graph=smile_graph)

        
    return

if __name__ == '__main__':
    read_response_data_and_process('data\dataset\drug_pair_cell_line_triple.csv')
        

