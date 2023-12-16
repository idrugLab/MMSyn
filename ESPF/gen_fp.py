from subword_nmt.apply_bpe import BPE
import codecs
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
from rdkit.Chem.Fingerprints import FingerprintMols


vocab_path = 'D:/PANGYU/DDS-DFFDDS_ECFP/ESPF/info/codes_drug_chembl_1500.txt'
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
sub_csv = pd.read_csv('D:\PANGYU\DDS-DFFDDS_ECFP\ESPF\info\subword_units_map_drug_chembl_1500.csv')

idx2word_d = sub_csv['index'].values
word2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

     

def drug2espf(x):
    t1 = dbpe.process_line(x).split()
    try:
        i1 = np.asarray([word2idx_d[i] for i in t1])
    except:
        i1 = np.array([0])
    v1 = np.zeros(len(idx2word_d))
    v1[i1] = 1
    return v1

def smiles2morgan(s, radius=2, nBits=1024):
    try:
        mol = Chem.MolFromSmiles(s)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    except:
        print('rdkit not found this smiles for morgan: ' + s + ' convert to all 0 features')
        features = np.zeros((nBits, ))
    return features

def smiles2erg(s):
    try:
        mol = Chem.MolFromSmiles(s)
        features = np.array(GetErGFingerprint(mol))
    except:
        print('rdkit cannot find this SMILES for ErG: ' + s + 'convert to all 0 features')
        features = np.zeros((315,))
    return features

def smiles2rdkit2d(s):    
    try:
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = np.array(generator.process(s)[1:])
        NaNs = np.isnan(features)
        features[NaNs] = 0
    except:
        print('descriptastorus not found this smiles: ' + s + ' convert to all 0 features')
        features = np.zeros((200, ))
    return np.array(features)

def smiles2daylight(s):
	try:
		NumFinger = 2048
		mol = Chem.MolFromSmiles(s)
		bv = FingerprintMols.FingerprintMol(mol)
		temp = tuple(bv.GetOnBits())
		features = np.zeros((NumFinger, ))
		features[np.array(temp)] = 1
	except:
		print('rdkit not found this smiles: ' + s + ' convert to all 0 features')
		features = np.zeros((2048, ))
	return np.array(features)



def encode_drug(df_data, drug_encoding, column_name='SMILES', save_column_name = 'drug_encoding'):
    print("encoding drug...")
    print("unique drugs: " + str(len(df_data[column_name].unique())))
    if drug_encoding == 'Morgan':
        unique = pd.Series(df_data[column_name].unique()).apply(smiles2morgan)
        unique_dict = dict(zip(df_data[column_name].unique(), unique))
        df_data[save_column_name] = [unique_dict[i] for i in df_data[column_name]]
    elif drug_encoding == 'ESPF':
        unique = pd.Series(df_data[column_name].unique()).apply(drug2espf)
        unique_dict = dict(zip(df_data[column_name].unique(), unique))
        df_data[save_column_name] = [unique_dict[i] for i in df_data[column_name]]
    
    return df_data


