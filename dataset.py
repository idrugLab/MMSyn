from typing import List, Tuple, Union
import torch
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset
import os
import pandas as pd
import pickle


def get_drug_bert(filename, drug_smile_dict, drug_index_dict):
    bert_dict = pickle.load(open(filename, 'rb'))
    drug_bert = {}
    for drug, smile in drug_smile_dict.items():
        drug_bert[drug_index_dict[drug]] = bert_dict[smile]
    return drug_bert
drug_index_file = 'D:\PANGYU\DDS-DFFDDS_ECFP\ONeil\id_38_drugs.csv'
drug_index_dict = dict(zip(pd.read_csv(drug_index_file)['drug'], pd.read_csv(drug_index_file)['ID']))
drug_smile_file = 'D:\PANGYU\DDS-DFFDDS_ECFP\ONeil\smiles_title.csv'
drug_smile_dict = dict(zip(pd.read_csv(drug_smile_file)['DRUG'], pd.read_csv(drug_smile_file)['SMILE']))

# drug_bert = get_drug_bert('D:\PANGYU\DDS-DFFDDS_ECFP\ONeil\FG_BERT_embedding_256_oneil_smiles.pkl', drug_smile_dict, drug_index_dict)

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='set', xds=None, xespf=None, xecfp6=None, xdesc=None, xmixfp=None, xpub=None,
                 xge=None, xcn=None, xpw=None, y=None, transform=None, pre_transform=None, smile_graph=None):
        # root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xds, xespf, xecfp6, xdesc, xmixfp, xpub, xge, xcn, xpw, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + 'pt']
    
    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)


    def process(self, xds, xespf, xecfp6, xdesc, xmixfp, xpub, xge, xcn, xpw, y, smile_graph):
        assert (len(xds) == len(xespf) and len(xespf) == len(xecfp6) and len(xecfp6) == len(xdesc) and len(xdesc) == len(xmixfp) and len(xmixfp) == len(xpub) and len(xpub) == len(xge) and len(xge) == len(xcn), len(xcn) == len(xpw) and len(xpw) == len(y))
        data_list = []
        data_len = len(xds)
        for i in range(data_len):
            smiles = xds[i]
            # bert = drug_bert[smiles]
            espf = xespf[i]
            ecfp6 = xecfp6[i]
            desc = xdesc[i]
            mixfp = xmixfp[i]
            pub = xpub[i]
            ge = xge[i]
            cn = xcn[i]
            pw = xpw[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms
            processedData = DATA.Data(x=torch.Tensor(features),
                                      edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                      y=torch.FloatTensor([labels]))
            processedData.espf = torch.FloatTensor([espf])
            processedData.ecfp6 = torch.FloatTensor([ecfp6])
            processedData.desc = torch.FloatTensor([desc])
            processedData.mixfp = torch.FloatTensor([mixfp])
            processedData.pub = torch.FloatTensor([pub])
            # processedData.fgbert = torch.FloatTensor([bert])
            processedData.ge = torch.FloatTensor([ge])
            processedData.cn = torch.FloatTensor([cn])
            processedData.pw = torch.FloatTensor([pw])

            processedData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(processedData)
       
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for  data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data
        torch.save((data, slices), self.processed_paths[0])
    

