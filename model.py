import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_size, head_num, dropout, residual=True):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.head_num = head_num
        self.dropout = dropout
        self.use_residual = residual
        self.attention_head_size = embed_size // head_num

        self.W_Q = nn.Parameter(torch.Tensor(embed_size, embed_size))
        self.W_K = nn.Parameter(torch.Tensor(embed_size, embed_size))
        self.W_V = nn.Parameter(torch.Tensor(embed_size, embed_size))

        if self.use_residual:
            self.W_R = nn.Parameter(torch.Tensor(embed_size, embed_size))

        for weight in self.parameters():
            nn.init.xavier_uniform_(weight)
        
    def forward(self, x):
        Query = torch.tensordot(x, self.W_Q, dims=([-1], [0]))
        Key = torch.tensordot(x, self.W_K, dims=([-1], [0]))
        Value = torch.tensordot(x, self.W_V, dims=([-1], [0]))

        Query = torch.stack(torch.split(Query, self.attention_head_size, dim=2))
        Key = torch.stack(torch.split(Key, self.attention_head_size, dim=2))
        Value = torch.stack(torch.split(Value, self.attention_head_size, dim=2))

        inner = torch.matmul(Query, Key.transpose(-2, -1))
        inner = inner / self.attention_head_size ** 0.5

        attn_w = F.softmax(inner, dim=-1)
        attn_w = F.dropout(attn_w, p=self.dropout)

        results = torch.matmul(attn_w, Value)

        results = torch.cat(torch.split(results, 1, ), dim=-1)
        results = torch.squeeze(results, dim=0)  # (bs, fields, D)

        if self.use_residual:
            results = results + torch.tensordot(x, self.W_R, dims=([-1], [0]))
        results = F.relu(results)
        return results
    
class AttMechanism(nn.Module):

    def __init__(self, field_dim, embed_size, head_num, dropout=0.5):
        super(AttMechanism, self).__init__()

        hidden_dim = 1024

        self.multi_head_att = MultiHeadAttention(embed_size=embed_size, # 128
                                                      head_num=head_num, # 8
                                                      dropout=dropout)
        self.trans_nn = nn.Sequential(
            nn.LayerNorm(field_dim * embed_size),
            nn.Linear(field_dim * embed_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, field_dim * embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        b, f, e = x.shape
        multi_head_att_x = self.multi_head_attt(x).reshape(b, f*e)
        m_vec = self.trans_nn(multi_head_att_x)
        m_x = m_vec + x.reshape(b, f*e)
        return m_x
    
class MMSyn(nn.Module):
    def __init__(self, embed_dim=256,  num_features_xd=78, output_dim=256, gat_dim=128,  dropout=0.2):
        super(MMSyn, self).__init__()
        proj_dim = 256
        self.relu = nn.ReLU()
        dropout_rate = 0.2
        self.feature_interact = AttMechanism(field_dim=9, embed_size=proj_dim, head_num=4)
       
        # drug structure embedding
        self.drug1_gat1 = GATConv(num_features_xd, gat_dim, heads=10, dropout=dropout)
        self.drug1_gat2 = GATConv(gat_dim * 10, gat_dim, dropout=dropout)
        self.drug1_fc = nn.Linear(gat_dim, output_dim)

        self.drug2_gat1 = GATConv(num_features_xd, gat_dim, heads=10, dropout=dropout)
        self.drug2_gat2 = GATConv(gat_dim * 10, gat_dim, dropout=dropout)
        self.drug2_fc = nn.Linear(gat_dim, output_dim)


        # drug FCL embedding
        self.projection_espf1 = nn.Sequential(
            nn.Linear(2586, proj_dim),
            nn.ReLU(),
            # nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate)
        )

        self.projection_espf2 = nn.Sequential(
            nn.Linear(2586, proj_dim),
            nn.ReLU(),
            # nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate)
        )


        self.projection_mixfp1 = nn.Sequential(
            nn.Linear(1489, proj_dim),
            nn.ReLU(),
            # nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate)
        )

        self.projection_mixfp2 = nn.Sequential(
            nn.Linear(1489, proj_dim),
            nn.ReLU(),
            # nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate)
        )


        # cell feature embedding
        self.projection_context =  nn.Sequential(
            nn.Linear(512, proj_dim),
            nn.ReLU(),
            # nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate)
        )

        # cell copynumber enbedding
        self.projection_cnv = nn.Sequential(
            nn.Linear(512, proj_dim),  #drugcombdb是112,drugbankddi是86,288
            nn.ReLU(),
            # nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate)
        )

        # cell passway
        self.projection_pw = nn.Sequential(
            nn.Linear(1329, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )

        self.combine_function = nn.Linear(proj_dim*6, proj_dim*3, bias=False)

        self.transform = nn.Sequential(
            nn.LayerNorm(proj_dim*3),
            nn.Linear(proj_dim*3, 1),
        )



    def forward(self, data1, data2):
        # print(data1)
        x1, edge_index1, batch1, context, espf1, ecfp61, desc1, mixfp1, pub1, fgbert1, cnv, pw = data1.x, data1.edge_index, data1.batch, data1.ge, data1.espf, data1.ecfp6, data1.desc, data1.mixfp, data1.pub, data1.fgbert, data1.cn, data1.pw
        fgbert1 = fgbert1.squeeze(1)
        pw = pw.squeeze(1)
        pw = pw[:, None, :]
        # print(fp1.shape)
        # print(fgbert1.shape)
        # fp1 = fp1[:, None, :]
        x2, edge_index2, batch2, espf2, ecfp62, desc2, mixfp2, fgbert2, pub2 = data2.x, data2.edge_index, data2.batch, data2.espf, data2.ecfp6, data2.desc, data2.mixfp, data2.fgbert, data2.pub
        fgbert2 = fgbert2.squeeze(1)
        # fp2 = fp2[:, None, :]
 
        # drug1 structure embedding
        x1 = self.drug1_gat1(x1, edge_index1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x1 = self.drug1_gat2(x1, edge_index1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x1 = gmp(x1, batch1)
        x1 = self.drug1_fc(x1)
        struc_1_vectors = self.relu(x1)

        # drug2 structure embedding
        x2 = self.drug2_gat1(x2, edge_index2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x2 = self.drug2_gat2(x2, edge_index2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x2 = gmp(x2, batch2)
        x2 = self.drug2_fc(x2)
        struc_2_vectors = self.relu(x2)

        contextFeatures = self.projection_context(context)
        cnvFeatures = self.projection_cnv(cnv)
        pwFeatures = self.projection_pw(pw)

        espf1_vectors = self.projection_espf1(espf1)
        espf2_vectors = self.projection_espf2(espf2)

        mixfp1_vectors = self.projection_mixfp1(mixfp1)
        mixfp2_vectors = self.projection_mixfp2(mixfp2)

        all_features = torch.stack([struc_1_vectors, mixfp1_vectors, espf1_vectors, contextFeatures.squeeze(1), cnvFeatures.squeeze(1), pwFeatures.squeeze(1), struc_2_vectors, mixfp2_vectors, espf2_vectors], dim=1)
        all_features = self.feature_interact(all_features) # (b, 8e)
        all_features = all_features.reshape(-1, 9, 256)
        drug1_feature = all_features[:, :3, :].reshape(-1, 3*256)
        drug2_feature = all_features[:, 6:, :].reshape(-1, 3*256)
        cell_feature = all_features[:, 3:6 , :].reshape(-1, 3*256)
        
        combined_drug = self.combine_function(torch.cat([drug1_feature, drug2_feature], dim=1)) # (b, 2e)
        therapy_emb = (combined_drug * cell_feature) # (b, 2e)

        out = self.transform(therapy_emb)
        
        return out, therapy_emb


