from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from train_lgbm import *
import lightgbm as lgb
from  torch.utils.data import TensorDataset,DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import math
import time
import random
from sklearn.preprocessing import *
from model_train_util_vectorized import *
import sys
import os
import os.path as osp
import time
import datetime
from itertools import repeat
from typing import Optional
import gc
import pickle
from sklearn.preprocessing import *
from sys import argv


Device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print('using device:', Device)
dataset_name = argv[1]
use_gbm = argv[2]
use_botspotpp = argv[3]

ROOT_DIR = f'{dataset_name}/'


class BotSpotTrans(object):
    def __init__(self,train_filepath,test_filepath):
        
        train_file = train_filepath
        test_file = test_filepath
        print("Loading train & test file...")
        train_df = pd.read_csv(train_file) # ѵ����
        test_df = pd.read_csv(test_file) # ���Լ�
        total_df = pd.concat([train_df, test_df], axis=0)

        print("Graph Generating...")
        self.edge_index = total_df[["combin_index", "device_index", "target"]].astype(int).values
        self.edge_index_train = train_df[["combin_index", "device_index", "target"]].astype(int).values
        self.edge_index_test = test_df[["combin_index", "device_index", "target"]].astype(int).values
        
        
        # stat_columns��ʾ����ͳ����ص�������list
        # category_columns��ʾ����category��������list
        # stat_columns���������ctit��cvr_total, category_columns��һ����label encoder���channel_id

        stat_columns_file = osp.join(ROOT_DIR, "stat_columns.txt")
        category_columns_file = osp.join(ROOT_DIR, "category_columns.txt")
        stat_columns = self.pickle_load(stat_columns_file)
        category_columns = self.pickle_load(category_columns_file)

        # ����������
        feature_columns = stat_columns + category_columns

        # �������ֻ��ctit����normalization
        normalized_columns = [stat_columns[-2]]
        except_normalized_columns = [column for column in stat_columns if column not in normalized_columns]


        # channel-campagin��ص�����ѵ����������, ������ctit��ͳ������channel_id(���һ��)
        combin_feature_columns = except_normalized_columns + [category_columns[0]]
        # device�ڵ���ص�����ѵ���������У�����ctit + ʣ�µ�category��
        device_feature_columns = normalized_columns + category_columns[1:]
        #���е�channel-campaign��
        device_columns = ["device_index"] + device_feature_columns
        # ���е�device����
        combin_columns = ["combin_index"] + combin_feature_columns

        #��������Ǵ�total_df���ֳ�device_df,���������Ҫ�������train_dfҲһ��
        device_df = total_df[device_columns].sort_values(["device_index"])
        device_df.drop_duplicates(subset="device_index", keep="first", inplace=True)

        #ͬ��Ҳ�Ǵ�total_df���ֳ���combin_df,���������Ҫ���е���
        combin_df = total_df[combin_columns].sort_values(["combin_index"])
        combin_df.drop_duplicates(subset="combin_index", keep="first", inplace=True)

        # ��ctit�н��������С��һ��
        norm_data = RobustScaler().fit_transform(device_df.loc[:,normalized_columns[0]].values.reshape((-1,1)))
        device_df.loc[:,normalized_columns[0]] = norm_data.reshape(-1,)

        print("feature matrix generating...")
        self.device_matrix = device_df[device_feature_columns].values
        # �����folat�ľ���ת�����float16�����Ը��������Ҫ����
        self.combin_matrix = combin_df[combin_feature_columns].astype('float16').values

    def pickle_dump(self, data, filename):
        with open(filename, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def pickle_load(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

        
class Self_Attention(nn.Module):
    # a classical self_attention module as depicted in BERT
    def __init__(self, emb_size, num_head):
        super(Self_Attention, self).__init__()
        self.Q = nn.ModuleList([])
        self.K = nn.ModuleList([])
        self.V = nn.ModuleList([])
        output_size = emb_size // num_head
        self.output_size = output_size
        self.num_head = num_head
        self.final_linear = nn.Linear(emb_size, emb_size)

        for i in range(num_head):
            self.Q.append(nn.Linear(emb_size, output_size))
            self.K.append(nn.Linear(emb_size, output_size))
            self.V.append(nn.Linear(emb_size, output_size))

    def calc_attention(self, X, Q, K, V):
        query = Q(X)
        key = K(X)
        value = V(X)
        key_ = key.transpose(2, 1).contiguous()
        attn_weights = torch.softmax(torch.bmm(query, key_) / math.sqrt(self.output_size), dim=-1)
        output = torch.bmm(attn_weights, value)
        return output

    def forward(self, X):
        outputs = []
        for i in range(self.num_head):
            q, k, v = self.Q[i], self.K[i], self.V[i]
            out = self.calc_attention(X, q, k, v)
            outputs.append(out)
        outputs = torch.cat(outputs, dim=-1)
        return self.final_linear(outputs).mean(dim=1)

    

class BotSpot(nn.Module):
    def __init__(self,edge_index_train,edge_index_test,channel_feats,device_feats,device_package = None,use_enhance_botspot = False,use_gbm = False,gbm_model = None):
        super(BotSpot,self).__init__()
        if use_gbm:
            leaf_dim = 20
            assert gbm_model is not None
            self.leaf_emb_models = nn.ModuleList()
            for n in range(gbm_model.n_estimators):
                self.leaf_emb_models.append(nn.Embedding(31, leaf_dim)) # 31 is the max depth of decision tree
        self.gbm_best_model = gbm_model
        self.use_gbm = use_gbm
        self.self_attn_module = Self_Attention(32, 2)
        self.use_enhanced_botspot = use_enhance_botspot
        self.edge_index_train = edge_index_train
        self.edge_index_test = edge_index_test
        self.edge_num_train = edge_index_train.shape[0]
        self.edge_num_test = edge_index_test.shape[0]
        self.channel_feats = torch.from_numpy(channel_feats).float() # feature matrix for channel-campaign nodes
        self.device_feats = torch.from_numpy(device_feats).float()  # package_name already removed in the device feature matrix
        N_chan = channel_feats.shape[1]-1 + 16 # remove one col and add embeeding_size of 16
        N_dev = 16*(device_feats.shape[1]-1)+1 # add one col of ctit, others are embeddings
        self.edge_matrix = self.gen_adj_matrix() # generate adj matrix of shape (N_device,N_channel_campaign)
        self.device_split_val = 1 # the first col is ctit for device
        self.channel_split_val = -1 # the last col is channel id
        self.channel_neibr_cache = {}  # cache device neighbors as a boolean array of channel_campaign nodes for training stage
        self.channel_neibr_test_cache = {}  # cache neighbors as boolean array of channel_campaign nodes for test stage
        self.device_idx_cache = {} # cache device indices for a channel-campaign node
        if use_enhance_botspot:
            assert device_package is not None
            self.device_package = device_package
            self.super_device_neibr_cache = {} # cache channel_campaign indices for each super device node for training set
            self.super_device_neibr_test_cache = {} # cache channel_campaign indices for each super device node for testing set
        
        # precompute channel neighbors and  cache it
        N = self.edge_matrix.shape[1]
        for i in range(N):
            self.channel_neibr_cache[i] = self.edge_matrix[:,i]
#         N_pack = len(device_package)
        # precompute the super device neighbors for training set if botspot++ is used
        if self.use_enhanced_botspot:
            for idx,d in enumerate(device_package):
                if d not in self.super_device_neibr_cache:
                    self.super_device_neibr_cache[d] = set()
                t_ = np.where(self.edge_matrix[idx]==1)[0]
                for v in t_:
                    self.super_device_neibr_cache[d].add(v)

            for d in self.super_device_neibr_cache:
                self.super_device_neibr_cache[d] = list(self.super_device_neibr_cache)
        
        self.set_adj_test_matrix(True) # modify edge_matrix to include test edges
        for i in range(N):
            self.channel_neibr_test_cache[i] = self.edge_matrix[:,i]
        # precompute super device neighbors for test set, cache them in super_device_neibr_test_cache
        if self.use_enhanced_botspot:
            for idx,d in enumerate(device_package):
                if d not in self.super_device_neibr_test_cache:
                    self.super_device_neibr_test_cache[d] = set()
                t_ = np.where(self.edge_matrix[idx]==1)[0]
                for v in t_:
                    self.super_device_neibr_test_cache[d].add(v)

            for d in self.super_device_neibr_test_cache:
                self.super_device_neibr_test_cache[d] = list(self.super_device_neibr_test_cache)            
        self.set_adj_test_matrix(False) # modify the edge matrix to exclude test edges to proceed on training
        
        # initialze embedding matrix
        emb_size = 16
        channel_id_max = int(channel_feats[:, -1].max() + 1)
        temp = np.max(device_feats[:, 1:], axis=0) + 1
        temp = [int(i) for i in temp]
        lang, plat, os, country, carrier, device_brand, plat_os = temp  # bypass install city, be careful
        self.channel_id_emb = nn.Embedding(channel_id_max, emb_size)
        self.carrier_emb = nn.Embedding(carrier, emb_size)
        self.language_emb = nn.Embedding(lang, emb_size)
        self.device_brand_emb = nn.Embedding(device_brand, emb_size)
        self.plat_os_emb = nn.Embedding(plat_os, emb_size)
        self.plat_emb = nn.Embedding(plat, emb_size)
        self.os_emb = nn.Embedding(os, emb_size)
        self.country_emb = nn.Embedding(country, emb_size)
        
        # device modules if there is no super device convolution
        if not self.use_enhanced_botspot:
            self.dev_linear1 = nn.Linear(N_dev,int(0.6*N_dev)) # NOT also used in channel side for convolving device feats
            self.dev_relu1 = nn.ReLU()
            self.dev_dropout1 = nn.Dropout(0.2)
            self.dev_linear2 = nn.Linear(int(0.6*N_dev),int(0.75*0.6*N_dev))
            self.dev_relu2 = nn.ReLU()
        # channel linear and message passing modules
        self.channel_linear1 = nn.Linear(N_chan,int(0.6*N_chan))
        self.channel_msg_pass1 = nn.Linear(N_dev,int(0.6*N_dev))
        fusion_input = int(0.6*N_chan) + int(0.6*N_dev)
        self.fusion_linear1 = nn.Linear(fusion_input,int(0.6*fusion_input))
        self.fusion_relu1 = nn.ReLU()
        self.fusion_dropout1 = nn.Dropout(0.2)
        fusion_output_dim =  int(0.6*fusion_input)
        device_output_dim = int(0.75*0.6*N_dev)
        
        if not self.use_enhanced_botspot:
            # concat modules if no botspot++
            concat_input_dim = fusion_output_dim+device_output_dim if not self.use_gbm else fusion_output_dim+device_output_dim+leaf_dim
            self.concat_linear1 = nn.Linear(concat_input_dim,int(0.6*concat_input_dim))
            self.concat_relu1 = nn.ReLU()
            self.concat_linear2 = nn.Linear(int(0.6*concat_input_dim),int(0.5*0.6*concat_input_dim))
            self.concat_relu2 = nn.ReLU()
            self.concat_linear3 = nn.Linear(int(0.5*0.6*concat_input_dim),1)
        else:
            # device side gnn if botspot++ is used
            self.dev_linear1 = nn.Linear(N_dev,int(0.6*N_dev)) # NOT also used in channel side for convolving device feats
#             self.device_msg_passing = nn.Linear(N_chan,int(0.6*N_chan))
            in_dim = int(0.6*N_dev) + int(0.6*N_chan)
            self.sup_dev_fusion_linear1 = nn.Linear(in_dim,int(0.6*in_dim))
            self.sup_dev_fusion_relu1 = nn.ReLU()
            self.sup_dev_fusion_dropout1 = nn.Dropout(0.2)
            
            # concat layer for botspot++
            sup_dev_fusion_output_dim = int(0.6*in_dim)
            concat_input_dim = fusion_output_dim+sup_dev_fusion_output_dim if not self.use_gbm else fusion_output_dim+sup_dev_fusion_output_dim+ leaf_dim
            self.concat_linear1 = nn.Linear(concat_input_dim,int(0.6*concat_input_dim))
            self.concat_relu1 = nn.ReLU()
            self.concat_linear2 = nn.Linear(int(0.6*concat_input_dim),int(0.5*0.6*concat_input_dim))
            self.concat_relu2 = nn.ReLU()
            self.concat_linear3 = nn.Linear(int(0.5*0.6*concat_input_dim),1)
            
        

    def to_emb(self, arr, *models):
        '''
        :param arr: matrix for holding high-cardinality features, without one-hot encoding
        :param left: channel node if left is True else device node
        :param models: a list of embedding matrices to embed each high-cardinality feature to dense embeddings
        :return: 2-d tensor with dense embeddings for all the high-cardinality features.
        '''

        out_arr = []
#         arr = torch.from_numpy(arr)
        arr = arr.long().to(Device)
        # device node sparse features

        # N = arr.shape[0]
        num_models = len(models)
        for i in range(len(models)):
            # if num_models > 2 and i == 4:  # bypass install city, hardcoded
            #     continueed
            #             print (i,models[i])
            out_arr.append(models[i](arr[:, i]))
        return torch.cat(out_arr, dim=1)

    def concat_device_feats(self, dev_feats):  # NEED TO MODIFY IT
        '''
        this method invokes to_emb to embed device categorical features into dense embeddings
        :param dev_feats: normalized device features
        :param more_dev_feats: feature matrix with high-cardinality features
        :return: feature matrix with dense embeddings
        '''
        dev_feats = dev_feats.to(Device)
        cat_dev_feats = dev_feats[:, self.device_split_val:]
        emb_tensor = self.to_emb(cat_dev_feats, self.language_emb,
                                 self.plat_emb, self.os_emb, self.country_emb, self.carrier_emb,
                                 self.device_brand_emb, self.plat_os_emb)

        dev_emb_feats = torch.cat((dev_feats[:, :self.device_split_val], emb_tensor), dim=1).float().to(Device)
        return dev_emb_feats

    def concat_channel_feats(self, chan_feats):
        '''
        this method invokes to_emb to embed channel_campaign node's categorical feature into dense embeddings
        similar to concat_device_feats, to add dense embeddings
        '''
        chan_feats = chan_feats.to(Device)
        emb_tensor = self.to_emb(chan_feats[:, self.channel_split_val:], self.channel_id_emb)
#         print (chan_feats[:, :self.channel_split_val].shape)
#         print (emb_tensor.shape)
        return torch.cat((chan_feats[:, :self.channel_split_val], emb_tensor), dim=1).float().to(Device)

    def gen_adj_matrix(self):
        """
        this method build ajdacency matrix of shape (N_device,N_channel).
        adj_matrix[j,i] = True indicates device node j connects to channel-campaign node i
        """
        e = np.vstack((self.edge_index_train,self.edge_index_test))
        N_dev = np.max(e[:,1])+1
        N_channel = np.max(e[:,0])+1
        adj_matrix = np.zeros((N_dev,N_channel),dtype = bool)
        for i,j,_ in self.edge_index_train:
            adj_matrix[j,i] = True
        return adj_matrix
    
    def set_adj_test_matrix(self,set_test = True):
        """
        this method modifies edge_matrix
        when set_test = True, edge_matrix includes test edges in it, otherwise does not
        """
        count_not_in_train = 0
        for i,j,_ in self.edge_index_test:
            if set_test:
                self.edge_matrix[j,i] = True
            else:
                self.edge_matrix[j,i] = False
        return
    
    def sample_minibatch(self,channel_vertices,device_vertices,train_stage = True):
        """
        input:
        for a minitach of edges, channel_vertices is edge[:,0], device_vertices is edge[:,1]
        this method takes a minibatch of edges outputs:
        1) features for channel_campaign nodes
        2) features for device nodes
        3) neighboring device features for each channel_campaign node
        4) neighboring channel_campaign features for each super device node
        """
        #set number of neighbors for channel and device for different stages
        num_neibr = 50 if train_stage else 800
        sup_dev_num_neibr = 20 if train_stage else 50
        edge_matrix = self.edge_matrix

        # channel_vertices and device_vertices must be numpy array
        # original features
        neibr_feats_tensor = []
        sup_dev_neibr_feats_tensor = []
        minibatch_channel_feats = self.concat_channel_feats(self.channel_feats[channel_vertices]).to(Device)  # shape of (minibatch,feats_num_channel)
        minibatch_device_feats = self.concat_device_feats(self.device_feats[device_vertices]).to(Device) # shape of (minibatch,feats_num_device)
        channel_vertices = channel_vertices.cpu().numpy()

        # for each channel vertices, get neighbor devices and its features
        for i in channel_vertices:
            if train_stage:
                neibr_feats = self.adj_indice_to_feat_mat(self.channel_neibr_cache[i],num_neibr,i)
                neibr_feats_tensor.append(neibr_feats)
            else:
                neibr_feats = self.adj_indice_to_feat_mat(self.channel_neibr_test_cache[i],num_neibr,i)
                neibr_feats_tensor.append(neibr_feats)
        neibr_feats_tensor = torch.cat(neibr_feats_tensor,dim = 0).to(Device) # neibr_feats_tensor would be a tensor of shape: (Minibatch,num_neibr,feats_num_device)

        # if use botspot++, for each device node, retrieve its super device index and get its neighboring channel_campaign node features
        if self.use_enhanced_botspot:
            for i in device_vertices:
                sup_dev_neibr_feats = self.adj_indice_to_feat_mat_super_device(sup_dev_num_neibr,i,train_stage)
                sup_dev_neibr_feats_tensor.append(sup_dev_neibr_feats)
            sup_dev_neibr_feats_tensor = torch.cat(sup_dev_neibr_feats_tensor,dim = 0).to(Device) 
            return minibatch_channel_feats,minibatch_device_feats,neibr_feats_tensor,sup_dev_neibr_feats_tensor
        
        return minibatch_channel_feats,minibatch_device_feats,neibr_feats_tensor,-1
    
    def adj_indice_to_feat_mat(self,col,neibr_num,indice):
        """
        input:
        indice: indice of channel_campaign node
        neibr_num: num of sample size
        col: a boolean array of a col given channel_campaign node of $indice
        return:
        neighboring device features
        """
        if indice in self.device_idx_cache:
            ind = self.device_idx_cache[indice]
        else:
            ind = np.where(col==True)[0]
            self.device_idx_cache[indice] = ind
        if len(ind)>neibr_num:
            np.random.shuffle(ind)
            ind_subset = ind[:neibr_num]
        else:
            ind_subset = np.random.choice(ind,size = neibr_num,replace = True)

        dev_feats = self.concat_device_feats(self.device_feats[ind_subset].to(Device))
        return dev_feats.unsqueeze(0)
    
    def adj_indice_to_feat_mat_super_device(self,neibr_num,device_idx,train_stage = True):
        """
        input:
        device_idx: index of device node
        neibr_num: num of sample size
        return:
        neighboring channel_campaign features
        """
        i = self.device_package[device_idx]
        if train_stage:
            sup_dev_neibr = list(self.super_device_neibr_cache[i])
        else:
            sup_dev_neibr = list(self.super_device_neibr_test_cache[i])
        if len(sup_dev_neibr)>neibr_num:
            random.shuffle(sup_dev_neibr)
            c = self.concat_channel_feats(self.channel_feats[sup_dev_neibr[:neibr_num]].to(Device))
        else:
            sup_dev_neibr = np.random.choice(np.asarray(sup_dev_neibr),size = neibr_num,replace = True)
            c = self.concat_channel_feats(self.channel_feats[sup_dev_neibr].to(Device))
        return c.unsqueeze(0)
 

    def get_leaf_from_light_gbm(self, left_vertices, right_vertices, use_self_attn=False):
        # get leaf indices from gbm model and embed into dense matrix
        output_leaf_emb = []
        chan_data = self.channel_feats[left_vertices]
        dev_data = self.device_feats[right_vertices]
        try:
            edge_data = np.hstack((chan_data, dev_data))
        except:
            edge_data = torch.cat((chan_data, dev_data),
                                  dim=1)  # edge feature is the concatenation of channel_node and device_node
            edge_data = edge_data.cpu().numpy()
        # N = len(left_vertices)
        if len(edge_data.shape)==1:
            edge_data = edge_data.reshape((1,-1))
        pred_leaf = self.gbm_best_model.predict_proba(edge_data, pred_leaf=True)
        pred_leaf = torch.from_numpy(pred_leaf).long().to(Device)

        for i in range(pred_leaf.shape[1]):
            # print (self.leaf_emb_models[i](pred_leaf[:, i]).shape)
            output_leaf_emb.append(self.leaf_emb_models[i](pred_leaf[:, i]).unsqueeze(1))
            # ret = torch.cat(output_leaf_emb, dim=1).to(Device)  # leaf node concatenation
        if not use_self_attn:
            ret = torch.cat(output_leaf_emb, dim=1).mean(axis=1).to(Device)  # leaf node mean pooling
            return ret
        else:
            ret = torch.cat(output_leaf_emb, dim=1).to(Device)
            out = self.self_attn_module(ret)
            return out
    
    def forward(self,edges,train_stage = True):
        minibatch_channel_feats,minibatch_device_feats,neibr_feats_tensor,sup_dev_neibr_feats_tensor = self.sample_minibatch(edges[:,0],edges[:,1],train_stage)
        # forward device feats
        if not self.use_enhanced_botspot:
            device_out = self.dev_relu2(self.dev_linear2(self.dev_dropout1(self.dev_relu1(self.dev_linear1(minibatch_device_feats)))))
            channel_conv = self.channel_linear1(minibatch_channel_feats)
            dev_conv = self.dev_linear1(neibr_feats_tensor).mean(dim=1) # share dev_linear1
            fuse_conv = self.fusion_dropout1(self.fusion_relu1(self.fusion_linear1(torch.cat((channel_conv,dev_conv),dim=1))))
            if not self.use_gbm:
                h = self.concat_linear3(self.concat_relu2(self.concat_linear2(self.concat_relu1(self.concat_linear1(torch.cat((fuse_conv,device_out),dim=1))))))
            else:
                leaf_out = self.get_leaf_from_light_gbm(edges[:,0],edges[:,1])
                h = self.concat_linear3(self.concat_relu2(self.concat_linear2(self.concat_relu1(self.concat_linear1(torch.cat((fuse_conv,device_out,leaf_out),dim=1))))))
            return torch.sigmoid(h)
        else:
            channel_conv = self.channel_linear1(minibatch_channel_feats)
            dev_conv = self.dev_linear1(neibr_feats_tensor).mean(dim=1) # share dev_linear1
            fuse_conv = self.fusion_dropout1(self.fusion_relu1(self.fusion_linear1(torch.cat((channel_conv,dev_conv),dim=1))))
            # device side conv:
            sup_dev_conv = self.dev_linear1(minibatch_device_feats)
            sup_channel_conv = self.channel_linear1(sup_dev_neibr_feats_tensor).mean(dim=1)
            sup_fuse_conv = self.sup_dev_fusion_dropout1(self.sup_dev_fusion_relu1(self.sup_dev_fusion_linear1(torch.cat((sup_channel_conv,sup_dev_conv),dim=1)))) 
            if not self.use_gbm:
                h = self.concat_linear3(self.concat_relu2(self.concat_linear2(self.concat_relu1(self.concat_linear1(torch.cat((fuse_conv,sup_fuse_conv),dim=1))))))
            else:
                leaf_out = self.get_leaf_from_light_gbm(edges[:,0],edges[:,1])
                h = self.concat_linear3(self.concat_relu2(self.concat_linear2(self.concat_relu1(self.concat_linear1(torch.cat((fuse_conv,sup_fuse_conv,leaf_out),dim=1))))))
            return torch.sigmoid(h)

if __name__ == '__main__':
    bot_preprocess = BotSpotTrans(f'{dataset_name}/train.csv', f'{dataset_name}/test.csv')
    edge_index_train = bot_preprocess.edge_index_train
    edge_index_test = bot_preprocess.edge_index_test
    chan_feats = bot_preprocess.combin_matrix
    device_feats = bot_preprocess.device_matrix
    device_package = list(device_feats[:, -1])
    device_feats = device_feats[:, :-1]
    if use_gbm:
        gbm_model = make_dataset(chan_feats,device_feats,-1,1,edge_index_train,edge_index_test)
    else:
        gbm_model  = None
    botspot_model = BotSpot(edge_index_train, edge_index_test, chan_feats, device_feats,device_package = device_package,use_enhance_botspot = use_botspotpp,use_gbm = use_gbm,gbm_model = gbm_model)

    tr_dset = TensorDataset(torch.from_numpy(edge_index_train))
    tr_dloader = DataLoader(tr_dset, batch_size=500, shuffle=True)
    test_dset = TensorDataset(torch.from_numpy(edge_index_test))
    test_dloader = DataLoader(test_dset, batch_size=500, shuffle=False)
    optimizer = optim.Adam(botspot_model.parameters(), lr=2e-4, weight_decay=3e-6)
    # model_states = torch.load('model_results_1014/botspotpp/model_checkpoints_4.pt')
    # botspot_model.load_state_dict(model_states)

    botspot_model.to(Device)
    try:
        os.mkdir('model_results')
    except:
        pass
    _ = train(botspot_model, tr_dloader, test_dloader, optimizer, 10, save_name='model_results')


