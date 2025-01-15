# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np
from typing import Tuple, Dict

import IPython
e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

'''
temporal distance network from NoMAD (ICRA2024)
'''
class DenseNetwork(nn.Module):
    def __init__(self, embedding_dim, channel_dim):
        super(DenseNetwork, self).__init__()
        self.channel_dim = channel_dim
        self.embedding_dim = embedding_dim
        self.network = nn.Sequential(
            nn.Conv1d(self.channel_dim, self.channel_dim // 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.channel_dim // 4, 1, 3, padding=1),
            nn.Linear(self.embedding_dim, self.embedding_dim//4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//4, self.embedding_dim//16),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//16, 1)
        )
    
    def forward(self, x):
        # x = x.reshape((-1, self.embedding_dim))
        output = self.network(x)
        return output.squeeze(1)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, node_feat_dim, edge_feat_dim, image_shape: Tuple, num_nodes, token_per_node: int):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        
        
        node_feat_dim, edge_feat_dim, image_shape: Tuple are for arti project shlim & ksshin
        
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        '''
        arti mode
        '''
        # self.node_head = nn.Linear(node_feat_dim, hidden_dim)
        # self.edge_head = nn.Linear(edge_feat_dim, hidden_dim)
        H, W = image_shape[:2]
        # self.mask_head = nn.Linear(H*W, hidden_dim)
       
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(state_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.num_nodes = num_nodes
        
        
        self.edge_feat_dim = edge_feat_dim
        
        # arti proj for encoder
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_proj = nn.Linear(num_nodes*(edge_feat_dim-1), 2*hidden_dim)
        
         
        '''
        tokenize layer for cvae
        '''
        self.cvae_arti_conv1 = nn.Conv1d(num_nodes, 3*num_nodes, 3, padding=1)
        self.cvae_batchnorm1 = nn.BatchNorm1d(3*num_nodes)
        self.cvae_relu1 = nn.ReLU()
        self.cvae_dropout1 = nn.Dropout(0.5)
        
        self.cvae_arti_conv2 = nn.Conv1d(3*num_nodes, num_nodes, 3, padding=1)
        self.cvae_batchnorm2 = nn.BatchNorm1d(num_nodes) 
        self.cvae_relu2 = nn.ReLU()
        self.cvae_dropout2 = nn.Dropout(0.5)
        self.cvae_linear = nn.Linear(3*hidden_dim, hidden_dim)
        
        
        
        '''
        tokenize layer for transformer encoder
        '''
        self.encoder_arti_conv1 = nn.Conv1d(num_nodes, 3*num_nodes, 3, padding=1)
        self.encoder_batchnorm1 = nn.BatchNorm1d(3*num_nodes)
        self.encoder_relu1 = nn.ReLU()
        self.encoder_dropout1 = nn.Dropout(0.5)
        
        self.encoder_arti_conv2 = nn.Conv1d(3*num_nodes, 9*num_nodes, 3, padding=1)
        self.encoder_batchnorm2 = nn.BatchNorm1d(9*num_nodes)
        self.encoder_relu2 = nn.ReLU()
        self.encoder_dropout2 = nn.Dropout(0.5)
        
        self.encoder_arti_conv3 = nn.Conv1d(9*num_nodes, token_per_node*num_nodes, 3, padding=1)
        self.encoder_batchnorm3 = nn.BatchNorm1d(token_per_node*num_nodes) 
        self.encoder_relu3 = nn.ReLU()
        self.encoder_dropout3 = nn.Dropout(0.5)
        self.encoder_linear = nn.Linear(3*hidden_dim, hidden_dim)
        
        
        self.temp_dist_net = DenseNetwork(hidden_dim, token_per_node*num_nodes+1) # stack start_arti_info & target_arti_info & proprioceptive
        
        
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_nodes+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        
        # arti_mode에 따라 수정
        self.additional_pos_embed = nn.Embedding(2+token_per_node*num_nodes, hidden_dim)
        # self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

    def forward(self, qpos, image, env_state, actions=None, is_pad=None, arti_info=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        
        '''
        arti_mode
        '''
        target_node_features = arti_info['target']['node_features']
        _, num_nodes, _ = target_node_features.shape
        
        assert target_node_features.shape[1] == num_nodes
        target_edge_features = arti_info['target']['edge_features']
        target_edge_features_4d = target_edge_features.view(bs, num_nodes, num_nodes, self.edge_feat_dim)
        target_edge_features_4d = target_edge_features_4d[..., 0:1].clone() * target_edge_features_4d[..., 1:].clone() 
        target_edge_features_final = target_edge_features_4d.view(bs, num_nodes, num_nodes * (self.edge_feat_dim-1))
        
        
        


        assert target_edge_features.shape[1] == num_nodes * num_nodes, target_edge_features.shape
        
        
        '''
        for cvae input
        '''
        target_node_input = self.node_proj(target_node_features)
        target_edge_input = self.edge_proj(target_edge_features_final)
        
        
        target_arti_input = torch.concat((target_node_input, target_edge_input), axis=-1)
        target_arti_input_clone = target_arti_input.clone()
        
        target_arti_input = self.cvae_dropout1(self.cvae_relu1(self.cvae_batchnorm1(self.cvae_arti_conv1(target_arti_input))))
        target_arti_input = self.cvae_dropout2(self.cvae_relu2(self.cvae_batchnorm2(self.cvae_arti_conv2(target_arti_input))))
        target_arti_embed = self.cvae_linear(target_arti_input)
        

        target_arti_input_encoder = self.encoder_dropout1(self.encoder_relu1(self.encoder_batchnorm1(self.encoder_arti_conv1(target_arti_input_clone))))
        target_arti_input_encoder = self.encoder_dropout2(self.encoder_relu2(self.encoder_batchnorm2(self.encoder_arti_conv2(target_arti_input_encoder))))
        target_arti_input_encoder = self.encoder_dropout3(self.encoder_relu3(self.encoder_batchnorm3(self.encoder_arti_conv3(target_arti_input_encoder))))
        target_arti_embed_encoder = self.encoder_linear(target_arti_input_encoder)

        
        
        if is_training:
            
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)

            
            # node_embed = self.node_head(node_features)
            # edge_embed = self.edge_head(edge_features)
            # mask_embed = self.mask_head(mask_features)

            # encoder_input = torch.cat([cls_embed, node_embed, edge_embed, mask_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1+arti_dim, hidden_dim)
            encoder_input = torch.cat([cls_embed, target_arti_embed, qpos_embed, action_embed], axis=1)  # (bs, seq+1+1+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1+arti_dim, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            # arti_joint_is_pad = torch.full((bs, arti_dim), False).to(qpos.device)
            arti_joint_is_pad = torch.full((bs, self.num_nodes), False).to(qpos.device)

            is_pad = torch.cat([cls_joint_is_pad, arti_joint_is_pad, is_pad], axis=1)  # (bs, seq+1+arti_dim)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        
        # start_arti_input = start_arti_input.squeeze(-1) # bs hidden_dim
        # target_arti_input = target_arti_input.squeeze(-1)


        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id]) # HARDCODED
                features = features[0] # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos) # bs hidden_dim
            
            
            # temporal dist
            temporal_embed = torch.concatenate((target_arti_embed_encoder, proprio_input.unsqueeze(1)), axis=1)
            pred_temporal_dist = self.temp_dist_net(temporal_embed)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, (target_arti_embed_encoder), self.additional_pos_embed.weight)[0]
        else:
            raise NotImplementedError
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar], pred_temporal_dist



class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + 7
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=7, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 7
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = args.state_dim # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    transformer = build_transformer(args)

    encoder = build_encoder(args)

    node_feat_dim = args.node_feat_dim
    edge_feat_dim = args.edge_feat_dim
    image_shape = args.image_shape
    num_nodes = args.num_nodes
    token_per_node = args.token_per_node
    
    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        image_shape=image_shape,
        num_nodes=num_nodes,
        token_per_node=token_per_node,

        
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    state_dim = 7#7 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

