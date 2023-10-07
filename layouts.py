import numpy as np
import torch
from torch_geometric.data import HeteroData
from arguments import args
import torch_geometric.transforms as T



def EuclideanDistances(a, b):
    '''
        Calculate the euclidean distances
    '''
    sq_a = a**2
    sum_sq_a = np.sum(sq_a, axis=1)
    sum_sq_a = np.expand_dims(sum_sq_a, axis=1)       # m->[m, 1]
    sq_b = b**2
    sum_sq_b = np.sum(sq_b, axis=1)  
    sum_sq_b = np.expand_dims(sum_sq_b, axis=0)        # n->[1, n]
    bt = b.T
    distance = np.sqrt(np.abs(sum_sq_a+sum_sq_b-2*np.matmul(a, bt)))
    return distance


def build_graph(user, server, s2u_idx, u2s_index, u2u_idx, s2s_idx, u2u_distance, u2s_path_loss, u2s_path_loss_feat, env_len):
    
    user_feat = user
    server_feat = server
    user_ones = np.zeros((len(user), 1))
    server_ones = np.zeros((len(server), 1))
    user_feat = np.concatenate((user_feat, user_ones), axis=1)
    server_feat = np.concatenate((server_feat, server_ones), axis=1)
    user_feat = torch.tensor(user_feat, dtype=torch.float).to(args.device)
    server_feat = torch.tensor(server_feat, dtype=torch.float).to(args.device)
    u2s_path_loss = torch.tensor(u2s_path_loss, dtype=torch.float).to(args.device)
    
    u2s_path_loss_feat = torch.tensor(u2s_path_loss_feat, dtype=torch.float).to(args.device)

    s2u_attr = u2s_path_loss_feat.reshape((-1, 1))
    u2s_attr = u2s_path_loss_feat.reshape((-1, 1))
    
    u2u_attr = torch.tensor(u2u_distance.reshape((-1, 1)), dtype=torch.float).to(args.device)


    s2u_idx = torch.tensor(s2u_idx, dtype=torch.long).to(args.device)
    u2s_index = torch.tensor(u2s_index, dtype=torch.long).to(args.device)
    u2u_idx = torch.tensor(u2u_idx, dtype=torch.long).to(args.device)
    s2s_idx = torch.tensor(s2s_idx, dtype=torch.long).to(args.device)

    data = HeteroData().to(args.device)

    data['env_len'].x = torch.tensor([env_len])


    data['user'].x = user_feat      # locations of users，size of task to offloading
    data['server'].x = server_feat
    data['user', 'u2u', 'user'].edge_index = u2u_idx
    data['server', 's2u', 'user'].edge_index = s2u_idx
    data['user', 'u2s', 'server'].edge_index = u2s_index

    data['user', 'u2s', 'server'].path_loss = u2s_path_loss_feat.reshape((-1, 1))

    data['server', 's2s', 'server'].edge_index = s2s_idx
    data['server', 's2u', 'user'].edge_attr = s2u_attr
    data['user', 'u2s', 'server'].edge_attr = u2s_attr
    data['user', 'u2u', 'user'].edge_attr = u2u_attr

    return data

def compute_path_losses(args, distances):
    carrier_f = (args.carrier_f_start+args.carrier_f_end)/2
    carrier_lam = 2.998e8 / carrier_f
    signal_cof = args.signal_cof
    path_losses = (signal_cof * carrier_lam) / (distances**2)

    return path_losses

def generate_layouts(user_nums, server_nums,args):
    
    graphs = []
    
    for idx in range(len(server_nums)):

        env_len = np.sqrt(server_nums[idx]*50)

        # 归一化位置, tasksize, computing resource
        user_idx_feat = np.random.random([user_nums[idx], 2])
        user_idx = user_idx_feat * env_len
        
        # user_idx_tasksize = np.random.random((user_nums[idx], 1))*(args.init_max_size-args.init_min_size) + args.init_min_size
        user_idx_tasksize = np.random.random((user_nums[idx], 1))
        user_idx_feat = user_idx_tasksize.copy()

        # server_idx_comp = np.random.random((server_nums[idx], 1))*(args.init_max_comp - args.init_min_comp) + args.init_min_comp
        server_idx_comp = np.random.random((server_nums[idx], 1))
        server_idx_feat = server_idx_comp.copy()

        server_idx_feat = np.random.random([server_nums[idx], 2])
        server_idx = server_idx_feat * env_len


        mask_ones = np.ones(server_nums[idx])
        mask_idx = np.repeat(mask_ones[np.newaxis, :], repeats=user_nums[idx], axis=0)

        
        user_num_idx = user_nums[idx]
        # server_num_idx = server_nums[idx]

        # edge_index of users to users
        index_src = np.arange(user_num_idx).repeat(repeats=user_num_idx)
        index_dst = np.tile(np.arange(user_num_idx), reps=user_num_idx)
        u2u_index = np.concatenate([index_src[np.newaxis, :], index_dst[np.newaxis, :]], axis=0)

        # edge_index of servers to users
        index_s2u_dst, index_s2u_src = np.nonzero(mask_idx)
        s2u_index = np.concatenate([index_s2u_src[np.newaxis, :], index_s2u_dst[np.newaxis, :]], axis=0)

        # edge_index of users to servers
        index_u2s_src, index_u2s_dst = np.nonzero(mask_idx)
        u2s_index = np.concatenate([index_u2s_src[np.newaxis, :], index_u2s_dst[np.newaxis, :]], axis=0)

        # edge_index of servers to servers
        edge_mat = mask_ones[np.newaxis, :].repeat(repeats=server_nums[idx], axis=0)
        index_s2s_src, index_s2s_dst = np.nonzero(edge_mat)
        s2s_index = np.concatenate([index_s2s_src[np.newaxis, :], index_s2s_dst[np.newaxis, :]], axis=0)

        u2u_distances_idx = EuclideanDistances(user_idx, user_idx)
        u2s_distances_idx = EuclideanDistances(user_idx, server_idx)
        u2s_path_loss = compute_path_losses(args, u2s_distances_idx)


        # generate normalized channels randomly
        u2s_path_loss_feat = torch.rand((user_nums[idx], server_nums[idx]), device=args.device)

        # normalize the distances
        distance_mean = np.mean(u2s_distances_idx)
        distance_var = np.sqrt(np.mean(np.square(u2s_distances_idx-distance_mean)))
        u2s_distances_idx = (u2s_distances_idx-distance_mean)/distance_var
 
        user_idx_feat = user_idx_tasksize
        server_idx_feat = server_idx_comp
        
        graph = build_graph(user_idx_feat, server_idx_feat, s2u_index, u2s_index, u2u_index, s2s_index, u2u_distances_idx, u2s_path_loss, u2s_path_loss_feat, env_len)
        graphs.append(graph)
    
    return graphs
