import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, HeteroConv
from torch_geometric.utils import softmax
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
import numpy as np
import tqdm
from arguments import args



class TaskLoad(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, alpha) -> None:
        super(TaskLoad, self).__init__()
        self.user_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
            )
        self.server_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.convs = nn.ModuleList()

        for layer in range(num_layers):
            conv = HeteroConv({
            ('server', 's2u', 'user'): S2UGNN(input_dim, hidden_dim, output_dim, alpha),
            ('user', 'u2s', 'server'): U2SGNN(input_dim, hidden_dim, output_dim, alpha)
            })
            self.convs.append(conv)

        self.task_allocation_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.LeakyReLU(alpha)
        )

        self.power_alloc_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.LeakyReLU(alpha)
        )

        self.comp_power_alloc_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.LeakyReLU(alpha)
        )



    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict['user'] = self.user_encoder(x_dict['user'])
        x_dict['server'] = self.server_encoder(x_dict['server'])
        x_dict_0 = x_dict.copy()
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            # x_dict['user']  = x_dict['user'] + x_dict_0['user']
            # x_dict['server'] = x_dict['server'] + x_dict_0['server']
            # x_dict_0 = x_dict.copy()
        s2u_attr = self.convs[-1].convs['server__s2u__user'].message_attr
        u2s_attr = self.convs[-1].convs['user__u2s__server'].message_attr

        user_index = edge_index_dict['user', 'u2s', 'server'][0]
        # server_index = edge_index_dict['user', 'u2s', 'server'][1]
        server_index = edge_index_dict['server', 's2u', 'user'][0]

        task_allocation = self.task_allocation_mlp(u2s_attr)
        task_allocation = softmax(task_allocation, index=user_index)
        
        power_allocation = self.power_alloc_mlp(u2s_attr)
        power_allocation = softmax(power_allocation, index=user_index)
        
        comp_allocation = self.comp_power_alloc_mlp(s2u_attr)
        comp_allocation = softmax(comp_allocation, index=server_index)

        return task_allocation, power_allocation, comp_allocation

class U2SGNN(MessagePassing):
    def __init__(self, input_dim, hidden_dim, output_dim, alpha, aggr: Optional[str] = "add", flow: str = "source_to_target", node_dim: int = -2, decomposed_layers: int = 1):
        super(U2SGNN, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim+1, hidden_dim),
            nn.ReLU(),
        )
        self.update_lin = nn.Linear(2*hidden_dim, hidden_dim)
        self.att = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.LeakyReLU(alpha)
        )

        self.relation_mlp = nn.Sequential(
            nn.Linear(2*hidden_dim+1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # self.task_alloc_mlp = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LeakyReLU(alpha)
        # )

        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wr = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x, edge_index, edge_attr):
        x_src = x[0]      # user
        x_dst = x[1]       # server
        # x_src = F.relu(self.linear1(x_src))
        # x_dst = F.relu(self.linear2(x_dst))
        return self.propagate(x=(x_src, x_dst), edge_index=edge_index, edge_attr=edge_attr)
    def message(self, x_i, x_j, edge_index, edge_attr) -> Tensor:
        # 消息传播机制

        # message_mlp计算用户到server的信息传递
        tmp = torch.cat([x_i, edge_attr], dim=1)
        # 计算注意力
        att_weight = self.att(torch.cat([self.Wq(x_i), self.Wr(x_j)], dim=1))
        att_weight = softmax(att_weight, index=edge_index[0], dim=0)        # 根据user进行softmax

        outputs = self.message_mlp(tmp)
        outputs = att_weight*outputs        # 注意力

        # 将注意力特征与边特征结合得到user到server的关系特征
        tmp = torch.cat([x_j, outputs, edge_attr], dim=-1)       # user的特征，server到user的关系特征与边特征
        self.message_attr = self.relation_mlp(tmp)

        # # 关系特征mlp, softmax后得到user对server的分配结果
        # self.task_allocation = self.task_alloc_mlp(message_attr)        # 每个user到server的分配信息

        return outputs
    def update(self, aggr_out, x) -> Tensor:

        output = F.relu(self.update_lin(torch.column_stack([aggr_out, x[1]])))
        
        return output



class S2UGNN(MessagePassing):
    def __init__(self, input_dim, hidden_dim, output_dim, alpha, aggr: Optional[str] = "add", flow: str = "source_to_target", node_dim: int = -2, decomposed_layers: int = 1):
        super(S2UGNN, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim+1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.update_lin = nn.Linear(2*hidden_dim, hidden_dim)
        self.att = nn.Sequential(
            nn.Linear(2*hidden_dim, 1),
            nn.LeakyReLU(alpha)
        )

        self.relation_mlp = nn.Sequential(
            nn.Linear(2*hidden_dim+1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # self.comp_power_alloc_mlp = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 1),
        #     nn.LeakyReLU(alpha)
        # )

        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wr = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x, edge_index, edge_attr):
        x_src = x[0]
        x_dst = x[1]
        # x_src = F.relu(self.linear1(x_src))
        # x_dst = F.relu(self.linear2(x_dst))
        return self.propagate(x=(x_src, x_dst), edge_index=edge_index, edge_attr=edge_attr)
    def message(self, x_i, x_j, edge_index, edge_attr) -> Tensor:
        # 消息传播机制

        # message_mlp计算server到user的信息传递
        tmp = torch.cat([x_i, edge_attr], dim=1)
        # 计算注意力
        att_weight = self.att(torch.cat([self.Wq(x_i), self.Wr(x_j)], dim=1))
        att_weight = softmax(att_weight, index=edge_index[0], dim=0)

        outputs = self.message_mlp(tmp)
        outputs = att_weight*outputs

        # 将注意力特征与边特征结合得到user到server的关系特征
        tmp = torch.cat([x_j, outputs, edge_attr], dim=-1)       # user的特征，server到user的关系特征与边特征
        self.message_attr = self.relation_mlp(tmp)

        return outputs
    def update(self, aggr_out, x) -> Tensor:
        output = F.relu(self.update_lin(torch.column_stack([aggr_out, x[1]])))
        return output


'''
class TaskLoad(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, alpha) -> None:
        super(TaskLoad, self).__init__()
        self.user_linear = nn.Linear(input_dim, hidden_dim)
        self.server_linear = nn.Linear(input_dim-1, hidden_dim)
        
        self.conv1 = HeteroConv({
            ('server', 's2u' , 'user'): S2UGNN(input_dim, hidden_dim, output_dim, alpha)
        })
        self.conv2 = HeteroConv({
            ('user', 'u2u', 'user'): U2UGNN(hidden_dim, hidden_dim, output_dim, alpha)
        })
        self.conv3 = HeteroConv({       # user到server，user根据server获得权值特征，输出user到server的边特征(分配方案)
            ('user', 'u2s', 'server'): S2UGNN_U_Allocation(input_dim, hidden_dim, output_dim, alpha)
        })
        self.conv4 = HeteroConv({       # server到user，进行算力资源分配
            ('user', 'u2s', 'server'):  S2UGNN_S_Allocation(input_dim, hidden_dim, output_dim, alpha)
        })

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # s2u，user获取所有server的信息
        x_dict_1 = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        x_dict['user'] = x_dict_1['user']
        # u2u, user间共享位置信息
        x_dict_2 = self.conv2(x_dict, edge_index_dict, edge_attr_dict)
        x_dict['user'] = x_dict_2['user']
        # u2s, user将特征与server进行注意力获取，得到卸载分配与功率分配方案
        x_dict_3 = self.conv3(x_dict, edge_index_dict, edge_attr_dict)
        task_allocation = self.conv3.convs['user__u2s__server'].task_allocation
        power_allocation = self.conv3.convs['user__u2s__server'].power_allocation
        # 将卸载分配与功率分配方案作为边特征输入到s2u
        edge_attr_dict['user', 'u2s', 'server'] = torch.cat([edge_attr_dict['user', 'u2s', 'server'], task_allocation, power_allocation], dim=-1)
        x_dict_4 = self.conv4(x_dict, edge_index_dict, edge_attr_dict)
        comp_allocation = self.conv4.convs['user__u2s__server'].comp_power_allocation

        assert torch.isnan(task_allocation).sum()==0
        assert torch.isnan(power_allocation).sum()==0
        assert torch.isnan(comp_allocation).sum()==0

        return task_allocation, power_allocation, comp_allocation









class S2UGNN_U_Allocation(MessagePassing):
    def __init__(self, input_dim, hidden_dim, output_dim, alpha, aggr: Optional[str] = "add", flow: str = "source_to_target", node_dim: int = -2, decomposed_layers: int = 1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)

        self.att = nn.Sequential(
            nn.Linear(2*hidden_dim, 1),
            nn.LeakyReLU(alpha)
        )

        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim+1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.relation_mlp = nn.Sequential(
            nn.Linear(2*hidden_dim+1, hidden_dim),
            nn.ReLU()
        )
        self.task_alloc_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.LeakyReLU(alpha)
        )
        self.power_alloc_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.LeakyReLU(alpha)
        )
        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wr = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x, edge_index, edge_attr):
        user = x[0]
        server = x[1]
        user = F.relu(self.linear1(user))
        server = F.relu(self.linear2(server))
        return self.propagate(x=(user, server), edge_index=edge_index, edge_attr=edge_attr)
    def message(self, x_i, x_j, edge_index, edge_attr) -> Tensor:
        src_tmp = torch.cat([x_i, edge_attr], dim=1)
        
        # 计算server对user的注意力特征
        att_weight = self.att(torch.cat([self.Wq(x_i), self.Wr(x_j)], dim=1))
        att_weight = softmax(att_weight, index=edge_index[0])

        outputs = self.message_mlp(src_tmp)

        outputs = att_weight * outputs      # user到server的边特征，与edge_index维度相同

        # 将注意力特征与边特征结合得到user到server的关系特征
        tmp = torch.cat([x_j, outputs, edge_attr], dim=-1)       # user的特征，server到user的关系特征与边特征
        message_attr = self.relation_mlp(tmp)

        # 关系特征mlp, softmax后得到user对server的分配结果
        self.task_allocation = self.task_alloc_mlp(message_attr)
        self.power_allocation = self.power_alloc_mlp(message_attr)

        self.task_allocation = softmax(self.task_allocation, index=edge_index[0], dim=0)
        self.power_allocation = softmax(self.power_allocation, index=edge_index[0], dim=0)


        return super().message(x_j)


class S2UGNN_S_Allocation(MessagePassing):
    def __init__(self, input_dim, hidden_dim, output_dim, alpha, aggr: Optional[str] = "add", flow: str = "source_to_target", node_dim: int = -2, decomposed_layers: int = 1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim+3, hidden_dim),
            nn.ReLU()
        )
        self.att = nn.Sequential(
            nn.Linear(2*hidden_dim, 1),
            nn.LeakyReLU(alpha)
        )
        self.relation_mlp = nn.Sequential(
            nn.Linear(2*hidden_dim+3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.comp_power_alloc_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.LeakyReLU(alpha)
        )
        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wr = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x, edge_index, edge_attr):   # edge_attr为功率分配与task offloading分配结果，均为softmax比例
        x_src = x[0]
        x_dst = x[1]
        self.tasksize = x_dst[:, 2]
        x_src = F.relu(self.linear1(x_src))
        x_dst = F.relu(self.linear2(x_dst))
        return self.propagate(x=(x_src, x_dst), edge_index=edge_index, edge_attr=edge_attr)
    def message(self, x_i, x_j, edge_index, edge_attr) -> Tensor:
        src_tmp = torch.cat([x_j, edge_attr], dim=1)
        
        # 计算server对user的注意力特征
        att_weight = self.att(torch.cat([self.Wq(x_i), self.Wr(x_j)], dim=1))
        att_weight = softmax(att_weight, index=edge_index[1], dim=0)

        outputs = self.message_mlp(src_tmp)

        outputs = att_weight * outputs      # user到server的边特征，与edge_index维度相同

        # 将注意力特征与边特征结合得到user到server的关系特征
        tmp = torch.cat([x_i, outputs, edge_attr], dim=-1)       # server的特征，server到user的关系特征与边特征
        message_attr = self.relation_mlp(tmp)        # 

        # 关系特征mlp, softmax后得到server对user的算力分配结果
        self.comp_power_allocation = self.comp_power_alloc_mlp(message_attr)
        self.comp_power_allocation = softmax(self.comp_power_allocation, index=edge_index[1], dim=0)

        return super().message(x_j)
        
   


class S2UGNN(MessagePassing):
    def __init__(self, input_dim, hidden_dim, output_dim, alpha, aggr: Optional[str] = "add", flow: str = "source_to_target", node_dim: int = -2, decomposed_layers: int = 1):
        super(S2UGNN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim+1, hidden_dim),
            nn.ReLU(),
        )
        self.update_lin = nn.Linear(2*hidden_dim, hidden_dim)
        self.att = nn.Sequential(
            nn.Linear(2*hidden_dim, 1),
            nn.LeakyReLU(alpha)
        )
        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wr = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x, edge_index, edge_attr):
        x_src = x[0]
        x_dst = x[1]
        x_src = F.relu(self.linear1(x_src))
        x_dst = F.relu(self.linear2(x_dst))
        return self.propagate(x=(x_src, x_dst), edge_index=edge_index, edge_attr=edge_attr)
    def message(self, x_i, x_j, edge_index, edge_attr) -> Tensor:
        # 消息传播机制

        # message_mlp计算用户到uav的信息传递
        tmp = torch.cat([x_i, edge_attr], dim=1)
        # 计算注意力
        att_weight = self.att(torch.cat([self.Wq(x_i), self.Wr(x_j)], dim=1))
        att_weight = softmax(att_weight, index=edge_index[0], dim=0)

        outputs = self.message_mlp(tmp)
        outputs = att_weight*outputs
        return outputs
    def update(self, aggr_out, x) -> Tensor:
        output = F.relu(self.update_lin(torch.column_stack([aggr_out, x[1]])))
        return output

class U2UGNN(MessagePassing):
    def __init__(self, input_dim, hidden_dim, ouput_dim, alpha, aggr: Optional[str] = "add", flow: str = "source_to_target", node_dim: int = -2, decomposed_layers: int = 1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        # self.allocation_linear = nn.Linear(hidden_dim, ouput_dim)
        # self.power_linear = nn.Linear(hidden_dim, ouput_dim)
        self.att = nn.Sequential(
            nn.Linear(2*hidden_dim, 1),
            nn.LeakyReLU(alpha)
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # self.update_mlp = nn.Sequential(
        #     nn.Linear(2*hidden_dim, hidden_dim),
        #     nn.ReLU(),
        # )
        self.update_lin = nn.Linear(2*hidden_dim, hidden_dim)
        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wr = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.linear1(x))
        # self.user_num = edge_index[0][-1] + 1
        return self.propagate(edge_index=edge_index, x=x)
    def message(self, x_i, x_j, edge_index) -> torch.Tensor:
        # 消息传播机制

        # message_mlp计算用户到uav的信息传递
        self.outputs = self.message_mlp(torch.column_stack([x_i, x_j]))

        # 计算注意力
        self.att_weight = self.att(torch.cat([self.Wq(x_i), self.Wr(x_j)], dim=1))
        self.att_weight = softmax(self.att_weight, index=edge_index[0], dim=0)
        self.outputs = self.att_weight * self.outputs
        return self.outputs

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Optional[Tensor] = None, dim_size: Optional[int] = None) -> Tensor:
        # 消息聚合
        outputs = super().aggregate(inputs, index, ptr, dim_size)       # 进行aggr操作
        return outputs

    def update(self, aggr_out, x):

        output = F.relu(self.update_lin(torch.column_stack([aggr_out, x])))
        # 映射到01之间
        # power = F.softmax(self.power_linear(x))
        # allocation = F.softmax(self.allocation_linear(x))

        # ouput = torch.cat([allocation, power], dim=1)

        return output

'''


class GnnCritic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, alpha) -> None:
        super(GnnCritic, self).__init__()
        # output_dim: edge_index的数量 --> server_num x user_num
        self.gnn = TaskLoad(num_layers, input_dim, hidden_dim, output_dim, alpha)

        self.critic_mlp = nn.Sequential(
            nn.Linear(output_dim*3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.LeakyReLU(alpha)
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # s2u，user获取所有server的信息
        x_dict_1 = self.gnn.conv1(x_dict, edge_index_dict, edge_attr_dict)
        x_dict['user'] = x_dict_1['user']
        # u2u, user间共享位置信息
        x_dict_2 = self.gnn.conv2(x_dict, edge_index_dict, edge_attr_dict)
        x_dict['user'] = x_dict_2['user']
        # u2s, user将特征与server进行注意力获取，得到卸载分配与功率分配方案
        x_dict_3 = self.gnn.conv3(x_dict, edge_index_dict, edge_attr_dict)
        task_allocation = self.gnn.conv3.convs['user__u2s__server'].task_allocation
        power_allocation = self.gnn.conv3.convs['user__u2s__server'].power_allocation
        # 将卸载分配与功率分配方案作为边特征输入到s2u
        edge_attr_dict['user', 'u2s', 'server'] = torch.cat([edge_attr_dict['user', 'u2s', 'server'], task_allocation, power_allocation], dim=-1)
        x_dict_4 = self.gnn.conv4(x_dict, edge_index_dict, edge_attr_dict)
        comp_allocation = self.gnn.conv4.convs['user__u2s__server'].comp_power_allocation

        sches = torch.cat([task_allocation, power_allocation, comp_allocation], dim=0).squeeze()
        critic_value = self.critic_mlp(sches.detach())

        return task_allocation, power_allocation, comp_allocation, sches, critic_value






class PCNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, alpha) -> None:
        super(PCNet, self).__init__()
        
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.sche_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3*output_dim),
            # nn.LeakyReLU(alpha)
            # nn.Sigmoid()
            # nn.Tanh()
        )

        
        

    def forward(self, u2s_path_loss, user_tasksize, server_comp_resource, edge_index):
        '''
            input: path_losses of users to servers
            output: power_sche, task_sche, comp_sche
        '''
        feats = torch.cat([u2s_path_loss, user_tasksize, server_comp_resource], dim=1)
        batch_num = feats.shape[0]
        
        embeddings = self.encoder(feats)

        sches = self.sche_mlp(embeddings)
        task_sche = sches[:, :self.output_dim]
        task_sche = softmax(task_sche.reshape((-1, 1)), index=edge_index[0])
        task_sche = task_sche.reshape((batch_num, -1))

        power_sche = sches[:, self.output_dim:2*self.output_dim]
        power_sche = softmax(power_sche.reshape((-1, 1)), edge_index[0])
        power_sche = power_sche.reshape((batch_num, -1))

        comp_sche = sches[:, 2*self.output_dim:]
        comp_sche = softmax(comp_sche.reshape((-1, 1)), index=edge_index[1])
        comp_sche = comp_sche.reshape((batch_num, -1))

        # power_sche = self.power_mlp(embeddings)
        # power_sche = softmax(power_sche, edge_index[0])
        # task_sche = self.task_mlp(embeddings)
        # task_sche = softmax(task_sche, index=edge_index[0])
        # comp_sche = self.comp_mlp(embeddings)
        # comp_sche = softmax(comp_sche, index=edge_index[1])

        return task_sche, power_sche, comp_sche


class PCNetCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, alpha, args) -> None:
        super(PCNetCritic, self).__init__()
        
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.sche_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3*output_dim),
            nn.Tanh()
        )
        

        self.critic_mlp = nn.Sequential(
            nn.Linear(output_dim*3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.LeakyReLU(alpha)
        )

    def forward(self, u2s_path_loss, user_tasksize, server_comp_resource, edge_index):
        
        '''
            input: path_losses of users to servers
            output: power_sche, task_sche, comp_sche
        '''

        feats = torch.cat([u2s_path_loss, user_tasksize, server_comp_resource], dim=0)
        embeddings = self.encoder(feats)

        sches = self.sche_mlp(embeddings)
        task_sche = sches[:self.output_dim]
        task_sche = softmax(task_sche, index=edge_index[0])

        power_sche = sches[self.output_dim:2*self.output_dim]
        power_sche = softmax(power_sche, edge_index[0])

        comp_sche = sches[2*self.output_dim:]
        comp_sche = softmax(comp_sche, index=edge_index[1])

        sches = torch.cat([task_sche, power_sche, comp_sche], dim=-1)
        critic_value = self.critic_mlp(sches.detach())

        return task_sche, power_sche, comp_sche, sches, critic_value





class MMSE(nn.Module):
    def __init__(self, input_shape, args) -> None:
        super(MMSE, self).__init__()
        
        self.args = args

        self.task_allocation = nn.Parameter(torch.rand(input_shape, requires_grad=True))
        self.power_allocation = nn.Parameter(torch.rand(input_shape, requires_grad=True))
        self.comp_allocation = nn.Parameter(torch.rand(input_shape, requires_grad=True))


    def forward(self, compute_resource, path_losses, task_size, edge_index):
        epsilon = 1e-9
        extre = 1e-20
        user_index = edge_index[0]      # s2u中源节点的索引
        server_index = edge_index[1]    # s2u中目标节点的索引

        task_sche = softmax(self.task_allocation, edge_index[0])
        power_sche = softmax(self.power_allocation, edge_index[0])
        comp_sche = softmax(self.comp_allocation, edge_index[0])

        task_size = task_size[user_index]
        # task_size = task_size[user_index]*args.tasksize_cof       # 重复采样映射到边中
         
        tasks = task_size * task_sche.squeeze()
        compute_resource = compute_resource[server_index]
        comp = compute_resource * comp_sche.squeeze()

        # power_sche = torch.clamp(power_sche, 1e-5, 1)
        pw = power_sche.squeeze() * path_losses.squeeze()


        pws_list = []
        for idx in range(server_index[-1]+1):
            # 取出所有到同一个server的pw加和
            pws_idx = pw[torch.where(server_index==idx)].sum()
            pws_list.append(pws_idx)
        pws_list = torch.stack(pws_list)
        pws_list = pws_list[server_index]
        interference = pws_list-pw
        rate = torch.log2(1 + torch.div(pw, interference+epsilon))
        # rate = args.band_width * torch.log2(1+torch.div(pw, interference+epsilon))
        
        offloading_time = torch.div(tasks, rate+extre)
        # offloading_time = torch.div(tasks, rate+extre) * (self.args.tasksize_cof/self.args.band_width)

        # compute_time = torch.div(tasks, comp+extre) * (self.args.tasksize_cof*self.args.cons_factor/self.args.comp_cof)
        compute_time = torch.div(tasks, comp+extre)

        time_loss = offloading_time + compute_time
        assert torch.isnan(time_loss).sum()==0

        time_loss_list = []
        for idx in range(user_index[-1]+1):
            tl_idx = time_loss[torch.where(user_index==idx)].sum()
            time_loss_list.append(tl_idx)
        time_loss_list = torch.stack(time_loss_list)

        return time_loss_list.mean()
