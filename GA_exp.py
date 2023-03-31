import torch
import random
from tqdm import tqdm
from arguments import args
import numpy as np
from layouts import generate_layouts
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax
from time import time



class Evolution():
    def __init__(self, N, M, pop_num, total_epochs, compute_resource, path_losses, task_size, edge_index) -> None:
        
        self.N = N      # user数量
        self.M = M      # server数量
        self.pop_num = pop_num     # 初始种群规模
        self.retain_rate = 0.4      # 保存率
        self.mutate_rate = 0.2
        self.random_select_rate = 0.1
        # self.locations = locations
        self.total_epoch = total_epochs
        self.b = 2


        self.compute_resource = compute_resource
        self.path_losses = path_losses
        self.task_size = task_size
        self.edge_index = edge_index

    def evolution(self):
        populication = self.populication_create()
        loss_list = []
        time_list = []
        for i in range(self.total_epoch):
            # print(populication.shape)
            start = time()
            parents, output_i = self.selection(populication)
            cs = self.cross_over(parents)
            cs = self.mutation(cs, i)
            populication = torch.cat([parents, cs], dim=0)
            # output_i = self.adaptbility(populication)
            
            min_time_loss = torch.min(output_i)       # 最好的一个对应的time loss
            
            loss_list.append(min_time_loss)
            # print('epoch == {}, time of best_one == {}'.format(i, min_time_loss))
            end = time()
            time_list.append(end-start)
        loss_list = torch.stack(loss_list)
        time_list = np.array(time_list)
        return loss_list, time_list
    
        

    def cross_over(self, parent):
        # 交叉, 单点交叉
        '''
        #选择父代
        male = []
        female = []
        # 选择基因位并交叉      //单点交叉
        '''

        # 均匀交叉
        children = []
        get_child_num = self.pop_num-len(parent)
        while len(children) < get_child_num:
            i = random.randint(0, len(parent)-1)
            j = random.randint(0, len(parent)-1)
            male = parent[i]
            female = parent[j]
            select_p = torch.rand(len(male), device=args.device)
            select_p[torch.where(select_p < 0.5)] = 0
            select_p[torch.where(select_p >= 0.5)] = 1
            child1 = select_p * male + (1-select_p) * female
            child2 = (1 - select_p) * male + select_p * female
            children.append(child1.reshape(1, len(child1)))
            children.append(child2.reshape(1, len(child2)))
        if len(children) != 0:
            children = torch.cat(children, dim=0)
        if get_child_num < len(children):
            children = children[:-1]
        return children

    def populication_create(self):
        # 生成种群
        self.populication = torch.rand((self.pop_num, 3*self.M*self.N), device=args.device)
        # self.users = torch.tensor(self.features[:2*self.N], device=args.device)
        
        return self.populication

    def mutation(self, cs, i):
        # 变异
        
        # 采用非一致性变异，每个位置都进行变异
        new_cs = cs.clone()
        for idx, c in enumerate(cs):
            if random.random() < self.mutate_rate:
                r = random.random()
                mut1 = (1-c)*torch.rand(len(c), device=args.device)*(1-i/self.total_epoch)**self.b
                mut2 = torch.rand(len(c), device=args.device)*(1-i/self.total_epoch)**self.b
                # print(mut1)
                if random.random() > 0.5:
                    c = c + mut1
                else:
                    c = c - mut2
                # print(c)
            new_cs[idx] = c
            # print(c)
        return new_cs
            

    def selection(self, populication):
        # 选择

        # 选择最佳的rate率的个体
        # 对种群从小到大进行排序
        adpt = self.adaptbility(populication)
        # grabed = [[ad, one] for ad, one in zip(adpt, populication)]
        
        sort_index = torch.argsort(adpt)
        grabed = populication[sort_index]
        sorted_adpt = adpt[sort_index]
        # sorted_grabed = sorted(grabed, key=lambda x: x[0])
        # grabed = torch.tensor([x[1] for x in sorted_grabed], device=args.device)
        index = int(len(populication)*self.retain_rate)

        live = grabed[:index]
        
        live_adpt = sorted_adpt[:index]

        # 选择幸运个体
        for i, ad_i in zip(grabed[index:], adpt[index:]):
            if random.random() < self.random_select_rate:
                live = torch.cat([live, i.reshape(1, len(i))], dim=0)
                live_adpt = torch.cat([live_adpt, ad_i.unsqueeze(0)], dim=0)
                # live_adpt = torch.stack([live_adpt, ad_i.unsqueeze(0)])
        
        return live, adpt
    
    def adaptbility(self, populication):

        task_allocation = populication[:, :self.M*self.N]
        task_allocation = softmax(task_allocation, index=self.edge_index[0], dim=1)
        power_allocation = populication[:, self.M*self.N:2*self.M*self.N]
        power_allocation = softmax(power_allocation, index=self.edge_index[0], dim=1)
        comp_allocation = populication[:, 2*self.N*self.M:]
        comp_allocation = softmax(comp_allocation, index=self.edge_index[1], dim=1)


        
        time_losses = self.compute_loss(task_allocation, power_allocation, comp_allocation)
        
        # max_dist = []
        # for p in torch.FloatTensor(populication).to(args.device):
        #     p = p.reshape(int(len(p)/2), 2)
        #     # p = torch.FloatTensor(p).to(device)
        #     users_uav = torch.cat([self.users, p], dim=0)
        #     max_dist.append(self.flow_loss(users_uav).cpu().data.numpy())
        return time_losses.mean(-1)

    def compute_loss(self, task_allocation, power_allocation, comp_allocation):
        
        # task_size : vector N
        # task_allocation: mat pop_num x 3*M*N
        # index: vector 3*M*N
        
        epsilon = 1e-9
        extre = 1e-20
        user_index = self.edge_index[0]      # s2u中源节点的索引
        server_index = self.edge_index[1]    # s2u中目标节点的索引
        
        task_size = self.task_size[user_index]       # M*N    
        # task_size = task_size[user_index]*args.tasksize_cof       # 重复采样映射到边中
        
        tasks = task_size * task_allocation   # mat pop_num x M*N

        compute_resource = self.compute_resource[server_index]
        # compute_resource = compute_resource[server_index]*args.comp_cof       # 

        comp = compute_resource * comp_allocation

        pw = power_allocation * self.path_losses    # mat pop_num x M*N
        # pw = torch.clamp(pw, 1e-5, 1)

        pw_list = torch.zeros((pw.shape[0], pw.shape[1], server_index[-1]+1), device=args.device)   # mat pop_num x MN x N
        pw_list.scatter_(2, server_index.repeat((self.pop_num, 1)).unsqueeze(2), pw.unsqueeze(2))
        pws_list = pw_list.sum(1)[:, server_index]  # mat pop_num x MN


        interference = pws_list-pw
        rate = torch.log2(1+torch.div(pw, interference+epsilon))
        # rate = args.band_width * torch.log2(1+torch.div(pw, interference+epsilon))
        # offloading_time = torch.div(tasks, rate+extre) * (args.tasksize_cof/args.band_width)
        offloading_time = torch.div(tasks, rate+extre)

        # compute_time = torch.div(tasks, comp+extre) * (args.tasksize_cof*args.cons_factor/args.comp_cof)
        compute_time = torch.div(tasks, comp+extre)

        time_loss = offloading_time + compute_time      # pop_num x MN
        assert torch.isnan(time_loss).sum()==0


        time_loss_list = torch.zeros((time_loss.shape[0], time_loss.shape[1], user_index[-1]+1), device=args.device)
        time_loss_list.scatter_(2, user_index.repeat((self.pop_num, 1)).unsqueeze(2), time_loss.unsqueeze(2))
        time_loss_list = time_loss_list.sum(1)      # pop_num x MN

        return time_loss_list


if __name__=='__main__':
    server_num = 5

    pop_num = 200
    epochs = 600

    sample_num = 5


    train_user_nums = np.random.randint(server_num*3, server_num*3+1, sample_num)
    train_server_nums = np.random.randint(server_num, server_num+1, sample_num)
    # test_user_nums = np.random.randint(server_num*3, server_num*3+1, args.test_layouts)
    # test_server_nums = np.random.randint(server_num, server_num+1, args.test_layouts)

    # env_max_length = np.sqrt(server_num * 300)

    train_layouts = generate_layouts(train_user_nums, train_server_nums, args)
    # test_layouts = generate_layouts(test_user_nums, test_server_nums, args)

    train_loader = DataLoader(train_layouts, batch_size=args.batch_size, shuffle=True)
    # test_loader = DataLoader(test_layouts, batch_size=args.batch_size, shuffle=True)
    loss_list = []
    for data in train_layouts:
        compute_resource = data['server'].x[:, 0].squeeze()
        path_losses = data['user', 'u2s', 'server'].path_loss.squeeze()
        task_size = data['user'].x[:, 0].squeeze()
        edge_index = data['user', 'u2s', 'server'].edge_index
        ga = Evolution(server_num, server_num*3, pop_num, epochs, compute_resource=compute_resource, path_losses=path_losses, task_size=task_size, edge_index=edge_index)
        loss, _ = ga.evolution()
        loss_list.append(loss.item()[-1])
    print(np.mean(loss_list))

