
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from off_loading_models import PCNet, PCNetCritic, MMSE, GnnCritic, TaskLoad
from arguments import args
from torch_geometric.loader import DataLoader
from layouts import generate_layouts
from tqdm import tqdm


def compute_loss_nn(task_allocation, power_allocation, comp_allocation, task_size, compute_resource, path_losses, user_index, server_index):
    
    # task_size : vector N
    # task_allocation: mat pop_num x 3*M*N
    # index: vector 3*M*N
    
    epsilon = 1e-9
    extre = 1e-20
    server_index_first = server_index.reshape((batch_size, -1))[0]
    user_index_first = user_index.reshape((batch_size, -1))[0]
    # user_index = edge_index[0]      # s2u中源节点的索引
    # server_index = edge_index[1]    # s2u中目标节点的索引
    
    # power_allocation = torch.clamp(power_allocation, 1e-5, 1)
    pw_ini = power_allocation * path_losses    # mat pop_num x M*N
    
    # 将信道状态过小的设置为0
    mask_pw = torch.where(pw_ini<args.pw_threshold)
    
    pw = pw_ini.clone()

    pw[mask_pw] = 0

    comp_allocation_clone = comp_allocation.clone()
    comp_allocation_clone[mask_pw] = 0
    comp_allocation_normed = torch.zeros((comp_allocation_clone.shape[0], comp_allocation_clone.shape[1], server_index_first[-1]+1), device=args.device)
    comp_allocation_normed.scatter_(2, server_index_first.repeat((batch_size, 1)).unsqueeze(2), comp_allocation_clone.unsqueeze(2))
    comp_allocation_normed = comp_allocation_normed.sum(1)[:, server_index_first]
    comp_allocation_normed = torch.div(comp_allocation_clone, comp_allocation_normed+extre)

    task_allocation_clone = task_allocation.clone()
    task_allocation_clone[mask_pw] = 0
    task_allocation_normed = torch.zeros((task_allocation_clone.shape[0], task_allocation_clone.shape[1], user_index_first[-1]+1), device=args.device)
    task_allocation_normed.scatter_(2, user_index_first.repeat((batch_size, 1)).unsqueeze(2), task_allocation_clone.unsqueeze(2))
    task_allocation_normed = task_allocation_normed.sum(1)[:, user_index_first]
    task_allocation_normed = torch.div(task_allocation_clone, task_allocation_normed+extre)

    # 计算速率
    pw_list = torch.zeros((pw.shape[0], pw.shape[1], server_index_first[-1]+1), device=args.device)   # mat pop_num x MN x N
    pw_list.scatter_(2, server_index_first.repeat((batch_size, 1)).unsqueeze(2), pw.unsqueeze(2))
    pws_list = pw_list.sum(1)[:, server_index_first]  # mat pop_num x MN

    interference = pws_list-pw
    rate = torch.log2(1+torch.div(pw, interference+epsilon))
    # rate = args.band_width * torch.log2(1+torch.div(pw, interference+epsilon))


    task_size = task_size[:, user_index_first]       # M*N    
    # task_size = task_size[user_index]*args.tasksize_cof       # 重复采样映射到边中
    tasks = task_size * task_allocation_normed   # mat pop_num x M*N

    compute_resource = compute_resource[:, server_index_first]
    # compute_resource = compute_resource[server_index]*args.comp_cof       # 

    comp = compute_resource * comp_allocation_normed


    # offloading_time = torch.div(tasks, rate+extre) * (args.tasksize_cof/args.band_width)
    offloading_time = torch.div(tasks, rate+extre)

    # compute_time = torch.div(tasks, comp+extre) * (args.tasksize_cof*args.cons_factor/args.comp_cof)
    compute_time = torch.div(tasks, comp+extre)

    time_loss = offloading_time + compute_time      # pop_num x MN
    assert torch.isnan(time_loss).sum()==0


    time_loss_list = torch.zeros((time_loss.shape[0], time_loss.shape[1], user_index_first[-1]+1), device=args.device)
    time_loss_list.scatter_(2, user_index_first.repeat((batch_size, 1)).unsqueeze(2), time_loss.unsqueeze(2))
    time_loss_list = time_loss_list.sum(1)      # pop_num x MN

    return time_loss_list.mean()


def compute_loss(task_allocation, power_allocation, comp_allocation, compute_resource, path_losses, task_size, user_index, server_index):
    
    epsilon = 1e-9
    extre = 1e-20
    # user_index = edge_index[0]      # s2u中源节点的索引
    # server_index = edge_index[1]    # s2u中目标节点的索引
    
    pw_ini = power_allocation.squeeze() * path_losses.squeeze()
    # pw小于阈值的对应power设为0
    pw = pw_ini.clone()
    # mask_pw = torch.where(pw_ini<args.pw_threshold)
    # pw[mask_pw] = 0

    pw_user_list = torch.zeros((len(pw), user_index[-1]+1), device=args.device)
    # pw_user_ini_list = torch.zeros((len(pw), user_index[-1]+1), device=args.device)
    # pw_user_ini_list.scatter_(1, user_index.unsqueeze(1), pw_ini.unsqueeze(1))
    pw_user_list.scatter_(1, user_index.unsqueeze(1), pw_ini.unsqueeze(1))
    # 如果某一个user的发射功率均位于阈值以下
    pw_masked = pw_user_list.clone()
    pw_masked[torch.where(pw_masked < args.pw_threshold)] = 0
    invalid_index = torch.where(pw_masked.sum(0)==0)[0]   # 是否有对所有server都低于阈值的
    # assert len(invalid_index)==0
    max_pw_index = pw_user_list[:, invalid_index].argmax(0) # 对所有server的pw都低于阈值的user 取信号最强的server
    pw_masked[max_pw_index, invalid_index] = pw_user_list[max_pw_index, invalid_index]
    pw = pw_masked.sum(1)
    mask_pw = torch.where(pw==0)

    pw_list = torch.zeros((len(pw), server_index[-1]+1), device=args.device)
    pw_list.scatter_(1, server_index.unsqueeze(1), pw.unsqueeze(1))
    
    pws_list = pw_list.sum(0)[server_index]


    interference = pws_list-pw
    rate = torch.log2(1+torch.div(pw, interference+epsilon))

    task_allocation_clone = task_allocation.clone().squeeze() + 1e-8
    task_allocation_clone[mask_pw] = 0
    task_allocation_normed = torch.zeros((len(task_allocation_clone), user_index[-1]+1), device=args.device)
    task_allocation_normed.scatter_(1, user_index.unsqueeze(1), task_allocation_clone.unsqueeze(1))
    # assert len(torch.where(task_allocation_normed.sum(0)==0)[0]) == 0
    task_allocation_normed_2 = task_allocation_normed.sum(0)[user_index]
    task_allocation_final = torch.div(task_allocation_clone, task_allocation_normed_2+extre)
    # task_allocation_clone =  softmax(task_allocation_clone, user_index)

    task_size = task_size[user_index]
    # task_size = task_size[user_index]*args.tasksize_cof       # 重复采样映射到边中
    
    tasks = task_size * task_allocation_final

    comp_allocation_clone = comp_allocation.clone().squeeze()
    comp_allocation_clone[mask_pw] = 0
    comp_allocation_normed = torch.zeros((len(comp_allocation_clone), server_index[-1]+1), device=args.device)
    comp_allocation_normed.scatter_(1, server_index.unsqueeze(1), comp_allocation_clone.unsqueeze(1))
    comp_allocation_normed_2 = comp_allocation_normed.sum(0)[server_index]
    comp_allocation_final = torch.div(comp_allocation_clone, comp_allocation_normed_2+extre)
    # compute_resource = compute_resource[server_index]*args.comp_cof       # 

    compute_resource = compute_resource[server_index]
    comp = compute_resource * comp_allocation_final


    # offloading_time = torch.div(tasks, rate+extre) * (args.tasksize_cof/args.band_width)
    offloading_time = torch.div(tasks, rate+extre)

    # compute_time = torch.div(tasks, comp+extre) * (args.tasksize_cof*args.cons_factor/args.comp_cof)
    compute_time = torch.div(tasks, comp+extre)


    time_loss = offloading_time + compute_time
    assert torch.isnan(time_loss).sum()==0

    time_loss_list = torch.zeros((len(time_loss), user_index[-1]+1), device=args.device)
    time_loss_list.scatter_(1, user_index.unsqueeze(1), time_loss.unsqueeze(1))
    time_loss_list = time_loss_list.sum(0)

    return time_loss_list.mean()



def gnnCritic_train(model, train_layouts, batch_size):
    sum_loss = 0
    length = 0
    train_losses = []
    test_losses = []
    # actor_lr = 5e-5
    # critic_lr = 1e-4
    actor_optimizer = torch.optim.Adam([
        {'params': model.gnn.parameters(), 'lr': actor_lr},
    ])
    critic_optimizer = torch.optim.Adam(model.critic_mlp.parameters(), lr=critic_lr)
    critic_loss_func = torch.nn.MSELoss()
    for epoch in tqdm(range(Epochs)):
        model.train()
        sches_list = []
        for idx, batch in enumerate(train_layouts):
            task_allocation, power_allocation, comp_allocation, sches, critic_value = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            user_index = batch['user', 'u2s', 'server'].edge_index[0]
            server_index = batch['user', 'u2s', 'server'].edge_index[1]
            total_time = compute_loss(task_allocation, power_allocation, comp_allocation, batch['server'].x[:, 0], batch['user', 'u2s', 'server'].path_loss, batch['user'].x[:, 0], user_index, server_index)
            
            sches_list.append(sches)

            critic_loss = critic_loss_func(total_time.detach(), critic_value)
            critic_loss.backward()

            sum_loss += total_time.item()
            length += 1

            if idx % batch_size == 0:
                critic_optimizer.step()
                critic_optimizer.zero_grad()
                
                sches_list = torch.vstack(sches_list)
                value_loss = model.critic_mlp(sches_list)
                sches_list = []
                value_loss.mean().backward()
                actor_optimizer.step()
                actor_optimizer.zero_grad()

                # sum_loss = 0
        
        mean_loss = sum_loss/length
        train_losses.append(mean_loss)
        print('Epoch: {} \t\t HGnnCritic training loss: {}'.format(epoch, mean_loss))
        
    return train_losses


def NN_critic_train(model, train_layouts, batch_size):
    sum_loss = 0
    length = 0
    train_losses = []
    test_losses = []
    # actor_lr = 5e-5
    # critic_lr = 1e-4
    actor_optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': actor_lr},
        {'params': model.sche_mlp.parameters(), 'lr': actor_lr}
    ])
    critic_optimizer = torch.optim.Adam(model.critic_mlp.parameters(), lr=critic_lr)
    critic_loss_func = torch.nn.MSELoss()
    for step in tqdm(range(args.train_steps)):
        model.train()
        sches_list = []
        for idx, batch in enumerate(train_layouts):
            
            u2s_index = batch['user', 'u2s', 'server'].edge_index
            user_index = u2s_index[0]
            server_index = u2s_index[1]
            u2s_path_loss = batch['user', 'u2s', 'server'].path_loss.squeeze().reshape((batch_size, -1))
            u2s_path_loss_feat = batch['user', 'u2s', 'server'].edge_attr.squeeze().reshape((batch_size, -1))
            user_tasksize = batch['user'].x[:, 0].reshape((batch_size, -1))
            server_comp_resource = batch['server'].x[:, 0].reshape((batch_size, -1))

            task_sche, power_sche, comp_sche, sches, critic_value = model(u2s_path_loss_feat, user_tasksize, server_comp_resource, u2s_index)
            total_time = compute_loss_nn(task_sche, power_sche, comp_sche, user_tasksize, server_comp_resource, u2s_path_loss, user_index, server_index)
            
            # u2s_index = batch['user', 'u2s', 'server'].edge_index
            # u2s_path_loss = batch['user', 'u2s', 'server'].edge_attr.squeeze()
            # user_tasksize = batch['user'].x[:, 0]
            # server_comp_resource = batch['server'].x[:, 0]
            # task_sche, power_sche, comp_sche, sches, critic_value = model(u2s_path_loss, user_tasksize, server_comp_resource, u2s_index)
            # task_sche, power_sche, comp_sche, sches, critic_value = self.nnCritic_model(u2s_path_loss, u2s_index)
            # total_time = compute_loss_nn(task_sche, power_sche, comp_sche, batch['server'].x[:, 0], batch['user', 'u2s', 'server'].path_loss, batch['user'].x[:, 0], u2s_index)
            
            sches_list.append(sches)

            critic_loss = critic_loss_func(total_time.detach(), critic_value)
            critic_loss.backward()

            # if idx % batch_size == 0:
            critic_optimizer.step()
            critic_optimizer.zero_grad()
            
            sches_list = torch.vstack(sches_list)
            value_loss = -model.critic_mlp(sches_list)
            sches_list = []
            value_loss.mean().backward()
            actor_optimizer.step()
            actor_optimizer.zero_grad()
            sum_loss += total_time.item()
            length += 1
        
        mean_loss = sum_loss/length
        train_losses.append(mean_loss)
        print('Time step: {} \t\t PCNetCritic training loss: {}'.format(step, mean_loss))
    
    return train_losses



def MMSE_test(loader):
    mean_loss_list = []
    task_sche_list = []
    power_sche_list = []
    comp_sche_list = []
    for idx, batch in tqdm(enumerate(loader)):
        compute_resource = batch['server'].x[:, 0]
        path_loss = batch['user', 'u2s', 'server'].path_loss
        edge_index = batch['user', 'u2s', 'server'].edge_index
        task_size = batch['user'].x[:, 0]
        
        model = MMSE(input_shape=path_loss.shape, args=args).to(args.device)
        lr = mse_lr
        task_optimizer = torch.optim.SGD([{'params': model.task_allocation}], lr, momentum=0.9)
        power_optimizer = torch.optim.SGD([{'params': model.power_allocation}], lr, momentum=0.9)
        comp_optimizer = torch.optim.SGD([{'params': model.comp_allocation}], lr, momentum=0.9)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
        for time_step in range(mse_epochs):
            batch_loss = model(compute_resource, path_loss, task_size, edge_index)
            batch_loss.backward()
            # optimizer.step()
            if time_step % 3 == 0:
                task_optimizer.step()
                task_optimizer.zero_grad()
            elif time_step % 3 == 1:
                power_optimizer.step()
                power_optimizer.zero_grad()
            elif time_step % 3 == 2:
                comp_optimizer.step()
                comp_optimizer.zero_grad()
            # if time_step % 2 == 0:
            #     task_optimizer.step()
            # elif time_step % 2 == 1:
            #     comp_optimizer.step()
            mean_loss_list.append(batch_loss.item())
            print('batch: {}, time_step: {}, loss : {}'.format(idx, time_step, batch_loss.item()))
        print('batch: {} \t\t loss: {}'.format(idx, batch_loss.item()))
        task_sche_list.append(model.task_allocation.data)
        power_sche_list.append(model.power_allocation.data)
        comp_sche_list.append(model.comp_allocation.data)

    return task_sche_list, power_sche_list, comp_sche_list


def gnn_mse_supervised_train(layouts, model, task_target, power_target, comp_target):

    # task_target, power_target, comp_target = MMSE_test(layouts)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hgnn_sv_lr)
    policy_losses = []
    for epoch in tqdm(range(Epochs)):
        
        # training
        model.train()
        loss_sum = 0
        length = 0
        for graph, tt, pt, ct in zip(layouts, task_target, power_target, comp_target):    # graph为一个batch
            task_allocation, power_allocation, comp_allocation = model(graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict)
            
            loss_batch = compute_loss(task_allocation, power_allocation, comp_allocation, graph['server'].x[:, 2], graph['user', 'u2s', 'server'].path_loss, graph['user'].x[:, 2], graph['user', 'u2s', 'server'].edge_index)
            task_loss = loss_func(task_allocation, tt)
            power_loss = loss_func(power_allocation, pt)
            comp_loss = loss_func(comp_allocation, ct)

            loss = task_loss + power_loss + comp_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss_batch.item()
            length += 1
        policy_loss = loss_sum/length
        policy_losses.append(policy_loss)
        print('epoch=={}, HgnnSupervised policy_loss=={}'.format(epoch, policy_loss))
    
    return policy_losses

def pcnet_mse_supervised_train(layouts, model, task_target, power_target, comp_target):
    
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=pcnet_sv_lr)
    policy_losses = []
    for epoch in tqdm(range(Epochs)):
        
        # training
        model.train()
        loss_sum = 0
        length = 0
        for graph, tt, pt, ct in zip(layouts, task_target, power_target, comp_target):    # graph为一个batch
            
            u2s_index = graph['user', 'u2s', 'server'].edge_index
            u2s_path_loss = graph['user', 'u2s', 'server'].edge_attr.squeeze()
            user_tasksize = graph['user'].x[:, 2]
            server_comp_resource = graph['server'].x[:, 2]
            task_sche, power_sche, comp_sche = model(u2s_path_loss, user_tasksize, server_comp_resource, u2s_index)
            total_time = compute_loss(task_sche, power_sche, comp_sche, server_comp_resource, graph['user', 'u2s', 'server'].path_loss, user_tasksize, u2s_index)
            
            # task_allocation, power_allocation, comp_allocation = model(graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict)
            
            # loss_batch = compute_loss(task_allocation, power_allocation, comp_allocation, graph['server'].x[:, 2], graph['user', 'u2s', 'server'].path_loss, graph['user'].x[:, 2], graph['user', 'u2s', 'server'].edge_index)
            task_loss = loss_func(task_sche, tt)
            power_loss = loss_func(power_sche, pt)
            comp_loss = loss_func(comp_sche, ct)

            loss = task_loss + power_loss + comp_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += total_time.item()
            length += 1
        policy_loss = loss_sum/length

        policy_losses.append(policy_loss)
        print('epoch=={}, PcNetSupervised policy_loss=={}'.format(epoch, policy_loss))
    
    return policy_losses

def HGNN_train(model, train_loader):
    policy_losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=hgnn_lr)
    optimizer_stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20,  gamma=0.9)
    for time_step in tqdm(range(Epochs)):
        
        # training
        model.train()
        loss_sum = 0
        length = 0
        for graph in train_loader:    # graph为一个batch
            task_allocation, power_allocation, comp_allocation = model(graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict)
            
            loss_batch = compute_loss(task_allocation, power_allocation, comp_allocation, graph['server'].x[:, 0], graph['user', 'u2s', 'server'].path_loss, graph['user'].x[:, 0], graph['user', 'u2s', 'server'].edge_index)
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            loss_sum += loss_batch.item()
            length += 1
        optimizer_stepLR.step()
        policy_loss = loss_sum/length

        policy_losses.append(policy_loss)
        print('step=={}, policy_loss=={}'.format(time_step, policy_loss))
    return policy_losses


def NN_train(model, loader):
    sum_loss = 0
    length = 0
    train_losses = []
    test_losses = []

    optimizer = torch.optim.Adam(model.parameters(), lr=pcnet_lr)
    schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.98)
    for step in tqdm(range(Epochs)):
        model.train()
        for idx, batch in enumerate(loader):
            u2s_index = batch['user', 'u2s', 'server'].edge_index
            user_index = u2s_index[0]
            server_index = u2s_index[1]
            u2s_path_loss = batch['user', 'u2s', 'server'].path_loss.squeeze().reshape((batch_size, -1))
            u2s_path_loss_feat = batch['user', 'u2s', 'server'].edge_attr.squeeze().reshape((batch_size, -1))
            user_tasksize = batch['user'].x[:, 0].reshape((batch_size, -1))
            server_comp_resource = batch['server'].x[:, 0].reshape((batch_size, -1))

            task_sche, power_sche, comp_sche = model(u2s_path_loss_feat, user_tasksize, server_comp_resource, u2s_index)
            loss = compute_loss_nn(task_sche, power_sche, comp_sche, user_tasksize, server_comp_resource, u2s_path_loss, user_index, server_index)
            loss.backward()
            # if idx % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()
            sum_loss += loss.item()
            length += 1
        schedular.step()
        
        mean_loss = sum_loss/length
        train_losses.append(mean_loss)
        print('Time step: {} \t\t PCNet training loss: {}'.format(step, mean_loss))
        
    
    return train_losses, test_losses



if __name__ == '__main__':
    # 生成场景
    np.random.seed(50)
    torch.cuda.manual_seed_all(50)
    num_layouts = 256
    server_num = 15
    batch_size = 16
    Epochs = 2000

    hgnn_lr = 5e-3
    pcnet_lr = 1e-5
    
    hgnn_sv_lr = 1e-4
    pcnet_sv_lr = 5e-4

    actor_lr = 1e-4
    critic_lr = 1e-4

    mse_epochs = 300
    mse_lr = 0.05

    train_user_nums = np.random.randint(server_num*3, server_num*3+1, num_layouts)
    train_server_nums = np.random.randint(server_num, server_num+1, num_layouts)
    # test_user_nums = np.random.randint(server_num*3, server_num*3+1, args.test_layouts)
    # test_server_nums = np.random.randint(server_num, server_num+1, args.test_layouts)

    # env_max_length = np.sqrt(server_num * 300)

    train_layouts = generate_layouts(train_user_nums, train_server_nums, args)
    # test_layouts = generate_layouts(test_user_nums, test_server_nums, args)

    train_loader = DataLoader(train_layouts, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_layouts, batch_size=args.batch_size, shuffle=False)

    # single_task_target, single_power_target, single_comp_target = MMSE_test(train_layouts)
    # batch_task_target, batch_power_target, batch_comp_target = MMSE_test(train_loader)

    # gnn_supervised_model = TaskLoad(args.num_layers, args.input_dim, args.hidden_dim, args.max_server_num, args.alpha).to(args.device)
    gnn_model = TaskLoad(args.num_layers, args.input_dim, args.hidden_dim, args.max_server_num, args.alpha).to(args.device)
    # gnn_model = torch.load('./TO_models/medium/medium_600.pt', map_location=args.device)
    # hgnn_supervised_losses = gnn_mse_supervised_train(train_loader, gnn_supervised_model, batch_task_target, batch_power_target, batch_power_target)
    # hgnn_losses = HGNN_train(gnn_model, train_layouts)

    
    
    
    # PcNetCritic_model = PCNetCritic((server_num**2)*3+server_num*4, args.hidden_dim, (server_num**2)*3, args.alpha, args).to(args.device)
    # HGnnCritic_model = GnnCritic(args.num_layers, args.input_dim, args.hidden_dim, (server_num**2)*3, args.alpha).to(args.device)

    # pcNet_critic_losses = NN_critic_train(PcNetCritic_model, train_loader, batch_size)
    # HGNN_critic_losses= gnnCritic_train(HGnnCritic_model, train_layouts, batch_size)


    

    pcnet = PCNet((server_num**2)*3+server_num*4, args.hidden_dim, (server_num**2)*3, args.alpha).to(args.device)
    pcnet_supervised_model = PCNet((server_num**2)*3+server_num*4, args.hidden_dim, (server_num**2)*3, args.alpha).to(args.device)

    # pcnet_supervised_losses = pcnet_mse_supervised_train(train_layouts, pcnet_supervised_model, single_task_target, single_power_target, single_comp_target)
    pcnet_losses = NN_train(pcnet, train_loader)
