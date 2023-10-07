import torch
from arguments import args
import numpy as np
from torch_geometric.loader import DataLoader
from layouts import generate_layouts
from off_loading_models import TaskLoad, PCNet
from tqdm import tqdm
import random

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
    # pw = torch.clamp(pw, 1e-5, 1)
    
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

    # compute_time = torch.clamp(compute_time, -1, 3000)
    # offloading_time = torch.clamp(offloading_time, -1, 3000)

    time_loss = offloading_time + compute_time
    assert torch.isnan(time_loss).sum()==0

    time_loss_list = torch.zeros((len(time_loss), user_index[-1]+1), device=args.device)
    time_loss_list.scatter_(1, user_index.unsqueeze(1), time_loss.unsqueeze(1))
    time_loss_list = time_loss_list.sum(0)

    return time_loss_list.mean()

def HGNN_train(model, train_loader, test_loader):
    policy_losses = []
    test_policy_losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=hgnn_lr)
    optimizer_stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20,  gamma=0.9)
    for time_step in tqdm(range(Epochs)):
        # training
        
        loss_sum = 0
        length = 0
        for graph in train_loader:    # graph为一个batch
            task_allocation, power_allocation, comp_allocation = model(graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict)
            user_index = graph['user', 'u2s', 'server'].edge_index[0]
            server_index = graph['user', 'u2s', 'server'].edge_index[1]
            loss_batch = compute_loss(task_allocation, power_allocation, comp_allocation, graph['server'].x[:, 0], graph['user', 'u2s', 'server'].path_loss, graph['user'].x[:, 0], user_index, server_index)
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            loss_sum += loss_batch.item()
            length += 1
        optimizer_stepLR.step()
        policy_loss = loss_sum/length
        if (time_step+1) % eval_fre == 0:
            eval_loss = HGNN_eval(time_step+1, model, test_loader)
            test_policy_losses.append(eval_loss)
            model.train()
        if (time_step + 1) % save_fre == 0:
            torch.save(model, './TO_models/Pretrain/'+multi_scales+'/HGNN_{}_{}.pt'.format(train_num_layouts, time_step+1))
        policy_losses.append(policy_loss)
        print('step=={}, policy_loss=={}'.format(time_step, policy_loss))
    return model, np.array(policy_losses), np.array(test_policy_losses)

def HGNN_eval(time_step, model, loader):
    loss_sum = 0
    length = 0        
    model.eval()
    for graph in loader:    # graph为一个batch
        task_allocation, power_allocation, comp_allocation = model(graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict)
        user_index = graph['user', 'u2s', 'server'].edge_index[0]
        server_index = graph['user', 'u2s', 'server'].edge_index[1]
        loss_batch = compute_loss(task_allocation, power_allocation, comp_allocation, graph['server'].x[:, 0], graph['user', 'u2s', 'server'].path_loss, graph['user'].x[:, 0], user_index, server_index)
    
        loss_sum += loss_batch.item()
        length += 1
    policy_loss = loss_sum/length

    print('step=={}, evaluate_policy_loss=={}'.format(time_step, policy_loss))
    return policy_loss


def NN_train(train_loader, test_loader):
    server_num = min_server
    sum_loss = 0
    length = 0
    train_losses = []
    test_losses = []

    model = PCNet((server_num**2)*3+server_num*4, args.hidden_dim, (server_num**2)*3, args.alpha).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=pcnet_lr)
    schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    for step in tqdm(range(Epochs)):
        for idx, batch in enumerate(train_loader):
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
            optimizer.step()
            optimizer.zero_grad()
            sum_loss += loss.item()
            length += 1
        # schedular.step()
        mean_loss = sum_loss/length
        train_losses.append(mean_loss)
        if (step+1) % eval_fre == 0:
            eval_loss = NN_eval(step+1, test_loader, model)
            test_losses.append(eval_loss)
            model.train()

        if (step + 1) % save_fre == 0:
            torch.save(model, './TO_models/Pretrain/'+multi_scales+'/PcNet_{}_{}.pt'.format(train_num_layouts, step+1))

        print('Time step: {} \t\t PCNet training loss: {}'.format(step, mean_loss))
        

    return model, np.array(train_losses), np.array(test_losses)

def NN_eval(step, loader, model):
    sum_loss = 0
    length = 0
    model.eval()
    for idx, batch in enumerate(loader):
        u2s_index = batch['user', 'u2s', 'server'].edge_index
        user_index = u2s_index[0]
        server_index = u2s_index[1]
        u2s_path_loss = batch['user', 'u2s', 'server'].path_loss.squeeze().reshape((batch_size, -1))
        u2s_path_loss_feat = batch['user', 'u2s', 'server'].edge_attr.squeeze().reshape((batch_size, -1))
        user_tasksize = batch['user'].x[:, 0].reshape((batch_size, -1))
        server_comp_resource = batch['server'].x[:, 0].reshape((batch_size, -1))

        task_sche, power_sche, comp_sche = model(u2s_path_loss_feat, user_tasksize, server_comp_resource, u2s_index)
        loss = compute_loss_nn(task_sche, power_sche, comp_sche, user_tasksize, server_comp_resource, u2s_path_loss,  user_index, server_index)
        sum_loss += loss.item()
        length += 1
    mean_loss = sum_loss/length
    print('Time step: {}, PCNet evaluate loss: {}'.format(step, mean_loss))
    return mean_loss


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     np.random.seed(seed)



if __name__ == '__main__':
    setup_seed(20)
    Epochs =500
    hgnn_lr = 5e-4
    pcnet_lr = 1e-3
    train_num_layouts = 2048
    test_num_layouts = 512
    batch_size = 32
    eval_fre = 5
    save_fre = 50
    multi_scales = 'large'
    if multi_scales == 'small':
        min_server = 2
        max_server = 10
    elif multi_scales == 'medium':
        min_server = 15
        max_server = 25
    elif multi_scales == 'large':
        min_server = 25
        max_server = 35
    train_server_nums = np.random.randint(min_server, max_server+1, train_num_layouts)
    train_user_nums = 3*train_server_nums
    test_server_nums = np.random.randint(min_server, max_server+1, test_num_layouts)
    test_user_nums = 3*test_server_nums

    nn_train_server_nums = np.random.randint(min_server, min_server+1, train_num_layouts)
    nn_train_user_nums = 3*nn_train_server_nums
    nn_test_server_nums = np.random.randint(min_server, min_server+1, train_num_layouts)
    nn_test_user_nums = 3*nn_test_server_nums


    # env_max_length = np.sqrt(10*300)
    gnn_train_layouts = generate_layouts(train_user_nums, train_server_nums, args)
    gnn_test_layouts = generate_layouts(test_user_nums, test_server_nums, args)
    nn_train_layouts = generate_layouts(nn_train_user_nums, nn_train_server_nums, args)
    nn_test_layouts = generate_layouts(nn_test_user_nums, nn_test_server_nums, args)
    
    
    gnn_train_loader = DataLoader(gnn_train_layouts, batch_size=batch_size, shuffle=True)
    gnn_test_loader = DataLoader(gnn_test_layouts, batch_size=batch_size, shuffle=True)

    nn_train_loader = DataLoader(nn_train_layouts, batch_size=batch_size, shuffle=True)
    nn_test_loader = DataLoader(nn_test_layouts, batch_size=batch_size, shuffle=True)

    # task offloading trainning process
    Ol_Model = TaskLoad(args.num_layers, args.input_dim, args.hidden_dim, args.max_server_num, args.alpha).to(args.device)
    hgnn_model, train_loss, test_loss = HGNN_train(Ol_Model, gnn_train_loader, gnn_test_loader)

    pcnet_model, nn_train_losses, nn_test_losses = NN_train(nn_train_loader, nn_test_loader)
