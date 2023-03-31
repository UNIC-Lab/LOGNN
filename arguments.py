
import argparse
args = argparse.ArgumentParser()
#offloading 模型参数
args.add_argument('--learning_rate', default=1e-4)
args.add_argument('--critic_lr', default=5e-4)
args.add_argument('--input_dim', default=2)
args.add_argument('--state_dim', default=3)
args.add_argument('--action_dim', default=2)
args.add_argument('--hidden_dim', default=64)
args.add_argument('--alpha', default=0.2)
args.add_argument('--device', default='cuda:0')
args.add_argument('--load_pretrained', default=False)
args.add_argument('--num_layers', default=2)

# 实验环境参数
args.add_argument('--user_num', default=20)
args.add_argument('--server_num', default=5)
args.add_argument('--test_user_num', default=20)
args.add_argument('--test_server_num', default=5)
args.add_argument('--p_max', default=1)
args.add_argument('--pw_threshold', default=1e-6)
args.add_argument('--train_layouts', default=128)
args.add_argument('--test_layouts', default=64)
args.add_argument('--env_max_length', default=400)
args.add_argument('--server_height', default=20)
args.add_argument('--carrier_f_start', default=2.4e9)
args.add_argument('--carrier_f_end', default=2.4835e9)
args.add_argument('--signal_cof', default=4.11)
args.add_argument('--band_width', default=1e6)
args.add_argument('--batch_size', default=32)
args.add_argument('--max_server_num', default=15)
args.add_argument('--init_min_size', default=2)
args.add_argument('--init_max_size', default=8)

args.add_argument('--cons_factor', default=10)
args.add_argument('--init_min_comp', default=0.1)
args.add_argument('--init_max_comp', default=1)
args.add_argument('--comp_cof', default=1024**2)
args.add_argument('--tasksize_cof', default=1024*100)



args.add_argument('--multi_scales_train', default=False)

args.add_argument('--multi_scales_test', default=False)

args.add_argument('--single_scale_test', default=True)

args.add_argument('--comparison_hgnn', default=True)
args.add_argument('--comparison_pcnet', default=True)
args.add_argument('--comparison_pcnetCritic', default=False)


args.add_argument('--train_steps', default=600)
args.add_argument('--evaluate_steps', default=10)
args.add_argument('--save_steps', default=50)



args = args.parse_args()

