from sim import Environment
from center import Platform
import sys
import numpy as np
import argparse
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from runner import train, test, train_competitive, test_competitive
from federation import FederatedServer
from regulator import Regulator


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='SpatioTemporal Pricing')
    # Simulation settings
    parser.add_argument('-d', '--data_folder',
                        help='the folder that contains all the input data')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-m', '--mode', type=str, default='all',
                        help='running mode, test or all which means include both train and test. (DEFAULT: all)')
    parser.add_argument('-n', '--name', type=str, default='0', help='name_of the run')
    parser.add_argument('-rs', '--seed', type=int, default=47, help='random seed')
    # Training settings
    parser.add_argument('-e', '--n_epochs', type=int, default=10,
                        help='number of epochs (DEFAULT: 10)')
    parser.add_argument('-p', '--pricing_alg', type=str, default='dummy',
                        help='pricing algorithms, can be ddpg_MLP, ddpg_CNN, dummy, or equilibrium')
    parser.add_argument('-c', '--device', type=str, default='cpu',
                        help='cpu or gpu (cuda)')
    parser.add_argument('-f', '--frequency', type=int, default=1,
                        help='pricing freqency') 
    parser.add_argument('-pc', '--policy_constr', action='store_true', default=False,
                        help='constrained pricing change speed')  
    parser.add_argument('-re', '--resume', type=int, default=0,
                        help='resume_prev_training')
  
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='batch_size. (DEFAULT: 32)')
    parser.add_argument('-k', '--kernel_size', type=int, default=3,
                        help='kernel size of CNN layer')
    parser.add_argument('-s', '--stride', type=int, default=2,
                        help='stride of CNN layer')
    parser.add_argument('-o', '--od_permutation', action='store_false', default=True,
                        help='enable the permutation of OD features')
    parser.add_argument('-alr', '--actor_lr', type=float, default=0.00001, help='learning rate of actor')
    parser.add_argument('-clr', '--critic_lr', type=float, default=0.001, help='learning rate of critic')
    parser.add_argument('-u', '--update_frequency', type=int, default=25,
                        help='base frequency for policy update') 
    parser.add_argument('-ps', '--pooling', type=int, default=2, help='kernel/stride size of pooling layer for downsampling')
    parser.add_argument('-sa', '--searching', type=str, default='Gaussian', help='searching mechanism')
    parser.add_argument('-pd', '--policy_delay', type = int, default = 0,
                        help='incremental policy delay')
    parser.add_argument('-pe', '--position_encode', type=int, default= 64,\
                        help='whether to use NN for positional encoding')
    parser.add_argument('-ac', '--auto_correlation', action='store_true', default=False,
                        help='whether to use auto-correlated noise for exploration')
    parser.add_argument('-fg', '--forget', action='store_true', default=False,
                        help='whether to use forget in the critic network')
    parser.add_argument('-pm', '--pricing_mode', action='store_true', default=False,
                        help='demand sensitivity to price, either inverse or linear')
    parser.add_argument('--competitive', action='store_true', default=False,
                        help='run multi-platform competitive mode ')
    parser.add_argument('--num_platforms', type=int, default=2,
                        help='number of competing platforms when --competitive is set ')
    parser.add_argument('--brand_prefs', nargs='+', type=float, default=None,
                        help='per-platform brand preference ASC, e.g. --brand_prefs 0.3 0.0 ')
    parser.add_argument('--match_time_caps', nargs='+', type=int, default=None,
                        help='per-platform pickup time cutoff in minutes, e.g. --match_time_caps 15 20; '
                             'lower cap = better dispatch efficiency; default is 15 for all platforms')
    parser.add_argument('--fleet_fractions', nargs='+', type=float, default=None,
                        help='fraction of total fleet assigned to each platform, e.g. --fleet_fractions 0.6 0.4; '
                             'must sum to 1.0; default is equal split')
    # learning architectures compared in the paper (Section 5.3); all require --competitive
    parser.add_argument('--architecture', type=str, default='local', choices=['local', 'frl', 'pfrl', 'hrl'],
                        help='local RL, regulator-weighted FRL, personalized FRL, or hierarchical RL')
    parser.add_argument('--no_outside_option', action='store_true', default=False,
                        help='split the raw rider pool without price elasticity')
    parser.add_argument('--lambda_rs', type=float, default=1.0, help='regulator weight on the rider-surplus index')
    parser.add_argument('--lambda_served', type=float, default=1.0, help='regulator weight on served demand')
    parser.add_argument('--lambda_tv', type=float, default=1.0, help='regulator weight on temporal price volatility')
    parser.add_argument('--lambda_sd', type=float, default=1.0, help='regulator weight on spatial price dispersion')
    parser.add_argument('--lambda_gini', type=float, default=1.0, help='regulator weight on rider-surplus inequality')
    parser.add_argument('--tau_frl', type=float, default=0.5, help='softmax temperature of the FRL aggregation weights')
    parser.add_argument('--share_beta', type=float, default=0.4,
                        help='personalized FRL mixing coefficient beta')
    parser.add_argument('--fed_interval', type=int, default=1440,
                        help='simulation steps between federated rounds')
    parser.add_argument('--val_steps', type=int, default=1440,
                        help='length of each candidate validation rollout in simulation steps')
    parser.add_argument('--hrl_interval', type=int, default=10, help='HRL regulatory interval H in simulation steps')
    parser.add_argument('--guidance_dim', type=int, default=1, help='dimension of the HRL guidance signal')
    args = parser.parse_args(argv)

    if args.architecture == 'frl':
        args.share_beta = 1.0
    if args.architecture != 'local':
        if not args.competitive or args.pricing_alg != 'TD3_MLP':
            parser.error('--architecture ' + args.architecture + ' requires --competitive and --pricing_alg TD3_MLP')
        if args.fed_interval % args.frequency != 0 or args.hrl_interval % args.frequency != 0:
            parser.error('--fed_interval and --hrl_interval must be multiples of --frequency')

    return args


if __name__ == '__main__':

    # loading arguments
    args = get_arguments(sys.argv[1:])

    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    permutation_tag = '' if not args.od_permutation else '_perm'
    constraint_tag = '' if not args.policy_constr else '_constr' 
    auto_tag = '' if not args.auto_correlation else '_auto'
    forget_tag = '' if not args.forget else '_forget'

    scenario_tag = args.data_folder.replace("/", "_")

    arch_tag = '_fixedpool' if args.no_outside_option else ''
    if args.architecture in ['frl', 'pfrl']:
        arch_tag += '_lrs' + str(args.lambda_rs) + '_tau' + str(args.tau_frl) + '_beta' + str(args.share_beta)
    elif args.architecture == 'hrl':
        arch_tag += '_lrs' + str(args.lambda_rs) + '_H' + str(args.hrl_interval)
    settings_tag = args.pricing_alg + permutation_tag + constraint_tag + auto_tag + forget_tag + "_" +args.searching + "_" +\
                   str(args.policy_delay) +"_"+str(args.position_encode) + arch_tag + "_" + args.name
    hyperp_tag = str(args.batch_size) + "_" + str(args.actor_lr) + "_" + str(args.critic_lr) + "_" + str(args.kernel_size)

  
    store_run_folder = f"results/{scenario_tag[:-1]}/runs"
    store_model_folder = f"results/{scenario_tag[:-1]}/models/{settings_tag}/{hyperp_tag}" 
    if (args.mode == 'test'):
        store_res_folder = f"results/{scenario_tag[:-1]}/tests/{settings_tag}/{hyperp_tag}"
    else:
        store_res_folder = f"results/{scenario_tag[:-1]}/results/{settings_tag}/{hyperp_tag}"

    if (not os.path.isdir(
            store_run_folder + "/")):
        os.makedirs(
            store_run_folder + "/")

    if (not os.path.isdir(
            store_model_folder + "/")):
        os.makedirs(
            store_model_folder + "/")

    if (not os.path.isdir(
            store_res_folder + "/")):
        os.makedirs(
            store_res_folder + "/")

    args.store_run_folder = store_run_folder
    args.store_model_folder = store_model_folder
    args.store_res_folder = store_res_folder

    if args.mode == 'all' and (os.path.exists(args.store_res_folder+"/"+"train_log_"+str(args.n_epochs)+".csv")):
        print("Files are already there, exit!")
        sys.exit()

    if (args.mode == 'test') and (not args.competitive) and (len(os.listdir(args.store_model_folder)) == 0) and (args.pricing_alg not in ['dummy','equilibrium']):
        print("No trained model can be found, exit!")
        sys.exit()

    if (args.mode == 'test') and os.path.exists(args.store_res_folder+"/"+"test_log_1.csv"):
        print("Files are already there, exit!")
        sys.exit()

    if args.verbose:
        print("Loading data")

    writer = None
    if args.mode == 'all':
        writer = SummaryWriter(store_run_folder + "/" + settings_tag + "_" + hyperp_tag)

    td = np.load(args.data_folder + 'td.npy')
    tt = np.load(args.data_folder + 'tt.npy')
    vp = np.load(args.data_folder + 'vp.npy')

    dd_train = np.load(args.data_folder + 'dd_train.npy', mmap_mode='r')
    dd_test = np.load(args.data_folder + 'dd_test.npy', mmap_mode='r')
    permutation = np.load(args.data_folder + 'permutation.npy')
    permutation2 = np.load(args.data_folder + 'permutation2.npy')
    args.num_zone = td.shape[0]
    T_train = dd_train.shape[0]
    T_test = dd_test.shape[0]

    guidance_dim = args.guidance_dim if args.architecture == 'hrl' else 0

    if args.verbose:
        print("Initialize environment, vehicle num: " + str(len(vp)))
    if args.pricing_alg == 'PPO_CNN':
        controller = Platform(td, tt, args.num_zone, actor_lr=args.actor_lr, critic_lr=args.critic_lr, option='PPO_CNN',
                              permutation=permutation, permutation2=permutation2, kernel_size=args.kernel_size,
                              stride=args.stride, pooling=args.pooling, device=args.device, writer=writer,
                              od_permutation=args.od_permutation, update_freq=args.frequency,
                              veh_num=len(vp), \
                              demand_mean=np.mean(dd_train, axis=0), demand_std=np.std(dd_train, axis=0),
                              wage_mean=np.mean(vp, axis=0), wage_std=np.std(vp, axis = 0),
                              searching=args.searching,
                              position_encode=args.position_encode,
                              auto_correlation=args.auto_correlation,
                              forget=args.forget,
                              T_train = T_train
                              )
    elif args.pricing_alg == 'PPO_MLP':
        controller = Platform(td, tt, args.num_zone, actor_lr=args.actor_lr, critic_lr=args.critic_lr, option='PPO_MLP',
                              device=args.device,
                              writer=writer, update_freq=args.frequency, veh_num=len(vp),
                              demand_mean=np.mean(dd_train, axis=0), demand_std=np.std(dd_train, axis=0),
                              wage_mean=np.mean(vp, axis=0), wage_std=np.std(vp, axis=0),
                              searching=args.searching,
                              position_encode=args.position_encode,
                              auto_correlation=args.auto_correlation,
                              forget=args.forget,
                              T_train = T_train
                              )
    elif args.pricing_alg == 'SAC_CNN':
        controller = Platform(td, tt, args.num_zone, actor_lr=args.actor_lr, critic_lr=args.critic_lr, option='SAC_CNN',
                              permutation=permutation, permutation2=permutation2, kernel_size=args.kernel_size,
                              stride=args.stride, pooling=args.pooling, device=args.device, writer=writer,
                              od_permutation=args.od_permutation, update_freq=args.frequency,
                              veh_num=len(vp), \
                              demand_mean=np.mean(dd_train, axis=0), demand_std=np.std(dd_train, axis=0),
                              wage_mean=np.mean(vp, axis=0), wage_std=np.std(vp, axis = 0),
                              searching=args.searching,
                              position_encode=args.position_encode,
                              auto_correlation=args.auto_correlation,
                              forget=args.forget,
                              T_train = T_train
                              )
    elif args.pricing_alg == 'SAC_MLP':
        controller = Platform(td, tt, args.num_zone, actor_lr=args.actor_lr, critic_lr=args.critic_lr, option='SAC_MLP',
                              device=args.device,
                              writer=writer, update_freq=args.frequency, veh_num=len(vp),
                              demand_mean=np.mean(dd_train, axis=0), demand_std=np.std(dd_train, axis=0),
                              wage_mean=np.mean(vp, axis=0), wage_std=np.std(vp, axis=0),
                              searching=args.searching,
                              position_encode=args.position_encode,
                              auto_correlation=args.auto_correlation,
                              forget=args.forget,
                              T_train = T_train
                              )
    elif args.pricing_alg == 'TD3_CNN':
        controller = Platform(td, tt, args.num_zone, actor_lr=args.actor_lr, critic_lr=args.critic_lr, option='TD3_CNN',
                              permutation=permutation, permutation2=permutation2, kernel_size=args.kernel_size,
                              stride=args.stride, pooling = args.pooling, device=args.device, writer=writer,
                              od_permutation=args.od_permutation, update_freq = args.frequency, veh_num = len(vp),\
                              demand_mean = np.mean(dd_train, axis = 0), demand_std = np.std(dd_train, axis = 0),
                              wage_mean=np.mean(vp, axis=0), wage_std=np.std(vp, axis=0),
                              searching = args.searching, policy_delay = args.policy_delay,
                              position_encode=args.position_encode,
                              auto_correlation=args.auto_correlation,
                              forget=args.forget,
                              T_train = T_train
                              )
    else:
        controller = Platform(td, tt, args.num_zone, actor_lr=args.actor_lr, critic_lr=args.critic_lr, device=args.device,
                              writer=writer, update_freq = args.frequency, veh_num= len(vp),
                              demand_mean = np.mean(dd_train, axis = 0), demand_std = np.std(dd_train, axis = 0),
                              wage_mean=np.mean(vp, axis=0), wage_std=np.std(vp, axis=0),
                              searching = args.searching, policy_delay = args.policy_delay,
                              position_encode=args.position_encode,
                              auto_correlation=args.auto_correlation,
                              forget=args.forget,
                              T_train = T_train,
                              guidance_dim = guidance_dim
                              )

    env = Environment(td, tt, vp, args.num_zone, frequency=args.frequency)


    _chunk = 200
    baseline = np.empty(T_train)
    for _i in range(0, T_train, _chunk):
        _sl = dd_train[_i:_i + _chunk]  
        baseline[_i:_i + _chunk] = (1 - env.delta) * (
            env.alpha0 * np.sum(_sl, axis=(1, 2)) +
            env.alpha1 * np.einsum('tij,ij->t', _sl, td) +
            env.beta1 * np.einsum('tij,ij->t', _sl, tt)
        )

    if args.competitive:

        num_plat = args.num_platforms
        env_comp = Environment(td, tt, vp, args.num_zone, frequency=args.frequency,
                               num_platforms=num_plat, fleet_fractions=args.fleet_fractions, competitive=True)

        # compute per-platform fleet sizes to match how the environment split the vehicle profiles
        n = len(vp)
        if args.fleet_fractions is not None:
            fleet_sizes = [int(round(f * n)) for f in args.fleet_fractions]
            fleet_sizes[-1] = n - sum(fleet_sizes[:-1])
        else:
            fleet_size = n // num_plat
            fleet_sizes = [fleet_size] * (num_plat - 1) + [n - fleet_size * (num_plat - 1)]

        # each platform's reward baseline is its proportional share of the market-level baseline
        baselines = [baseline * (fleet_sizes[k] / n) for k in range(num_plat)]

        match_caps = args.match_time_caps if args.match_time_caps is not None else [15] * num_plat
        controller.match_time_cap = match_caps[0]
        controller.veh_num = fleet_sizes[0]

        controllers = [controller]
        for k in range(1, num_plat):
            controllers.append(Platform(td, tt, args.num_zone, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                                        device=args.device, writer=writer, update_freq=args.frequency,
                                        veh_num=fleet_sizes[k],
                                        demand_mean=np.mean(dd_train, axis=0), demand_std=np.std(dd_train, axis=0),
                                        wage_mean=np.mean(vp, axis=0), wage_std=np.std(vp, axis=0),
                                        searching=args.searching, policy_delay=args.policy_delay,
                                        position_encode=args.position_encode,
                                        auto_correlation=args.auto_correlation,
                                        forget=args.forget, T_train=T_train,
                                        match_time_cap=match_caps[k], guidance_dim=guidance_dim))

        regulator_lambdas = {'rs': args.lambda_rs, 'served': args.lambda_served, 'tv': args.lambda_tv,
                             'sd': args.lambda_sd, 'gini_rs': args.lambda_gini}
        fed_server = None
        regulator = None
        if args.architecture in ['frl', 'pfrl']:
            fed_server = FederatedServer(regulator_lambdas, tau=args.tau_frl, share_beta=args.share_beta)
        elif args.architecture == 'hrl':
            regulator = Regulator(regulator_lambdas, guidance_dim=args.guidance_dim)


        comp_folder = "results/" + scenario_tag[:-1] + "/competitive_" + args.architecture
        comp_res_folder = comp_folder + "/" + settings_tag + "/" + hyperp_tag
        comp_model_folder = comp_folder + "/models/" + settings_tag + "/" + hyperp_tag
        os.makedirs(comp_res_folder + "/", exist_ok=True)
        os.makedirs(comp_model_folder + "/", exist_ok=True)
        args.store_res_folder = comp_res_folder
        args.store_model_folder = comp_model_folder

        if args.mode == 'all':
            if args.verbose:
                veh_str = ", ".join(str(fleet_sizes[k]) + " veh (P" + str(k) + ")" for k in range(num_plat))
                print("Competitive training: " + veh_str)
            train_competitive(env_comp, controllers, dd_train, T_train,
                              baselines, fleet_sizes, writer, args,
                              fed_server=fed_server, regulator=regulator)
        else:
            for k in range(num_plat):
                controllers[k].pricer.load_weights(comp_model_folder + "/" + str(k) + "/", args.n_epochs)
                if args.device == 'cuda':
                    controllers[k].pricer.cuda()
            if regulator is not None:
                regulator.load(comp_model_folder + "/regulator_" + str(args.n_epochs) + ".pkl")
            for k in range(1, 6):
                test_competitive(env_comp, controllers, dd_train, dd_test,
                                 T_test, fleet_sizes, args, k, regulator=regulator)

    else:

        if args.mode == 'all':
            if args.resume > 0:
                if args.verbose:
                    print("Load model")
                print("Resume training from epoch " + str(args.resume))
                controller.pricer.load_weights(
                    args.store_model_folder + "/", args.resume)
                controller.buffer.load(args.store_model_folder + "/", args.resume)
                if args.device == 'cuda':
                    controller.pricer.cuda()

                # update the searching variance
                for i in range(args.resume):
                    controller.decay_searching_variance()

            train(env, controller, dd_train, T_train, baseline, writer, args)

        else:
            # load the model
            if args.verbose:
                print("Load model")
            if args.pricing_alg.startswith('TD3'):
                controller.pricer.load_weights(
                    args.store_model_folder + "/", args.n_epochs)
                if args.device == 'cuda':
                    controller.pricer.cuda()
            if args.pricing_alg.startswith('PPO'):
                controller.pricer.load_weights(
                    args.store_model_folder + "/", args.n_epochs)
                controller.pricer.set_action_std(controller.min_action_std)
                if args.device == 'cuda':
                    controller.pricer.cuda()
            if args.pricing_alg.startswith('SAC'):
                controller.pricer.load_weights(
                    args.store_model_folder + "/", args.n_epochs)
                controller.pricer.set_action_std(controller.min_action_std)
                if args.device == 'cuda':
                    controller.pricer.cuda()
            # run exp
            for k in range(1, 6):
                test(env, controller, dd_train, dd_test, T_test, args, k)

    # collect result
    if args.mode == 'all':
        writer.close()