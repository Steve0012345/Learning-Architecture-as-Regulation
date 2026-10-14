import os
import copy
import numpy as np
import pandas as pd
import torch
from demand import split_demand
from metrics import consumer_surplus, market_snapshot, MarketRecord
from federation import clone_state
from plot import plot_competitive

def price_sensitivity(pricing_multipliers, mode = False):
    # A monotonic decreasing function that describe the price elasticity
    if mode:
        return max(2.0 - pricing_multipliers, 0)
    else:
        return 1.0 / pricing_multipliers


def pool_demand(dd_raw, prices, args):

    if getattr(args, 'no_outside_option', False):
        return dd_raw
    mean_p = np.mean(np.stack(prices), axis=0)
    dd_elastic = np.floor(dd_raw * price_sensitivity(mean_p[None, :, None], args.pricing_mode))
    dd_elastic += np.random.binomial(1, (
        dd_raw * price_sensitivity(mean_p[None, :, None], args.pricing_mode) - dd_elastic))
    return dd_elastic

def train(env, controller, dd_train, T_train, baseline, writer, args):
    thrng = torch.Generator("cpu")
    # print(np.sum(baseline))
    controller.pricer.initialize(thrng)
    if args.device == 'cuda':
        controller.pricer.cuda()
    if args.verbose:
        print("Start training")

    res = []  # pd.DataFrame(columns = 'epoch', 't', 'policy', 'profit', 'welfare')

    # one day to warm up to obtain the initial active vehicle distribution
    for t in range(max(T_train-1440, 0), T_train):
        # print(t)
        temp_cost, temp_schedule = controller.batch_matching(np.sum(env.pass_count, axis=(0, 2)),
                                                             env.veh_count)
        ongoing_veh = env.get_ongoing_veh()
        if t % args.frequency == 0:
            if args.pricing_alg == 'dummy':
                temp_p = controller.dummy_price()
            elif args.pricing_alg == 'equilibrium':
                temp_p = controller.equilibrium_price(np.sum(env.pass_count, axis=(0)), env.active_veh,
                                                      [dd_train[(t + i) % T_train, :, :] for i in range(10)], args.data_folder)
            else:
                temp_p = controller.update_price(env.pass_count, env.veh_count, \
                                             ongoing_veh, env.avg_profit, env.zone_profit,
                                             env.pricing_multipliers[0:max(5 // args.frequency, 1), :],
                                             t, 'train',
                                             od_permutation=args.od_permutation,
                                             policy_constr=args.policy_constr,
                                             last_pricing=env.pricing_multipliers[0, :])
            dd_train_ = np.zeros(dd_train[t:(t + args.frequency), :, :].shape)
            ind = dd_train[t:(t + args.frequency), :, :] > 0
            dd_train_[ind] = np.random.poisson(dd_train[t:(t + args.frequency), :, :][ind])
            temp_demands = np.floor(dd_train_ * price_sensitivity(temp_p[None, :, None], args.pricing_mode))
            temp_demands += np.random.binomial(1, (
                        dd_train_ * price_sensitivity(temp_p[None, :, None], args.pricing_mode) - temp_demands))
        temp_demand = temp_demands[t % args.frequency]
        env.step(temp_demand, temp_schedule, temp_p)

    for e in range(args.resume):
        controller.decay_searching_variance()

    for e in range(args.resume, args.n_epochs):
        total_time_step = 0
        total_profit = 0
        total_reposition = 0
        total_expense = 0
        total_served_pass = 0
        total_left_pass = 0

        total_occu_mile = 0
        total_occu_minute = 0
        total_empty_mile = 0
        total_empty_minute = 0

        temp_reward = 0
        temp_demands = np.zeros((args.frequency, args.num_zone))

        if args.verbose:
            print("Epoch " + str(e))
        for t in range(T_train):
            with torch.no_grad():
                temp_cost, temp_schedule = controller.batch_matching(np.sum(env.pass_count, axis=(0, 2)),
                                                                     env.veh_count)
                if t % args.frequency == 0:
                    if args.pricing_alg == 'dummy':
                        temp_p = controller.dummy_price()
                    elif args.pricing_alg == 'equilibrium':
                        temp_p = controller.equilibrium_price(np.sum(env.pass_count, axis=(0)), env.active_veh,
                                                              [dd_train[(t + i) % T_train, :, :] for i in range(10)], args.data_folder)
                    else:
                        ongoing_veh = env.get_ongoing_veh()
                        controller.add_memory(env.pass_count, env.veh_count, ongoing_veh, env.avg_profit, env.zone_profit,\
                                              env.pricing_multipliers[0:max(5 // args.frequency, 1), :], temp_reward, t,
                                              args.od_permutation)
                        temp_p = controller.update_price(env.pass_count, env.veh_count, \
                                                         ongoing_veh, env.avg_profit, env.zone_profit,
                                                         env.pricing_multipliers[0:max(5 // args.frequency, 1), :],
                                                         t, 'train',
                                                         od_permutation=args.od_permutation,
                                                         policy_constr=args.policy_constr,
                                                         last_pricing=env.pricing_multipliers[0, :])

                    dd_train_ = np.zeros(dd_train[t:(t + args.frequency), :, :].shape)
                    ind = dd_train[t:(t + args.frequency), :, :] > 0
                    dd_train_[ind] = np.random.poisson(dd_train[t:(t + args.frequency), :, :][ind])
                    temp_demands = np.floor(dd_train_ * price_sensitivity(temp_p[None, :, None], args.pricing_mode))
                    temp_demands += np.random.binomial(1, (
                            dd_train_ * price_sensitivity(temp_p[None, :, None], args.pricing_mode) - temp_demands))
                    temp_reward = 0  # refresh the temp_reward

                temp_demand = temp_demands[t % args.frequency]
                temp_profit, temp_expense, temp_served_pass, temp_left_pass, occu_mile, occu_minute, empty_mile, empty_minute = env.step(temp_demand, temp_schedule, temp_p)
                temp_reward += temp_profit - temp_expense - baseline[t]
            # train the model
            # if t == 0 and e == 1:
            #     controller.train_pricing_value(1000, args.batch_size)
            if args.pricing_alg.startswith('TD3') and (e * T_train + t + 1) % (args.frequency) == 0: 
                controller.train_pricing_policy(((e - 1) * T_train + t + 1) //args.frequency, \
                                                T_train // args.frequency, args.batch_size, args.update_frequency)
            elif args.pricing_alg.startswith('PPO') and (e * T_train + t + 1) % (args.frequency * args.update_frequency) == 0:
                controller.train_pricing_policy(50, \
                                                args.update_frequency, args.batch_size)
            elif args.pricing_alg.startswith('SAC') and (e * T_train + t + 1) % (args.frequency * args.update_frequency) == 0:
                controller.train_pricing_policy(((e - 1) * T_train + t + 1) //(args.frequency * args.update_frequency), \
                                                args.update_frequency, args.batch_size)

            total_time_step += 1
            total_profit += temp_profit
            total_reposition += temp_cost
            total_expense += temp_expense
            total_served_pass += temp_served_pass
            total_left_pass += temp_left_pass
            total_occu_mile += occu_mile
            total_occu_minute += occu_minute
            total_empty_mile += empty_mile
            total_empty_minute += empty_minute

            if t % 120 == 0:
                writer.flush()
                if args.verbose:
                    # print(np.diag(env.veh_count))
                    print("Time step: " + str(t) + ", income: " + str(temp_profit) + \
                          ", expense: " + str(temp_expense) + \
                          ", active veh: " + str(env.active_veh) + \
                          ", served_pass: " + str(temp_served_pass) + \
                          ", left_pass: " + str(temp_left_pass) + \
                          ", new demand: " + str(np.sum(temp_demand)))

        controller.decay_searching_variance()
        print("Total time step: " + str(total_time_step) + ", reward: " + str(
            total_profit - total_expense) + ", profit: " + str(total_profit) + \
              ", expense: " + str(total_expense) + ", served_pass: " + str(
            total_served_pass) + ", left_pass: " + str(total_left_pass))
        res.append([e, total_profit, total_expense, total_served_pass, total_left_pass,\
                    total_occu_mile, total_occu_minute, total_empty_mile, total_empty_minute])
        if env.active_veh == 0: # run into an absorbing stage in which no drive wants to work, need to reset the env
            env.reset()
            controller.buffer.clear()
        writer.add_scalar("Reward/train", total_profit - total_expense, e)
        writer.add_scalar("Served_passenger/train", total_served_pass, e)
        writer.add_scalar("Left_passenger/train", total_left_pass, e)

    res = pd.DataFrame(res, columns=['epoch', 'income', 'expense', 'served', 'left', 'occu_mile', 'occu_minute', 'empty_mile', 'empty_minute'])
    res.to_csv(
        args.store_res_folder+"/"+"train_log_"+str(args.n_epochs)+".csv",
        index=None)

    # save the model in the end
    controller.pricer.save_model(
        args.store_model_folder + "/", args.n_epochs)
    controller.buffer.save(args.store_model_folder + "/", args.n_epochs)

    # store the model
    # print("Store model")
    # if (not os.path.isdir(
    #         'models/' + args.name+'/'+ args.data_folder + args.pricing_alg + permutation_tag + "/" + str(
    #             args.batch_size) + "_" + str(
    #             args.actor_lr) + "_" + str(args.critic_lr) + str(args.epoch) + "/")):
    #     os.makedirs('models/' +  args.name+'/'+args.data_folder + args.pricing_alg + permutation_tag + "/" + str(
    #         args.batch_size) + "_" + str(
    #         args.actor_lr) + "_" + str(args.critic_lr) + str(args.epoch) + "/")
    # controller.pricer.save_model(
    #     'models/' +args.name+'/' + args.data_folder + args.pricing_alg + permutation_tag + "/" + str(args.batch_size) + "_" + str(
    #         args.actor_lr) + "_" + str(args.critic_lr) + str(args.epoch) + "/")

def test(env, controller, dd_train, dd_test, T_test, args, k):
    if args.verbose:
        print("Start testing")
    res = []
    controller.pricer.eval()
    total_time_step = 0
    total_profit = 0
    total_reposition = 0
    total_expense = 0
    total_served_pass = 0
    total_left_pass = 0
    temp_demands = np.zeros((args.frequency, args.num_zone))

    for t in range(max(T_test-1440, 0), T_test):
        print(t)
        temp_cost, temp_schedule = controller.batch_matching(np.sum(env.pass_count, axis=(0, 2)),
                                                             env.veh_count)
        # temp_p = controller.dummy_price()
        if t % args.frequency == 0:
            if args.pricing_alg == 'dummy':
                # no need to train
                temp_p = controller.dummy_price()
            elif args.pricing_alg == 'equilibrium':
                temp_p = controller.equilibrium_price(np.sum(env.pass_count, axis=(0)), env.active_veh ,
                                                          [dd_train[(t + i) % T_test, :, :] for i in range(10)], args.data_folder)
            else:
                ongoing_veh = env.get_ongoing_veh()
                temp_p = controller.update_price(env.pass_count, env.veh_count, \
                                                 ongoing_veh, env.avg_profit, env.zone_profit,
                                                 env.pricing_multipliers[0:max(5 // args.frequency, 1), :],
                                                 t, 'train',
                                                 od_permutation=args.od_permutation,
                                                 policy_constr=args.policy_constr,
                                                 last_pricing=env.pricing_multipliers[0, :])
            dd_train_ = np.zeros(dd_train[t:(t + args.frequency), :, :].shape)
            ind = dd_train[t:(t + args.frequency), :, :] > 0
            dd_train_[ind] = np.random.poisson(dd_train[t:(t + args.frequency), :, :][ind])
            temp_demands = np.floor(dd_train_ * price_sensitivity(temp_p[None, :, None], args.pricing_mode))
            temp_demands += np.random.binomial(1, (
                    dd_train_ * price_sensitivity(temp_p[None, :, None], args.pricing_mode) - temp_demands))
        temp_demand = temp_demands[t % args.frequency]
        dd_train_ = np.random.poisson(dd_train[t, :, :]).astype(int)
        temp_demand = dd_train_
        # temp_demand += np.random.binomial(1, (dd_train_ / temp_p[:, None] - temp_demand))
        env.step(temp_demand, temp_schedule, temp_p)

    for t in range(T_test):
        print(t)
        temp_cost, temp_schedule = controller.batch_matching(np.sum(env.pass_count, axis=(0, 2)),
                                                             env.veh_count)
        if t % args.frequency == 0:
            if args.pricing_alg == 'dummy':
                temp_p = controller.dummy_price()
            elif args.pricing_alg == 'equilibrium':
                temp_p = controller.equilibrium_price(np.sum(env.pass_count, axis=(0)),env.active_veh ,
                                                          [dd_train[(t + i) % T_test, :, :] for i in range(10)], args.data_folder)
            else:
                ongoing_veh = env.get_ongoing_veh()
                temp_p = controller.update_price(env.pass_count, env.veh_count, \
                                                         ongoing_veh, env.avg_profit, env.zone_profit,
                                                         env.pricing_multipliers[0:max(5 // args.frequency, 1), :],
                                                         t, 'test',
                                                         od_permutation=args.od_permutation,
                                                         policy_constr=args.policy_constr,
                                                         last_pricing=env.pricing_multipliers[0, :])
            temp_demands = np.floor(dd_test[t:(t + args.frequency), :, :] * price_sensitivity(temp_p[None, :, None], args.pricing_mode))
            temp_demands += np.random.binomial(1, (
                    dd_test[t:(t + args.frequency), :, :] * price_sensitivity(temp_p[None, :, None], args.pricing_mode) - temp_demands))

        temp_demand = temp_demands[t % args.frequency]
        temp_profit, temp_expense, temp_served_pass, temp_left_pass, occu_mile, occu_minute, empty_mile, empty_minute = env.step(temp_demand, temp_schedule, temp_p)
        total_time_step += 1
        total_profit += temp_profit
        total_reposition += temp_cost
        total_expense += temp_expense
        total_served_pass += temp_served_pass
        total_left_pass += temp_left_pass

        # writer.add_scalar("Reward/test", temp_profit + temp_welfare, t)
        # writer.add_scalar("Served_passenger/test", temp_served_pass, t)
        # writer.add_scalar("Left_passenger/test", temp_left_pass, t)

        res.append([t, temp_p, temp_profit, temp_expense, env.active_veh, temp_served_pass, temp_left_pass, occu_mile, occu_minute, empty_mile, empty_minute])
        if t % 120 == 0:
            # writer.flush()
            if args.verbose:
                # print(np.diag(env.veh_count))
                print("Time step: " + str(t) + ", profit: " + str(temp_profit) + ", expense: " + str(
                    temp_expense) + \
                      ", active veh: " + str(env.active_veh) + \
                      ", served_pass: " + str(temp_served_pass) + \
                      ", left_pass: " + str(temp_left_pass) + \
                      ", new demand: " + str(np.sum(temp_demand)))
    print("Total time step: " + str(total_time_step) + ", reward: " + str(
        total_profit - total_expense) + ", profit: " + str(total_profit) + \
          ", expense: " + str(total_expense) + ", served_pass: " + str(
        total_served_pass) + ", left_pass: " + str(
        total_left_pass))
    res = pd.DataFrame(res, columns=['t', 'policy', 'profit', 'expense', 'act_veh', 'served', 'left', 'occu_mile', 'occu_minute', 'empty_mile', 'empty_minute'])
    res.to_csv(
        args.store_res_folder + "/" + f"test_log_{k}.csv",
        index=None)
    env.reset()


def train_competitive(env, controllers, dd_train, T_train,
                      baselines, total_vehs, writer, args, fed_server=None, regulator=None):
    """Multi-platform competitive training loop.

    All controllers optimize their own profit independently.
    Demand is split at each step using a logit model that sees all platforms'
    prices and availability simultaneously and this is what creates the competition.

    controllers : list of Platform, one per platform
    baselines   : list of per-step baseline arrays, one per platform
    total_vehs  : list of fleet sizes, one per platform
    fed_server  : FederatedServer for regulator-weighted / personalized FRL, None otherwise
    regulator   : Regulator for HRL, None otherwise. With both None this is Local RL.
    """
    num_plat = len(controllers)
    thrng = torch.Generator("cpu")
    for ctrl in controllers:
        ctrl.pricer.initialize(thrng)
        if args.device == 'cuda':
            ctrl.pricer.cuda()
    if args.verbose:
        print("Start competitive training")
    if fed_server is not None:
        fed_server.snapshot(controllers)

    res = []
    # HRL interval statistics carry over epoch boundaries, the regulator's clock does not reset
    interval_market = MarketRecord()

    prices = [ctrl.dummy_price() for ctrl in controllers]
    demand_blocks = [np.zeros((args.frequency, args.num_zone, args.num_zone)) for _ in range(num_plat)]

    for t in range(max(T_train - 1440, 0), T_train):
        schedules = [controllers[k].batch_matching(
            np.sum(env.pass_count_plat[k], axis=(0, 2)), env.veh_count_plat[k])[1]
            for k in range(num_plat)]

        if t % args.frequency == 0:
            prices = [ctrl.dummy_price() for ctrl in controllers]
            dd_raw = np.zeros(dd_train[t:(t + args.frequency), :, :].shape)
            ind = dd_train[t:(t + args.frequency), :, :] > 0
            dd_raw[ind] = np.random.poisson(dd_train[t:(t + args.frequency), :, :][ind])
            demand_pool = pool_demand(dd_raw, prices, args)
            demand_blocks = split_demand(demand_pool, prices, env.veh_count_plat, total_vehs,
                                         brand_prefs=getattr(args, 'brand_prefs', None))

        demands = [demand_blocks[k][t % args.frequency] for k in range(num_plat)]
        env.step_competitive(demands, schedules, prices)

    for e in range(args.resume):
        for ctrl in controllers:
            ctrl.decay_searching_variance()

    for e in range(args.resume, args.n_epochs):
        ep_profit   = [0.0] * num_plat
        ep_expense  = [0.0] * num_plat
        ep_served   = [0]   * num_plat
        ep_left     = [0]   * num_plat
        ep_cs       = [0.0] * num_plat
        ep_empty_mi = [0.0] * num_plat
        ep_prices   = [[]   for _ in range(num_plat)]
        temp_reward = [0.0] * num_plat
        demand_blocks = [np.zeros((args.frequency, args.num_zone, args.num_zone)) for _ in range(num_plat)]
        ep_market   = MarketRecord()

        print("\n=== Epoch " + str(e + 1) + " / " + str(args.n_epochs) + " ===")

        for t in range(T_train):
            with torch.no_grad():
                schedules = [controllers[k].batch_matching(
                    np.sum(env.pass_count_plat[k], axis=(0, 2)), env.veh_count_plat[k])[1]
                    for k in range(num_plat)]

                if t % args.frequency == 0:
                    if regulator is not None and t % args.hrl_interval == 0:
                        guidance = regulator.observe(interval_market.summary(), explore=True)
                        for ctrl in controllers:
                            ctrl.guidance = guidance
                        interval_market = MarketRecord()

                    ongoing = [env.get_ongoing_veh_plat(k) for k in range(num_plat)]

                    # store transitions before updating prices
                    for k in range(num_plat):
                        controllers[k].add_memory(
                            env.pass_count_plat[k], env.veh_count_plat[k], ongoing[k],
                            env.avg_profit_plat[k], env.zone_profit_plat[k],
                            env.pricing_mults_plat[k][0:max(5 // args.frequency, 1), :],
                            temp_reward[k], t, args.od_permutation)

                    if args.pricing_alg == 'dummy':
                        prices = [ctrl.dummy_price() for ctrl in controllers]
                    elif args.pricing_alg == 'equilibrium':
                        prices = [controllers[k].equilibrium_price(
                            np.sum(env.pass_count_plat[k], axis=0), env.active_veh_plat[k],
                            [dd_train[(t + i) % T_train, :, :] for i in range(10)],
                            args.data_folder) for k in range(num_plat)]
                    else:
                        prices = [controllers[k].update_price(
                            env.pass_count_plat[k], env.veh_count_plat[k], ongoing[k],
                            env.avg_profit_plat[k], env.zone_profit_plat[k],
                            env.pricing_mults_plat[k][0:max(5 // args.frequency, 1), :],
                            t, 'train',
                            od_permutation=args.od_permutation,
                            policy_constr=args.policy_constr,
                            last_pricing=env.pricing_mults_plat[k][0, :])
                            for k in range(num_plat)]

                    dd_raw = np.zeros(dd_train[t:(t + args.frequency), :, :].shape)
                    ind = dd_train[t:(t + args.frequency), :, :] > 0
                    dd_raw[ind] = np.random.poisson(dd_train[t:(t + args.frequency), :, :][ind])
                    demand_pool = pool_demand(dd_raw, prices, args)
                    # rider-side outcomes are measured on the pre-split pool and pre-decision availability
                    rs_zone, pbar_zone = market_snapshot(prices, env.veh_count_plat, total_vehs,
                                                         np.sum(demand_pool, axis=(0, 2)),
                                                         brand_prefs=getattr(args, 'brand_prefs', None))
                    ep_market.record(rs_zone, pbar_zone)
                    interval_market.record(rs_zone, pbar_zone)
                    demand_blocks = split_demand(demand_pool, prices, env.veh_count_plat, total_vehs,
                                                 brand_prefs=getattr(args, 'brand_prefs', None))

                    temp_reward = [0.0] * num_plat
                    for k in range(num_plat):
                        ep_prices[k].append(prices[k].copy())

                    mu0_zone = np.sum(dd_train[t:t + args.frequency], axis=(0, 2))
                    demand_zones = [np.sum(demand_blocks[k], axis=(0, 2)) for k in range(num_plat)]
                    total_zone = np.maximum(sum(demand_zones), 1e-6)
                    for k in range(num_plat):
                        ep_cs[k] += consumer_surplus(mu0_zone, prices[k], demand_zones[k] / total_zone)

                demands = [demand_blocks[k][t % args.frequency] for k in range(num_plat)]
                step_results = env.step_competitive(demands, schedules, prices)

                for k in range(num_plat):
                    profit_k, expense_k, served_k, left_k, _, _, empty_mi_k, _ = step_results[k]
                    temp_reward[k] += profit_k - expense_k - baselines[k][t]
                    ep_profit[k]   += profit_k
                    ep_expense[k]  += expense_k
                    ep_served[k]   += served_k
                    ep_left[k]     += left_k
                    ep_empty_mi[k] += empty_mi_k
                    ep_market.served += served_k
                    interval_market.served += served_k


            step_count = e * T_train + t + 1
            if args.pricing_alg.startswith('TD3') and step_count % args.frequency == 0:
                for k in range(num_plat):
                    controllers[k].train_pricing_policy(
                        ((e - 1) * T_train + t + 1) // args.frequency,
                        T_train // args.frequency, args.batch_size, args.update_frequency)
            elif args.pricing_alg.startswith('PPO') and step_count % (args.frequency * args.update_frequency) == 0:
                for k in range(num_plat):
                    controllers[k].train_pricing_policy(50, args.update_frequency, args.batch_size)
            elif args.pricing_alg.startswith('SAC') and step_count % (args.frequency * args.update_frequency) == 0:
                for k in range(num_plat):
                    controllers[k].train_pricing_policy(
                        ((e - 1) * T_train + t + 1) // (args.frequency * args.update_frequency),
                        args.update_frequency, args.batch_size)


            if (fed_server is not None and step_count % args.fed_interval == 0
                    and len(controllers[0].buffer) >= T_train // args.frequency):
                rng_state = np.random.get_state()
                outcomes = []
                for k in range(num_plat):

                    np.random.set_state(rng_state)
                    outcomes.append(validation_rollout(env, controllers, k, fed_server.reference_actors,
                                                       dd_train, (t + 1) % T_train, total_vehs, args))
                # validation must not shift the training stream relative to the round it interrupts
                np.random.set_state(rng_state)
                weights = fed_server.aggregate(controllers, outcomes)
                print("FRL round " + str(len(fed_server.history)) + " at t=" + str(t + 1) + " | weights=" +
                      str(np.round(weights, 3)))

            if t % 120 == 0:
                writer.flush()
                if args.verbose:
                    parts = ["t=" + str(t)]
                    for k in range(num_plat):
                        profit_k, expense_k, served_k = step_results[k][0], step_results[k][1], step_results[k][2]
                        parts.append("P" + str(k) + ": profit=" + str(round(profit_k, 1)) +
                                     " expense=" + str(round(expense_k, 1)) +
                                     " served=" + str(served_k) +
                                     " veh=" + str(env.active_veh_plat[k]))
                    print(" | ".join(parts))

        for ctrl in controllers:
            ctrl.decay_searching_variance()

        mean_prices = [float(np.mean(ep_prices[k])) if ep_prices[k] else 1.0 for k in range(num_plat)]
        std_prices  = [float(np.std(ep_prices[k]))  if ep_prices[k] else 0.0 for k in range(num_plat)]

        summary_parts = []
        for k in range(num_plat):
            summary_parts.append(
                "P" + str(k) + ": net=" + str(round(ep_profit[k] - ep_expense[k], 1)) +
                " served=" + str(ep_served[k]) +
                " veh=" + str(env.active_veh_plat[k]))
        print("Epoch " + str(e + 1) + "/" + str(args.n_epochs) + " done | " +
              "  ".join(summary_parts) + "  | CS_total=" + str(round(sum(ep_cs), 1)))

        row = [e]
        for k in range(num_plat):
            row += [ep_profit[k], ep_expense[k], ep_served[k], ep_left[k]]
        for k in range(num_plat):
            row += [env.active_veh_plat[k], mean_prices[k], std_prices[k], ep_cs[k], ep_empty_mi[k]]
        market_outcome = ep_market.summary()
        row += [market_outcome['rs'], market_outcome['served'], market_outcome['tv'],
                market_outcome['sd'], market_outcome['gini_rs']]
        res.append(row)
        print("Market | RS=" + str(round(market_outcome['rs'], 1)) + " S=" + str(int(market_outcome['served'])) +
              " TV=" + str(round(market_outcome['tv'], 4)) + " SD=" + str(round(market_outcome['sd'], 4)) +
              " G_RS=" + str(round(market_outcome['gini_rs'], 4)))

        # reset if all platforms have lost their entire active fleet (absorbing state)
        if all(env.active_veh_plat[k] == 0 for k in range(num_plat)):
            env.reset_competitive()
            for ctrl in controllers:
                ctrl.buffer.clear()

        for k in range(num_plat):
            writer.add_scalar("Reward_P" + str(k) + "/train", ep_profit[k] - ep_expense[k], e)
        writer.add_scalar("CS_total/train", sum(ep_cs), e)

    columns = ['epoch']
    for k in range(num_plat):
        columns += ['profit_' + str(k), 'expense_' + str(k), 'served_' + str(k), 'left_' + str(k)]
    for k in range(num_plat):
        columns += ['active_veh_' + str(k), 'mean_price_' + str(k), 'price_std_' + str(k),
                    'cs_' + str(k), 'empty_mile_' + str(k)]
    columns += ['rs_index', 'served_total', 'temporal_vol', 'spatial_disp', 'gini_rs']

    res_df = pd.DataFrame(res, columns=columns)
    csv_path = args.store_res_folder + "/train_log_" + str(args.n_epochs) + ".csv"
    res_df.to_csv(csv_path, index=None)

    # save each platform's model in a numbered subfolder
    for k in range(num_plat):
        os.makedirs(args.store_model_folder + "/" + str(k) + "/", exist_ok=True)
        controllers[k].pricer.save_model(args.store_model_folder + "/" + str(k) + "/", args.n_epochs)
        controllers[k].buffer.save(args.store_model_folder + "/" + str(k) + "/", args.n_epochs)
    if fed_server is not None:
        fed_server.save_history(args.store_res_folder + "/fed_rounds_" + str(args.n_epochs) + ".csv")
    if regulator is not None:
        regulator.save(args.store_model_folder + "/regulator_" + str(args.n_epochs) + ".pkl")
        pd.DataFrame(regulator.history).to_csv(args.store_res_folder + "/regulator_intervals_" +
                                               str(args.n_epochs) + ".csv", index=None)

    args_summary = {
        'pricing_algorithm': args.pricing_alg,
        'num_platforms': num_plat,
        'epochs': args.n_epochs,
        'frequency_minutes': args.frequency,
        'batch_size': args.batch_size,
        'actor_lr': args.actor_lr,
        'critic_lr': args.critic_lr,
    }
    plot_competitive(csv_path, args.store_res_folder, num_plat=num_plat, args_dict=args_summary)


def validation_rollout(env, controllers, candidate, reference_actors, dd_train, t_start, total_vehs, args):
    """Run a validation rollout for one platform in a multi-platform competitive setting."""
    num_plat = len(controllers)
    T_train = dd_train.shape[0]
    market = copy.deepcopy(env)
    trained_actors = [clone_state(ctrl.pricer.actor) for ctrl in controllers]

    last_actions = [ctrl.prices.clone() for ctrl in controllers]
    for k in range(num_plat):
        if k != candidate:
            controllers[k].pricer.actor.load_state_dict(reference_actors[k])

    record = MarketRecord()
    with torch.no_grad():
        for i in range(args.val_steps):
            t = (t_start + i) % T_train
            schedules = [controllers[k].batch_matching(
                np.sum(market.pass_count_plat[k], axis=(0, 2)), market.veh_count_plat[k])[1]
                for k in range(num_plat)]
            if i % args.frequency == 0:
                prices = [controllers[k].update_price(
                    market.pass_count_plat[k], market.veh_count_plat[k], market.get_ongoing_veh_plat(k),
                    market.avg_profit_plat[k], market.zone_profit_plat[k],
                    market.pricing_mults_plat[k][0:max(5 // args.frequency, 1), :],
                    t, 'test', od_permutation=args.od_permutation,
                    policy_constr=args.policy_constr,
                    last_pricing=market.pricing_mults_plat[k][0, :])
                    for k in range(num_plat)]
                dd_raw = np.zeros(dd_train[t:(t + args.frequency), :, :].shape)
                ind = dd_train[t:(t + args.frequency), :, :] > 0
                dd_raw[ind] = np.random.poisson(dd_train[t:(t + args.frequency), :, :][ind])
                demand_pool = pool_demand(dd_raw, prices, args)
                rs_zone, pbar_zone = market_snapshot(prices, market.veh_count_plat, total_vehs,
                                                     np.sum(demand_pool, axis=(0, 2)),
                                                     brand_prefs=getattr(args, 'brand_prefs', None))
                record.record(rs_zone, pbar_zone)
                demand_blocks = split_demand(demand_pool, prices, market.veh_count_plat, total_vehs,
                                             brand_prefs=getattr(args, 'brand_prefs', None))
            demands = [demand_blocks[k][i % args.frequency] for k in range(num_plat)]
            step_results = market.step_competitive(demands, schedules, prices)
            record.served += sum(step_results[k][2] for k in range(num_plat))

    for k in range(num_plat):
        controllers[k].pricer.actor.load_state_dict(trained_actors[k])
        controllers[k].prices = last_actions[k]
    return record.summary()


def test_competitive(env, controllers, dd_train, dd_test, T_test,
                     total_vehs, args, k, regulator=None):
    """Run one test episode for the multi-platform competitive setting."""
    num_plat = len(controllers)
    if args.verbose:
        print("Start competitive testing, run " + str(k))

    for ctrl in controllers:
        ctrl.pricer.eval()

    res = []
    prices = [ctrl.dummy_price() for ctrl in controllers]
    demand_blocks = [np.zeros((args.frequency, args.num_zone, args.num_zone)) for _ in range(num_plat)]

    # warm-up
    for t in range(max(T_test - 1440, 0), T_test):
        schedules = [controllers[kk].batch_matching(
            np.sum(env.pass_count_plat[kk], axis=(0, 2)), env.veh_count_plat[kk])[1]
            for kk in range(num_plat)]

        if t % args.frequency == 0:
            ongoing = [env.get_ongoing_veh_plat(kk) for kk in range(num_plat)]
            prices = [controllers[kk].update_price(
                env.pass_count_plat[kk], env.veh_count_plat[kk], ongoing[kk],
                env.avg_profit_plat[kk], env.zone_profit_plat[kk],
                env.pricing_mults_plat[kk][0:max(5 // args.frequency, 1), :],
                t, 'test', od_permutation=args.od_permutation,
                policy_constr=args.policy_constr,
                last_pricing=env.pricing_mults_plat[kk][0, :])
                for kk in range(num_plat)]

            dd_raw = np.random.poisson(dd_train[t:(t + args.frequency), :, :])
            demand_pool = pool_demand(dd_raw, prices, args)
            demand_blocks = split_demand(demand_pool, prices, env.veh_count_plat, total_vehs,
                                         brand_prefs=getattr(args, 'brand_prefs', None))

        demands = [demand_blocks[kk][t % args.frequency] for kk in range(num_plat)]
        env.step_competitive(demands, schedules, prices)

    # main test loop
    cs_step = [0.0] * num_plat
    rs_step = 0.0
    test_market = MarketRecord()
    interval_market = MarketRecord()
    for t in range(T_test):
        schedules = [controllers[kk].batch_matching(
            np.sum(env.pass_count_plat[kk], axis=(0, 2)), env.veh_count_plat[kk])[1]
            for kk in range(num_plat)]

        if t % args.frequency == 0:
            if regulator is not None and t % args.hrl_interval == 0:
                # the trained regulator acts on its mean guidance and no longer learns
                guidance = regulator.observe(interval_market.summary(), explore=False)
                for ctrl in controllers:
                    ctrl.guidance = guidance
                interval_market = MarketRecord()

            ongoing = [env.get_ongoing_veh_plat(kk) for kk in range(num_plat)]
            prices = [controllers[kk].update_price(
                env.pass_count_plat[kk], env.veh_count_plat[kk], ongoing[kk],
                env.avg_profit_plat[kk], env.zone_profit_plat[kk],
                env.pricing_mults_plat[kk][0:max(5 // args.frequency, 1), :],
                t, 'test', od_permutation=args.od_permutation,
                policy_constr=args.policy_constr,
                last_pricing=env.pricing_mults_plat[kk][0, :])
                for kk in range(num_plat)]

            dd_raw = np.random.poisson(dd_test[t:(t + args.frequency), :, :])
            demand_pool = pool_demand(dd_raw, prices, args)
            rs_zone, pbar_zone = market_snapshot(prices, env.veh_count_plat, total_vehs,
                                                 np.sum(demand_pool, axis=(0, 2)),
                                                 brand_prefs=getattr(args, 'brand_prefs', None))
            test_market.record(rs_zone, pbar_zone)
            interval_market.record(rs_zone, pbar_zone)
            rs_step = float(np.sum(rs_zone))
            demand_blocks = split_demand(demand_pool, prices, env.veh_count_plat, total_vehs,
                                         brand_prefs=getattr(args, 'brand_prefs', None))

            mu0_zone = np.sum(dd_test[t:t + args.frequency], axis=(0, 2))
            demand_zones = [np.sum(demand_blocks[kk], axis=(0, 2)) for kk in range(num_plat)]
            total_zone = np.maximum(sum(demand_zones), 1e-6)
            cs_step = [consumer_surplus(mu0_zone, prices[kk], demand_zones[kk] / total_zone)
                       for kk in range(num_plat)]

        demands = [demand_blocks[kk][t % args.frequency] for kk in range(num_plat)]
        step_results = env.step_competitive(demands, schedules, prices)

        row = [t]
        for kk in range(num_plat):
            profit_k, expense_k, served_k, left_k = (step_results[kk][0], step_results[kk][1],
                                                      step_results[kk][2], step_results[kk][3])
            row += [round(float(np.mean(prices[kk])), 4),
                    profit_k, expense_k, env.active_veh_plat[kk], served_k, left_k, cs_step[kk]]
            test_market.served += served_k
            interval_market.served += served_k
        row.append(rs_step)
        res.append(row)

        if t % 120 == 0 and args.verbose:
            parts = ["t=" + str(t)]
            for kk in range(num_plat):
                parts.append("P" + str(kk) + " profit=" + str(round(step_results[kk][0], 1)) +
                              " served=" + str(step_results[kk][2]))
            print(" | ".join(parts))

    # each platform contributes 7 columns per row: mean_price, profit, expense, act_veh, served, left, cs
    net_rewards = [sum(r[1 + kk * 7 + 1] - r[1 + kk * 7 + 2] for r in res) for kk in range(num_plat)]
    cs_total = sum(sum(r[1 + kk * 7 + 6] for r in res) for kk in range(num_plat))
    net_str = "  ".join("P" + str(kk) + " net=" + str(round(net_rewards[kk], 1)) for kk in range(num_plat))
    print("Test " + str(k) + " | " + net_str + " | CS_total=" + str(round(cs_total, 1)))

    columns = ['t']
    for kk in range(num_plat):
        columns += ['mean_price_' + str(kk), 'profit_' + str(kk), 'expense_' + str(kk),
                    'act_veh_' + str(kk), 'served_' + str(kk), 'left_' + str(kk), 'cs_' + str(kk)]
    columns.append('rs_index')
    res_df = pd.DataFrame(res, columns=columns)
    res_df.to_csv(args.store_res_folder + "/test_log_" + str(k) + ".csv", index=None)

    # one row per test episode with the Section 7.6 outcomes, tagged with the run configuration
    # so summarize.py can pool seeds without parsing folder names
    market_outcome = test_market.summary()
    episode = {'architecture': args.architecture, 'scenario': args.data_folder, 'num_platforms': num_plat,
               'seed': args.seed, 'name': args.name, 'lambda_rs': args.lambda_rs, 'tau_frl': args.tau_frl,
               'share_beta': args.share_beta, 'hrl_interval': args.hrl_interval,
               'no_outside_option': args.no_outside_option, 'test_episode': k,
               'profit': float(sum(net_rewards))}
    for kk in range(num_plat):
        episode['profit_' + str(kk)] = float(net_rewards[kk])
    episode.update(market_outcome)
    pd.DataFrame([episode]).to_csv(args.store_res_folder + "/test_metrics_" + str(k) + ".csv", index=None)

    np.save(args.store_res_folder + "/test_pbar_" + str(k) + ".npy", np.array(test_market.pbar_zone, dtype=np.float32))
    np.save(args.store_res_folder + "/test_rs_zone_" + str(k) + ".npy", np.array(test_market.rs_zone, dtype=np.float32))
    env.reset_competitive()