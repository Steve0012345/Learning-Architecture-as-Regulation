from collections import deque
import numpy as np

class Environment:
    def __init__(self, travel_distance, travel_time, veh_profile, num_zone, max_waiting=5, frequency=1, num_platforms=1, fleet_fractions=None,
                 competitive=False):
        assert num_zone == travel_distance.shape[0] and num_zone == travel_time.shape[0]
        self.travel_distance = travel_distance
        self.travel_time = travel_time
        self.veh_profiles = veh_profile
        self.num_zone = num_zone
        self.max_waiting = max_waiting
        self.frequency = frequency
        self.num_platforms = num_platforms

        # parameters, alpha2 and beta2 selected based on the min wage of ridehailing drivers
        # https://www.engadget.com/nyc-gig-drivers-pay-increase-012304976.html
        # https://www.ridester.com/uber-rates-cost/
        self.alpha0 = 2 # base fare for passengers
        self.alpha1 = 2 # payments $2/ miles from passengers
        self.alpha2 = 0.1 # 0.1 # $welfare cost for empty mile, set 0 when targeting only the platform profit

        self.beta1 = 0.4 # payment $0.4/ minutes from passengers
        self.beta2 = 0.1 # 0.1 # welfare cost for empty minute

        self.delta = 0.75 # 3/4 passenger payments goes to drivers

        # random pools for repositioning decisions — shared across platforms to avoid seeding issues
        self.rand_int = list(np.random.randint(0, self.num_zone, size=100000))
        self.rand_double = list(np.random.random(size=100000))

        # competitive=True with one platform gives the K = 1 point of the platform-scaling experiment:
        # the per-platform state and logit layer of the competitive loop, with a single fleet
        if num_platforms == 1 and not competitive:
            self.pricing_multipliers = np.ones((max_waiting, num_zone))
            # data for sim
            self.pass_count = np.zeros((max_waiting, num_zone, num_zone), dtype=int)
            self.veh_count = np.zeros(num_zone, dtype=int)
            self.veh_queue = [deque() for i in range(num_zone)]  # queue for waiting vehicles
            self.veh_list = []  # list of all vehicles
            self.avg_profits = [np.mean(
                veh_profile)] * 30  # active or not depends on the average profit, default_profit is average veh_profit
            self.zone_profits = np.ones((30, num_zone))
            self.avg_profit = np.mean(veh_profile)
            self.zone_profit = np.ones(num_zone)
            self.active_veh = 0
            # initialization
            self.initialize_veh()
        else:
            # split vehicle profiles across platforms; each platform owns its drivers
            n = len(veh_profile)
            if fleet_fractions is not None:
                # asymmetric split: e.g. [0.6, 0.4] gives platform 0 a larger fleet
                sizes = [int(round(f * n)) for f in fleet_fractions]
                sizes[-1] = n - sum(sizes[:-1])  # absorb rounding error so all drivers are assigned
                veh_profiles_list = [
                    veh_profile[sum(sizes[:k]):sum(sizes[:k + 1])]
                    for k in range(num_platforms)]
            else:
                fleet_size = n // num_platforms
                veh_profiles_list = [veh_profile[k * fleet_size:(k + 1) * fleet_size]
                                      for k in range(num_platforms - 1)]
                # last platform absorbs any remainder so no driver is left out
                veh_profiles_list.append(veh_profile[(num_platforms - 1) * fleet_size:])
            self._init_competitive(veh_profiles_list)

    def initialize_veh(self):
        veh_locs = np.array([i for i in range(self.num_zone)] * (len(self.veh_profiles) // self.num_zone) + [i for i in
                                                                                                             range(
                                                                                                                 len(self.veh_profiles) - len(
                                                                                                                     self.veh_profiles) // self.num_zone * self.num_zone)])  # np.random.choice(np.arange(self.num_zone), size = len(self.veh_profiles)) #initial state is constant and given (assume we know the initial distribution)
        vid = 0
        for v_profile, v_loc in zip(self.veh_profiles, veh_locs):
            v = Vehicle(vid, v_profile, v_loc)
            if v_profile <= self.avg_profit:
                self.veh_count[v_loc] += 1
                self.veh_queue[v_loc].append(v)
                self.active_veh += 1
            else:
                v.state = 2
            self.veh_list.append(v)
            vid += 1

    def get_ongoing_veh(self, max_step=6):
        ongoing_veh_count = np.zeros((max_step, self.num_zone), dtype=int)
        for v in self.veh_list:
            if v.state == 1:
                ongoing_veh_count[min(max_step - 1, int(v.remaining_time) // self.frequency), v.loc] += 1
        return ongoing_veh_count

    def step(self, new_demand, veh_schedule, price_multiplier):
        #### at time t
        # dispatch_veh
        tot_profit = 0
        tot_expense = np.zeros(self.num_zone)
        tot_count = np.zeros(self.num_zone)
        occu_mile = 0
        occu_minute = 0
        empty_mile = 0
        empty_minute = 0
        served_pass = 0

        # time step t
        for v_loc, p_loc in veh_schedule:
            p_loc2 = -1
            for j in range(self.max_waiting - 1, -1, -1):
                for i in range(self.num_zone):
                    if self.pass_count[j, p_loc, i] > 0:
                        self.pass_count[j, p_loc, i] -= 1
                        p_loc2 = i
                        served_pass += 1
                        break
                if p_loc2 != -1:
                    break
            assert self.veh_count[v_loc] > 0 and p_loc2 != -1, print(
                "Error" + "," + str(np.sum(self.pass_count, axis=(0, 2))) + "," + str(p_loc) + "," + str(v_loc))
            v = self.veh_queue[v_loc].popleft()
            v.loc = p_loc2
            v.remaining_time = self.travel_time[v_loc, p_loc] + self.travel_time[p_loc, p_loc2]
            v.state = 1
            v.waiting = 0
            self.veh_count[v_loc] -= 1
            # price calculation function is here
            tmp_payment = self.pricing_multipliers[j, p_loc] * (self.alpha0 + self.alpha1 * self.travel_distance[p_loc, p_loc2] + \
                                                     self.beta1 * self.travel_time[p_loc, p_loc2])
            tot_profit += tmp_payment
            # for the pickup trip, the platform pays drivers according to the empty trip wage
            tot_expense[v.loc] += self.alpha2 * self.travel_distance[v_loc, p_loc] + self.beta2 * self.travel_time[v_loc, p_loc] + \
                           self.delta * tmp_payment

            occu_mile += self.travel_distance[p_loc, p_loc2]
            occu_minute += self.travel_time[p_loc, p_loc2]
            empty_mile += self.travel_distance[v_loc, p_loc]
            empty_minute += self.travel_time[v_loc, p_loc]
            tot_count[v.loc] += 1
        # update total and zone profit
        self.avg_profit -= self.avg_profits[0] / 30
        self.avg_profits[:-1] = self.avg_profits[1:]
        self.avg_profits[-1] = tot_profit / (np.sum(self.veh_count) + len(veh_schedule) + 1e-4) # current available vehicles
        self.avg_profit += self.avg_profits[-1] / 30
        self.zone_profit -= self.zone_profits[0, :] / 30
        self.zone_profits[:-1, :] = self.zone_profits[1:, :]
        self.zone_profits[-1, :] = tot_expense / (self.veh_count + tot_count + 1e-4) # wage per avail vehicle in the corresponding zone
        self.zone_profit += self.zone_profits[-1, :] / 30

        # generate reposition destination, implement using a uniform double
        reposition_probabilities = [[] for i in range(self.num_zone)]
        for i in range(self.num_zone):
            if self.veh_count[i] > 0:
                reposition_probabilities[i] = (self.zone_profit + 1e-4) / (self.travel_distance[i, :] + 1e-4)
                reposition_probabilities[i][i] = 0
                reposition_probabilities[i] /= np.sum(reposition_probabilities[i])
                reposition_probabilities[i] = np.cumsum(reposition_probabilities[i])

        # vehicle cruising within its original zone
        tot_expense = np.sum(tot_expense)
        tot_expense += self.beta2 * np.sum(self.veh_count)
        empty_minute += np.sum(self.veh_count)

        # update price_multiplier
        self.pricing_multipliers[1:self.max_waiting, :] = self.pricing_multipliers[0:(self.max_waiting - 1), :]
        self.pricing_multipliers[0, :] = price_multiplier

        #### between time t and t+1
        # update_veh
        for v in self.veh_list:
            ## vehicle movement
            if v.state == 1:
                v.remaining_time -= 1
                if v.remaining_time <= 0:
                    # v arrival
                    v.state = 0
                    self.veh_count[v.loc] += 1
                    self.veh_queue[v.loc].append(v)
            ## vehicles quit/reposition/rejoin
            elif v.state == 0:
                if v.profile > self.avg_profit:
                    v.state = 2
                    self.veh_count[v.loc] -= 1
                    self.veh_queue[v.loc].remove(v)
                    self.active_veh -= 1
                else:
                    if len(self.rand_double) == 0:
                        self.rand_double = list(np.random.random(size=100000))
                    new_loc, remaining_time = v.reposition(self.travel_time, self.rand_double.pop(),\
                                                           (self.rand_double.pop() < reposition_probabilities[v.loc]).argmax())
                    if v.loc != new_loc:
                        tot_expense += self.alpha2 * self.travel_distance[v.loc, new_loc] + self.beta2 * self.travel_time[v.loc, new_loc]
                        empty_mile += self.travel_distance[v.loc, new_loc]
                        empty_minute += self.travel_time[v.loc, new_loc]
                        self.veh_count[v.loc] -= 1
                        self.veh_queue[v.loc].remove(v)
                        v.state = 1
                        v.loc = new_loc
                        v.remaining_time = remaining_time
            elif v.state == 2:
                if v.profile <= self.avg_profit:
                    v.state = 0
                    v.waiting = 0
                    v.loc = self.rand_int.pop()
                    if len(self.rand_int) == 0:
                        self.rand_int = list(np.random.randint(0, self.num_zone, size = 100000))
                    self.veh_count[v.loc] += 1
                    self.veh_queue[v.loc].append(v)
                    self.active_veh += 1

        # generate_pass
        left_pass = np.sum(self.pass_count[-1, :])
        self.pass_count[1:, :] = self.pass_count[0:-1, :]
        self.pass_count[0, :] = new_demand

        return tot_profit, tot_expense, served_pass, left_pass, occu_mile, occu_minute, empty_mile, empty_minute

    def reset(self):
        self.pass_count *= 0
        self.veh_count *= 0
        self.veh_queue = [deque() for i in range(self.num_zone)]  # queue for waiting vehicles
        self.veh_list = []  # list of all vehicles
        self.avg_profit = np.mean(self.veh_profiles)
        self.avg_profits = [np.mean(
            self.veh_profiles)] * 30  # active or not depends on the average profit, default_profit is average veh_profit
        self.zone_profits = np.ones((30, self.num_zone))
        self.zone_profit = np.ones(self.num_zone)
        self.active_veh = 0
        self.initialize_veh()


    def _init_competitive(self, veh_profiles_list):
        """Allocate separate fleet state for each platform.

        veh_profiles_list[k] contains the driver wage profiles for platform k.
        Per-platform state is stored in lists indexed by platform number so the
        same code works for 2, 4, or any other number of platforms.
        """
        self.veh_profiles_plat = veh_profiles_list

        self.pass_count_plat = [
            np.zeros((self.max_waiting, self.num_zone, self.num_zone), dtype=int)
            for _ in range(self.num_platforms)]
        self.veh_count_plat = [
            np.zeros(self.num_zone, dtype=int) for _ in range(self.num_platforms)]
        self.veh_queue_plat = [
            [deque() for _ in range(self.num_zone)] for _ in range(self.num_platforms)]
        self.veh_list_plat = [[] for _ in range(self.num_platforms)]

        self.avg_profit_plat = [float(np.mean(vp)) for vp in veh_profiles_list]
        self.avg_profits_plat = [[float(np.mean(vp))] * 30 for vp in veh_profiles_list]
        self.zone_profit_plat = [np.ones(self.num_zone) for _ in range(self.num_platforms)]
        self.zone_profits_plat = [np.ones((30, self.num_zone)) for _ in range(self.num_platforms)]
        self.active_veh_plat = [0] * self.num_platforms
        self.pricing_mults_plat = [
            np.ones((self.max_waiting, self.num_zone)) for _ in range(self.num_platforms)]

        for k in range(self.num_platforms):
            self._initialize_fleet(
                veh_profiles_list[k],
                self.veh_count_plat[k], self.veh_queue_plat[k], self.veh_list_plat[k],
                self.avg_profit_plat[k], k)

    def _initialize_fleet(self, veh_profiles, veh_count, veh_queue, veh_list, avg_profit_val, platform_idx):
        """Distribute one platform's vehicles evenly across zones, mirroring initialize_veh logic."""
        n = len(veh_profiles)
        veh_locs = np.array(
            [i for i in range(self.num_zone)] * (n // self.num_zone) +
            [i for i in range(n - n // self.num_zone * self.num_zone)]
        )
        vid = 0
        for v_profile, v_loc in zip(veh_profiles, veh_locs):
            v = Vehicle(vid, v_profile, v_loc)
            if v_profile <= avg_profit_val:
                veh_count[v_loc] += 1
                veh_queue[v_loc].append(v)
                self.active_veh_plat[platform_idx] += 1
            else:
                v.state = 2
            veh_list.append(v)
            vid += 1

    def _dispatch(self, veh_schedule, pass_count, veh_count, veh_queue, pricing_multipliers):
        """Run vehicle-to-passenger matching for one platform at one time step.
        Modifies pass_count, veh_count, and veh_queue in place.
        Returns per-step accounting needed for profit/expense tracking."""
        tot_profit = 0
        tot_expense = np.zeros(self.num_zone)
        tot_count = np.zeros(self.num_zone)
        occu_mile = occu_minute = empty_mile = empty_minute = served_pass = 0

        for v_loc, p_loc in veh_schedule:
            p_loc2 = -1
            for j in range(self.max_waiting - 1, -1, -1):
                for i in range(self.num_zone):
                    if pass_count[j, p_loc, i] > 0:
                        pass_count[j, p_loc, i] -= 1
                        p_loc2 = i
                        served_pass += 1
                        break
                if p_loc2 != -1:
                    break
            assert veh_count[v_loc] > 0 and p_loc2 != -1, \
                "Dispatch error: " + str(np.sum(pass_count, axis=(0, 2))) + "," + str(p_loc) + "," + str(v_loc)
            v = veh_queue[v_loc].popleft()
            v.loc = p_loc2
            v.remaining_time = self.travel_time[v_loc, p_loc] + self.travel_time[p_loc, p_loc2]
            v.state = 1
            v.waiting = 0
            veh_count[v_loc] -= 1
            tmp_payment = pricing_multipliers[j, p_loc] * (
                self.alpha0 + self.alpha1 * self.travel_distance[p_loc, p_loc2] +
                self.beta1 * self.travel_time[p_loc, p_loc2])
            tot_profit += tmp_payment
            # platform pays driver: empty pickup cost + delta share of fare
            tot_expense[v.loc] += (self.alpha2 * self.travel_distance[v_loc, p_loc] +
                                    self.beta2 * self.travel_time[v_loc, p_loc] +
                                    self.delta * tmp_payment)
            occu_mile += self.travel_distance[p_loc, p_loc2]
            occu_minute += self.travel_time[p_loc, p_loc2]
            empty_mile += self.travel_distance[v_loc, p_loc]
            empty_minute += self.travel_time[v_loc, p_loc]
            tot_count[v.loc] += 1

        return tot_profit, tot_expense, tot_count, served_pass, occu_mile, occu_minute, empty_mile, empty_minute

    def get_ongoing_veh_plat(self, k, max_step=6):
        """Return ongoing vehicle counts for platform k, bucketed by remaining trip time."""
        ongoing_veh_count = np.zeros((max_step, self.num_zone), dtype=int)
        for v in self.veh_list_plat[k]:
            if v.state == 1:
                ongoing_veh_count[min(max_step - 1, int(v.remaining_time) // self.frequency), v.loc] += 1
        return ongoing_veh_count

    def step_competitive(self, demands, schedules, prices):
        """Advance one time step for all platforms simultaneously.

        demands[k], schedules[k], prices[k] are the inputs for platform k.
        Returns a list of per-platform result tuples:
            (profit, expense, served, left_pass, occu_mile, occu_minute, empty_mile, empty_minute)

        All platforms compete for riders from the same demand pool (split upstream),
        each operating an exclusive fleet. Shared state is only the road network.
        """
        num_plat = self.num_platforms

        # dispatch all fleets in parallel (each operates on its own exclusive fleet state)
        dispatch_out = [
            self._dispatch(schedules[k], self.pass_count_plat[k], self.veh_count_plat[k],
                           self.veh_queue_plat[k], self.pricing_mults_plat[k])
            for k in range(num_plat)]

        # rolling profit averages — kept separate per platform because driver quit decisions
        # depend on each platform's own earnings signal, not a shared market average
        for k in range(num_plat):
            profit_k, expense_vec_k, count_k = dispatch_out[k][0], dispatch_out[k][1], dispatch_out[k][2]
            self.avg_profit_plat[k] -= self.avg_profits_plat[k][0] / 30
            self.avg_profits_plat[k][:-1] = self.avg_profits_plat[k][1:]
            self.avg_profits_plat[k][-1] = profit_k / (np.sum(self.veh_count_plat[k]) + len(schedules[k]) + 1e-4)
            self.avg_profit_plat[k] += self.avg_profits_plat[k][-1] / 30
            self.zone_profit_plat[k] -= self.zone_profits_plat[k][0, :] / 30
            self.zone_profits_plat[k][:-1, :] = self.zone_profits_plat[k][1:, :]
            self.zone_profits_plat[k][-1, :] = expense_vec_k / (self.veh_count_plat[k] + count_k + 1e-4)
            self.zone_profit_plat[k] += self.zone_profits_plat[k][-1, :] / 30

        # reposition probability per platform: weight toward high-wage, nearby zones
        repos_probs = []
        for k in range(num_plat):
            rp = [[] for _ in range(self.num_zone)]
            for i in range(self.num_zone):
                if self.veh_count_plat[k][i] > 0:
                    probs = (self.zone_profit_plat[k] + 1e-4) / (self.travel_distance[i, :] + 1e-4)
                    probs[i] = 0
                    probs /= np.sum(probs)
                    rp[i] = np.cumsum(probs)
            repos_probs.append(rp)

        # finalize expenses: add idle vehicle cruising cost while waiting in zone
        expenses = [
            np.sum(dispatch_out[k][1]) + self.beta2 * np.sum(self.veh_count_plat[k])
            for k in range(num_plat)]
        empty_mi = [dispatch_out[k][6] for k in range(num_plat)]
        empty_min = [dispatch_out[k][7] + float(np.sum(self.veh_count_plat[k])) for k in range(num_plat)]

        # advance pricing multiplier history for next state observation
        for k in range(num_plat):
            self.pricing_mults_plat[k][1:, :] = self.pricing_mults_plat[k][:-1, :]
            self.pricing_mults_plat[k][0, :] = prices[k]

        # update vehicle states for each platform
        for k in range(num_plat):
            for v in self.veh_list_plat[k]:
                if v.state == 1:
                    v.remaining_time -= 1
                    if v.remaining_time <= 0:
                        v.state = 0
                        self.veh_count_plat[k][v.loc] += 1
                        self.veh_queue_plat[k][v.loc].append(v)
                elif v.state == 0:
                    if v.profile > self.avg_profit_plat[k]:
                        # driver's reservation wage exceeds this platform's earnings, they log off
                        v.state = 2
                        self.veh_count_plat[k][v.loc] -= 1
                        self.veh_queue_plat[k][v.loc].remove(v)
                        self.active_veh_plat[k] -= 1
                    else:
                        if len(self.rand_double) == 0:
                            self.rand_double = list(np.random.random(size=100000))
                        new_loc, remaining_time = v.reposition(
                            self.travel_time, self.rand_double.pop(),
                            (self.rand_double.pop() < repos_probs[k][v.loc]).argmax())
                        if v.loc != new_loc:
                            repo_cost = (self.alpha2 * self.travel_distance[v.loc, new_loc] +
                                         self.beta2 * self.travel_time[v.loc, new_loc])
                            expenses[k] += repo_cost
                            empty_mi[k] += self.travel_distance[v.loc, new_loc]
                            empty_min[k] += self.travel_time[v.loc, new_loc]
                            self.veh_count_plat[k][v.loc] -= 1
                            self.veh_queue_plat[k][v.loc].remove(v)
                            v.state = 1
                            v.loc = new_loc
                            v.remaining_time = remaining_time
                elif v.state == 2:
                    if v.profile <= self.avg_profit_plat[k]:
                        v.state = 0
                        v.waiting = 0
                        v.loc = self.rand_int.pop()
                        if len(self.rand_int) == 0:
                            self.rand_int = list(np.random.randint(0, self.num_zone, size=100000))
                        self.veh_count_plat[k][v.loc] += 1
                        self.veh_queue_plat[k][v.loc].append(v)
                        self.active_veh_plat[k] += 1

        # expire passengers who waited too long, then admit new demand for each platform
        left_pass = []
        for k in range(num_plat):
            left_pass.append(int(np.sum(self.pass_count_plat[k][-1, :])))
            self.pass_count_plat[k][1:, :] = self.pass_count_plat[k][:-1, :]
            self.pass_count_plat[k][0, :] = demands[k]

        return [(dispatch_out[k][0], expenses[k], dispatch_out[k][3], left_pass[k],
                 dispatch_out[k][4], dispatch_out[k][5], empty_mi[k], empty_min[k])
                for k in range(num_plat)]

    def reset_competitive(self):
        """Reset all platforms' fleet state for a fresh episode."""
        for k in range(self.num_platforms):
            self.pass_count_plat[k] *= 0
            self.veh_count_plat[k] *= 0
            self.veh_queue_plat[k] = [deque() for _ in range(self.num_zone)]
            self.veh_list_plat[k] = []
            self.avg_profit_plat[k] = float(np.mean(self.veh_profiles_plat[k]))
            self.avg_profits_plat[k] = [self.avg_profit_plat[k]] * 30
            self.zone_profits_plat[k] = np.ones((30, self.num_zone))
            self.zone_profit_plat[k] = np.ones(self.num_zone)
            self.active_veh_plat[k] = 0
            self._initialize_fleet(
                self.veh_profiles_plat[k],
                self.veh_count_plat[k], self.veh_queue_plat[k], self.veh_list_plat[k],
                self.avg_profit_plat[k], k)


class Vehicle:
    def __init__(self, vid, profile, zone_id):
        self.id = vid
        self.loc = zone_id
        self.profile = profile
        self.remaining_time = 0
        self.state = 0  # 0 for waiting in a zone, 1 for ongoing, 2 for rest
        self.waiting = 0

    def reposition(self, travel_time, prob, repos_dest):
        if prob < 0.90951:  # 0.1*np.exp(-0.1), mean as 10 minutes
            self.waiting += 1
            return self.loc, 0
        else:
            self.waiting = 0
            new_loc = repos_dest
            assert self.loc != new_loc
            remaining_time = travel_time[self.loc, new_loc]
        return new_loc, remaining_time
