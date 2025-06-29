import gymnasium
from gymnasium import spaces
import torch
import numpy as np
from utils.custom_state import PortfolioState
from tabulate import tabulate

class PortfolioEnv(gymnasium.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, initial_cash=10, T=30, max_steps=30):
        super(PortfolioEnv, self).__init__()
        self.initial_cash = initial_cash
        self.T = T
        self.max_steps = max_steps
        self.current_step = 0
        #self.total_credit_drawn = 0.0
        self.last_action = None
        self.last_reward = None

        #TODO: both return rates and credit costs will be automatically generated by sampling from various distributions from another file
        self.return_rates = {
            "Bubill_0.5": torch.tensor([0.0383], dtype=torch.float32),
            "Schnatz_1": torch.tensor([0.0278], dtype=torch.float32),
            "OAT_2": torch.tensor([0.03], dtype=torch.float32),
            "OAT_3": torch.tensor([0.02], dtype=torch.float32),
            "Bonos_3": torch.tensor([0.0313], dtype=torch.float32),
            "Bonos_4": torch.tensor([0.0347], dtype=torch.float32),
            "DSL_4": torch.tensor([0.022], dtype=torch.float32),
            "BTP_2": torch.tensor([0.027], dtype=torch.float32),
            "Schatz_2": torch.tensor([0.0279], dtype=torch.float32)
        }

        self.credit_costs = {
            "credit_1": torch.tensor([4.0, 12, 0.005], dtype=torch.float32),
            "credit_2": torch.tensor([2.0, 4, 0.08], dtype=torch.float32),
            "credit_3": torch.tensor([3.0, 10, 0.05], dtype=torch.float32)
        }

        self.loan_costs = {
            "loan_1": torch.tensor([4.0, 6, 0.1], dtype=torch.float32),
            "loan_2": torch.tensor([1.0, 2, 0.08], dtype=torch.float32),
            "loan_3": torch.tensor([2.0, 5, 0.05], dtype=torch.float32)
        }
        #self.loan_costs = {
        #    "loan_1": torch.tensor([4.0, 15, 0.003], dtype=torch.float32),
        #    "loan_2": torch.tensor([1.0, 4, 0.003], dtype=torch.float32),
        #    "loan_3": torch.tensor([2.0, 10, 0.002], dtype=torch.float32)
        #}
        # Initialize the portfolio state with fixed parameters.
        self.state_obj = PortfolioState(
            total_cash=self.initial_cash,
            return_rates=self.return_rates,
            credit_costs=self.credit_costs,
            loan_costs = self.loan_costs,
            T=self.T
        )


        low_bounds = np.array([0.0] * 15, dtype=np.float32)
        high_bounds = np.concatenate([
            np.array([self.initial_cash] * 9, dtype=np.float32),
            np.array([1.0] * 3, dtype=np.float32),
            np.array([1.0]*3, dtype = np.float32)
        ])
        # self.action_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
        # #TODO: action space should be spaces.Dict as composition of box and multibinary or whatever, in this paradigm we have nondifferentiable part
        # self.action_space = spaces.Dict({
        #     "bonds":  spaces.Box(0,10,shape=(9,),dtype=float32),
        #     "credit": spaces.MultiBinary(3),
        #     "loans":  spaces.MultiBinary(3),
        #     })
        self.n_buckets = 20  # e.g. split [0,10] into 20 levels of bond‐investment
        nvec = np.concatenate([
            np.full(9, self.n_buckets, dtype=np.int32),  # each bond slot ∈ {0,…,19}
            np.ones(3, dtype=np.int32)*2,           # each credit bit ∈ {0,1}
            np.ones(3, dtype=np.int32)*2            # each loan  bit ∈ {0,1}
        ])
        self.action_space = spaces.MultiDiscrete(nvec)

        obs_dim = 1 + 9 * self.T + 3 * self.T + 3 * self.T + 9
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _get_obs(self):
        total_cash = np.array([self.state_obj.total_cash], dtype=np.float32)

        bonds_keys = ["Bubill_0.5","Schnatz_1","OAT_2",
            "OAT_3","Bonos_3","Bonos_4","DSL_4","BTP_2","Schatz_2"]

        bonds_dynamic = [self.state_obj.assets[key][1].cpu().numpy().flatten() for key in bonds_keys]
        flat_bonds = np.concatenate(bonds_dynamic)

        credit_keys = ["credit_1", "credit_2", "credit_3"]
        credits_dynamic = [self.state_obj.credits[key][1].cpu().numpy().flatten() for key in credit_keys]
        flat_credits = np.concatenate(credits_dynamic)

        loan_keys = ["loan_1", "loan_2", "loan_3"]
        loans_dynamics = [ self.state_obj.loans[key][1].cpu().numpy().flatten() for key in loan_keys]
        flat_loans = np.concatenate(loans_dynamics)
        fixed_rates = np.array([self.state_obj.assets[key][0].item() for key in bonds_keys], dtype=np.float32)
        #fixed_rates = np.array([0.0 for key in bonds_keys], dtype=np.float32)

        obs = np.concatenate((total_cash, flat_bonds, flat_credits, flat_loans, fixed_rates))
        return obs.astype(np.float32)


    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)        
        self.current_step = 0
        self.state_obj = PortfolioState(
            total_cash=self.initial_cash,
            return_rates=self.return_rates,
            credit_costs=self.credit_costs,
            loan_costs   = self.loan_costs,    
            T=self.T
        )
        return self._get_obs(), {}


    def step(self, action):
        self.state_obj.shift()

        matured_cost_credits = self.state_obj.mature_credits()  
        matured_cash_bonds   = self.state_obj.mature_bonds()    
        matured_cash_loans   = self.state_obj.mature_loans()    

        old_cash         = self.state_obj.total_cash
        old_fmv_bonds    = self.state_obj.unrealized_bonds()
        old_fmv_loans    = self.state_obj.unrealized_loans()
        old_credit_liab  = self.state_obj.unpaid_credit()

        old_NAV = (old_cash + old_fmv_bonds + 0.5 * old_fmv_loans - old_credit_liab )

        action_bonds        = action[:9]
        action_credits_raw  = action[9:12]
        action_loans_raw    = action[12:15]

        bond_buckets = action[0:9]
        frac = bond_buckets.astype(np.float32)/(self.n_buckets-1)
        reward = 0.0


        action_credits = (action_credits_raw > 0.5).astype(np.int32)
        action_loans   = (action_loans_raw   > 0.5).astype(np.int32)
        credit_keys = ["credit_1","credit_2","credit_3"]
        loan_keys   = ["loan_1",  "loan_2",  "loan_3"]

        self.state_obj.apply_action(action_bonds, action_credits, action_loans)

        new_cash        = self.state_obj.total_cash
        new_fmv_bonds   = self.state_obj.unrealized_bonds()
        new_fmv_loans   = self.state_obj.unrealized_loans()
        new_credit_liab = self.state_obj.unpaid_credit()
        new_NAV = ( new_cash + new_fmv_bonds + new_fmv_loans - new_credit_liab)

        SCALE_FACTOR = 1.0
        pct_return   = (new_NAV / (old_NAV + 1e-8)) - 1.0

        credit_exposure = sum((tensor.sum().item() * (1+ info[2].item()) )
                            for info, tensor in self.state_obj.credits.values())
        loan_exposure = sum((tensor.sum().item() * (1+ info[2].item()) )
                            for info, tensor in self.state_obj.loans.values())
        immediate_return = pct_return * SCALE_FACTOR
        reward = immediate_return 

        credit_rate = sum( tensor[:,0].item() * info[2].item() for info, tensor in self.state_obj.credits.values())
        loan_rate = sum( tensor[:,0].item() * info[2].item() for info, tensor in self.state_obj.loans.values())
        if credit_rate > 0 and loan_rate > 0:
            reward += ( loan_rate - credit_rate)
        #else:
        #    reward -= sum(tensor[:,0].item()  for info, tensor in self.state_obj.credits.values()) + sum(tensor[:,0].item() * info[2].item() for info, tensor in self.state_obj.loans.values())
        if new_cash < 2.0:
            reward -= 1.0
        if credit_exposure > 10.0:
            reward -= 1.0
        if loan_exposure > 10.0:
            reward -= 1.0

        self.current_step += 1
        done = (self.current_step >= self.max_steps)

        if done:
            final_cash = self.state_obj.total_cash
            rem_bonds  = sum(self.state_obj.assets[k][1].sum().item()
                            for k in self.state_obj.assets)
            rem_loans  = sum(self.state_obj.loans[k][1].sum().item()
                            for k in self.state_obj.loans)
            rem_credit = self.state_obj.unpaid_credit()

            reward += final_cash + 0.5 * rem_bonds - rem_loans -rem_credit
        else:
            final_cash = None
            rem_bonds  = None
            rem_loans  = None
            rem_credit = None

        self.last_action = action
        self.last_reward = reward
        obs = self._get_obs()
        info = {}

        return obs, reward, done, False, info

    def step(self, action):
        self.state_obj.shift()
        self.state_obj.mature_credits()
        self.state_obj.mature_bonds()
        self.state_obj.mature_loans()

        old_cash       = self.state_obj.total_cash
        old_bonds_pv   = self.state_obj.unrealized_bonds()
        old_loans_pv   = self.state_obj.unrealized_loans()
        old_credit_pv  = self.state_obj.unpaid_credit()
        old_NAV        = old_cash + old_bonds_pv + old_loans_pv - old_credit_pv

        action_bonds       = action[:9]
        action_credits_raw = action[9:12]
        action_loans_raw   = action[12:15]

        action_credits = (action_credits_raw > 0.5).astype(np.int32)
        action_loans   = (action_loans_raw   > 0.5).astype(np.int32)

        self.state_obj.apply_action(action_bonds, action_credits, action_loans)

        new_cash       = self.state_obj.total_cash
        new_bonds_pv   = self.state_obj.unrealized_bonds()
        new_loans_pv   = self.state_obj.unrealized_loans()
        new_credit_pv  = self.state_obj.unpaid_credit()
        new_NAV        = new_cash + new_bonds_pv + new_loans_pv - new_credit_pv

        pct_return = new_NAV / (old_NAV + 1e-8) - 1.0
        reward     = pct_return * 1.0

        drawn_this_step = sum(tensor[0,0].item()
                              for _, tensor in self.state_obj.credits.values())
        #reward -= 0.2 * drawn_this_step

        leverage = new_credit_pv / (new_cash + new_bonds_pv + new_loans_pv + 1e-8)
        if leverage > 0.6:
            reward -= 100.0 * (leverage - 0.6)**2

        total_credit = new_credit_pv
        total_loan   = new_loans_pv
        matched      = min(total_credit, total_loan)
        if matched > 0:
            r_c_avg = sum(tensor[0,0].item()*info[2].item()
                          for info, tensor in self.state_obj.credits.values()) / (total_credit + 1e-8)
            r_l_avg = sum(tensor[0,0].item()*info[2].item()
                          for info, tensor in self.state_obj.loans.values())   / (total_loan   + 1e-8)
            if r_c_avg>0 and r_l_avg>0:
                arb_bonus = matched * (r_l_avg - r_c_avg) 
                reward   += 1.0 * arb_bonus

        if new_cash < 2.0:
            reward -= 10.0

        self.current_step += 1
        done = self.current_step >= self.max_steps

        if done:
            final_cash     = self.state_obj.total_cash
            rem_bonds      = sum(tensor.sum().item() for _, tensor in self.state_obj.assets.values())
            rem_loans      = sum(tensor.sum().item() for _, tensor in self.state_obj.loans.values())
            rem_credit     = self.state_obj.unpaid_credit()
            reward        += final_cash + 0.5 * rem_bonds - rem_loans - rem_credit

        obs        = self._get_obs()
        self.last_action = action
        self.last_reward = reward
        return obs, reward, done, False, {}

    def step(self, action):

        self.state_obj.shift()
        self.state_obj.mature_credits()
        self.state_obj.mature_bonds()
        self.state_obj.mature_loans()

        old_cash      = self.state_obj.total_cash
        old_bonds_pv  = self.state_obj.unrealized_bonds()
        old_loans_pv  = self.state_obj.unrealized_loans()
        old_credit_pv = self.state_obj.unpaid_credit()
        old_NAV       = old_cash + old_bonds_pv + old_loans_pv - old_credit_pv

        action_bonds       = action[:9]
        raw_credit_bits    = action[9:12]
        raw_loan_bits      = action[12:15]
        action_credits     = (raw_credit_bits > 0.5).astype(int)
        action_loans       = (raw_loan_bits   > 0.5).astype(int)

        prev_total_credit = self.state_obj.total_credit_drawn
        prev_total_loans  = self.state_obj.total_loans_given

        self.state_obj.apply_action(action_bonds, action_credits, action_loans)

        new_cash      = self.state_obj.total_cash
        new_bonds_pv  = self.state_obj.unrealized_bonds()
        new_loans_pv  = self.state_obj.unrealized_loans()
        new_credit_pv = self.state_obj.unpaid_credit()
        new_NAV       = new_cash + new_bonds_pv + new_loans_pv - new_credit_pv

        pct_return = (new_NAV / (old_NAV + 1e-8)) - 1.0
        reward     = 100 * pct_return

        total_credit = prev_total_credit + 1e-8
        total_loan   = prev_total_loans  + 1e-8
        avg_rc = sum(tensor[0,0].item()*info[2].item()
                     for info, tensor in self.state_obj.credits.values()) 
        avg_rl = sum(tensor[0,0].item()*info[2].item()
                     for info, tensor in self.state_obj.loans.values())   
        # if avg_rl > avg_rc:
        #     reward += 5 * (avg_rl - avg_rc)

        drawn_cr   = sum(tensor[0,0].item() for _,tensor in self.state_obj.credits.values())
        given_ln   = sum(tensor[0,0].item() for _,tensor in self.state_obj.loans.values())
        matched    = min(drawn_cr, given_ln)
        spread     = (avg_rl - avg_rc)
        reward    += matched * spread * 1.0

        RESERVE = 0.2 * self.initial_cash
        if new_cash < RESERVE:
            reward -= (RESERVE - new_cash) * 10.0



        cap = 2.0 * self.initial_cash
        over_credit = max(0.0, self.state_obj.total_credit_drawn - cap)
        over_loans  = max(0.0, self.state_obj.total_loans_given - cap)
        CREDIT_BOUND_COEF = 10
        LOAN_BOUND_COEF   = 5
        over_credit = CREDIT_BOUND_COEF * (over_credit)
        over_Loans = LOAN_BOUND_COEF   * (over_loans)

        reward -= over_credit
        reward -= over_loans
        #print(f"pct return:{pct_return}  and spread {matched*spread}  over credit: {over_credit} over loans: {over_loans}")

        if new_cash < 0:
            reward -= 100.0

        self.current_step += 1
        done = self.current_step >= self.max_steps
        if done:
            final_cash = self.state_obj.total_cash
            rem_bonds  = sum(t.sum().item() for _, t in self.state_obj.assets.values())
            rem_loans  = self.state_obj.unrealized_loans()
            rem_credit = self.state_obj.unpaid_credit()
            reward   += final_cash + 0.5*rem_bonds - rem_loans - rem_credit

        obs = self._get_obs()
        info = {
            "pct_return": pct_return,
            "over_credit": over_credit,
            "over_loans": over_loans
        }
        return obs, reward, done, False, info

    def step(self, action):
        self.state_obj.shift()
        self.state_obj.mature_credits()
        self.state_obj.mature_bonds()
        self.state_obj.mature_loans()

        old_cash      = self.state_obj.total_cash
        old_bonds_pv  = self.state_obj.unrealized_bonds()
        old_loans_pv  = self.state_obj.unrealized_loans()
        old_credit_pv = self.state_obj.unpaid_credit()
        old_NAV       = old_cash + old_bonds_pv + old_loans_pv - old_credit_pv

        a_bonds   = action[:9]
        a_credits = (action[9:12] > 0.5).astype(int)
        a_loans   = (action[12:15] > 0.5).astype(int)

        prev_credit = self.state_obj.total_credit_drawn
        prev_loans  = self.state_obj.total_loans_given

        self.state_obj.apply_action(a_bonds, a_credits, a_loans)

        new_cash      = self.state_obj.total_cash
        new_bonds_pv  = self.state_obj.unrealized_bonds()
        new_loans_pv  = self.state_obj.unrealized_loans()
        new_credit_pv = self.state_obj.unpaid_credit()
        new_NAV       = new_cash + new_bonds_pv + new_loans_pv - new_credit_pv

        pct = (new_NAV / (old_NAV + 1e-8)) - 1.0
        reward = 100.0 * pct

        drawn_cr  = self.state_obj.total_credit_drawn - prev_credit
        SUBSIDY   = 0.1
        reward   += SUBSIDY * drawn_cr

        total_cr = self.state_obj.total_credit_drawn + 1e-8
        total_ln = self.state_obj.total_loans_given  + 1e-8
        rc_avg = sum(t[0,0].item()*info[2].item() 
                    for info,t in self.state_obj.credits.values()) 
        rl_avg = sum(t[0,0].item()*info[2].item() 
                    for info,t in self.state_obj.loans.values())   
        matched = min(self.state_obj.total_credit_drawn,
                    self.state_obj.total_loans_given)
        if matched>0 and rl_avg>rc_avg:
            reward += 50.0 * matched * (rl_avg-rc_avg)

        RESERVE=0.2*self.initial_cash
        if new_cash<RESERVE:
            reward -= (RESERVE-new_cash)*10.0

        CAP=2.0*self.initial_cash
        over_cr = max(0.0, self.state_obj.total_credit_drawn - CAP)
        over_ln = max(0.0, self.state_obj.total_loans_given  - CAP)
        PEN_CAP=10.0
        reward -= PEN_CAP*(over_cr + over_ln)

        unbacked = max(0.0,
        self.state_obj.total_loans_given - self.state_obj.total_credit_drawn
        )
        PEN_UNB=20.0
        reward -= PEN_UNB * (unbacked)

        if new_cash<0:
            reward -= 100.0

        self.current_step += 1
        done = (self.current_step>=self.max_steps)
        if done:
            fc = self.state_obj.total_cash
            rb = sum(t.sum().item() for _,t in self.state_obj.assets.values())
            rl = self.state_obj.unrealized_loans()
            rc = self.state_obj.unpaid_credit()
            reward += fc + 0.5*rb - 5 * rl - 5 * rc

        obs = self._get_obs()
        info = {
            "pct": pct,
            "drawn": drawn_cr,
            "spread": matched*(rl_avg-rc_avg),
            "over_cr": over_cr,
            "over_ln": over_ln,
            "unbacked": unbacked
        }
        return obs, reward, done, False, info

    def step(self, action):
        self.state_obj.shift()
        self.state_obj.mature_credits()
        self.state_obj.mature_bonds()
        self.state_obj.mature_loans()

        old_cash      = self.state_obj.total_cash
        old_bonds_pv  = self.state_obj.unrealized_bonds()
        old_loans_pv  = self.state_obj.unrealized_loans()
        old_credit_pv = self.state_obj.unpaid_credit()
        old_NAV       = old_cash + old_bonds_pv + old_loans_pv - old_credit_pv

        a_bonds   = action[:9]
        a_credits = (action[9:12] > 0.5).astype(int)
        a_loans   = (action[12:15] > 0.5).astype(int)

        prev_credit = self.state_obj.total_credit_drawn
        prev_loans  = self.state_obj.total_loans_given

        self.state_obj.apply_action(a_bonds, a_credits, a_loans)

        new_cash      = self.state_obj.total_cash
        new_bonds_pv  = self.state_obj.unrealized_bonds()
        new_loans_pv  = self.state_obj.unrealized_loans()
        new_credit_pv = self.state_obj.unpaid_credit()
        new_NAV       = new_cash + new_bonds_pv + new_loans_pv - new_credit_pv

        pct = (new_NAV / (old_NAV + 1e-8)) - 1.0
        reward = 100.0 * pct

        drawn_cr  = self.state_obj.total_credit_drawn - prev_credit
        SUBSIDY   = 0.1
        reward   += SUBSIDY * drawn_cr

        total_cr = self.state_obj.total_credit_drawn + 1e-8
        total_ln = self.state_obj.total_loans_given  + 1e-8
        rc_avg = sum(t[0,0].item()*info[2].item() 
                    for info,t in self.state_obj.credits.values()) 
        rl_avg = sum(t[0,0].item()*info[2].item() 
                    for info,t in self.state_obj.loans.values())   
        matched = min(self.state_obj.total_credit_drawn,
                    self.state_obj.total_loans_given)
        if matched>0 and rl_avg>rc_avg:
            reward += 50.0 * matched * (rl_avg-rc_avg)

        RESERVE=0.2*self.initial_cash
        if new_cash<RESERVE:
            reward -= (RESERVE-new_cash)*10.0

        CAP=2.0*self.initial_cash
        over_cr = max(0.0, self.state_obj.total_credit_drawn - CAP)
        over_ln = max(0.0, self.state_obj.total_loans_given  - CAP)
        PEN_CAP=10.0
        reward -= PEN_CAP*(over_cr + over_ln)

        unbacked = max(0.0,
        self.state_obj.total_loans_given - self.state_obj.total_credit_drawn
        )
        PEN_UNB=20.0
        reward -= PEN_UNB * (unbacked)

        if new_cash<0:
            reward -= 100.0

        self.current_step += 1
        done = (self.current_step>=self.max_steps)
        if done:
            fc = self.state_obj.total_cash
            rb = sum(t.sum().item() for _,t in self.state_obj.assets.values())
            rl = self.state_obj.unrealized_loans()
            rc = self.state_obj.unpaid_credit()
            reward += fc + 0.5*rb - 5 * rl - 5 * rc

        obs = self._get_obs()
        info = {
            "pct": pct,
            "drawn": drawn_cr,
            "spread": matched*(rl_avg-rc_avg),
            "over_cr": over_cr,
            "over_ln": over_ln,
            "unbacked": unbacked
        }
        return obs, reward, done, False, info


    def render(self, mode='human'):
        print("=" * 30)
        print(f"📆 Month: {self.current_step}")
        print(f"💰 Total Cash: ${self.state_obj.total_cash:.2f}")
        print(f"Unrealized bonds: ${self.state_obj.unrealized_bonds():.2f}")
        print(f"Unrealized loans: ${self.state_obj.unrealized_loans():.2f}")
        print(f"Unpaid credits: ${self.state_obj.unpaid_credit():.2f}")

        print("📈 Bonds Portfolio:")
        for name, (rate, tensor) in self.state_obj.assets.items():
            flat_values = tensor.cpu().numpy().flatten()
            total_amount = np.sum(flat_values)
            rate_str = f"{rate.item() * 100:.2f}%"
            if total_amount == 0:
                print(f"  - {name} (Rate: {rate_str}) — No Holdings")
                continue

            timeline = "".join(["🟢" if v > 0 else "⚪" for v in flat_values])

            print(f"  - {name} (Rate: {rate_str})")
            print(f"    Holdings Timeline: {timeline}")
            print(f"    Total Held: ${total_amount:.5f}")
            print(f"    Monthly Values:")
            for i, val in enumerate(flat_values):
                display_val = f"${val:.5f}" if val > 0 else "—"
                print(f"      • Month {i:2d}: {display_val}")
            print("")

        print("📉 Credit Obligations:")
        credit_table = []
        total_credit_balance = 0.0
        for name, (info, tensor) in self.state_obj.credits.items():
            credit_cost = info.numpy()
            flat_values = tensor.cpu().numpy().flatten()
            credit_balance = flat_values.sum()
            total_credit_balance += credit_balance
            credit_table.append([
                name,
                f"${credit_cost[0]:.2f}",
                f"{credit_cost[2] * 100:.2f}%",
                f"${credit_balance:.2f}" if credit_balance > 0 else "—"
            ])

        print(tabulate(
            credit_table,
            headers=["Credit", "Principal", "Interest", "Balance"],
            tablefmt="pretty"
        ))

        print(f"\n💳 Total Credit Balance: ${total_credit_balance:.2f}")

        print("📉 Loans:")
        loan_table = []
        total_loan_balance = 0.0
        for name, (loan_info, tensor) in self.state_obj.loans.items():
            loan_cost = loan_info.numpy()
            flat_values = tensor.cpu().numpy().flatten()
            loan_balance = flat_values.sum()
            total_loan_balance += loan_balance
            loan_table.append([
                name,
                f"${loan_cost[0]:.2f}",
                f"{loan_cost[2] * 100:.2f}%",
                f"${loan_balance:.2f}" if loan_balance > 0 else "—"
            ])

        print(tabulate(
            loan_table,
            headers=["Loan", "Principal", "Interest", "Balance"],
            tablefmt="pretty"
        ))

        print(f"\n💳 Total Loan Balance: ${total_loan_balance:.2f}")
        print("=" * 30)


    def render(self, mode='human'):
        print("=" * 30)
        print(f"📆 Month: {self.current_step}")
        print(f"💰 Total Cash: ${self.state_obj.total_cash:.2f}")
        print(f"Unrealized bonds: ${self.state_obj.unrealized_bonds():.2f}")
        print(f"Unrealized loans: ${self.state_obj.unrealized_loans():.2f}")
        print(f"Unpaid credits: ${self.state_obj.unpaid_credit():.2f}\n")

        # --- Bonds Portfolio (unchanged) ---
        print("📈 Bonds Portfolio:")
        for name, (rate, tensor) in self.state_obj.assets.items():
            flat_values = tensor.cpu().numpy().flatten()
            total_amount = np.sum(flat_values)
            rate_str = f"{rate.item() * 100:.2f}%"
            if total_amount == 0:
                print(f"  - {name} (Rate: {rate_str}) — No Holdings")
                continue

            timeline = "".join(["🟢" if v > 0 else "⚪" for v in flat_values])

            print(f"  - {name} (Rate: {rate_str})")
            print(f"    Holdings Timeline: {timeline}")
            print(f"    Total Held: ${total_amount:.5f}")
            print(f"    Monthly Values:")
            for i, val in enumerate(flat_values):
                display_val = f"${val:.5f}" if val > 0 else "—"
                print(f"      • Month {i:2d}: {display_val}")
            print("")

        # --- Credit Obligations (month‐by‐month) ---
        print("📉 Credit Obligations:")
        total_credit_balance = 0.0
        for name, (info, tensor) in self.state_obj.credits.items():
            flat_values = tensor.cpu().numpy().flatten()
            credit_balance = np.sum(flat_values)
            total_credit_balance += credit_balance

            rate_str = f"{info[2].item() * 100:.2f}%"
            principal_str = f"${info[0].item():.2f}"

            if credit_balance == 0:
                print(f"  - {name} (Principal: {principal_str}, Interest: {rate_str}) — No Outstanding Balance")
                continue

            timeline = "".join(["🟢" if v > 0 else "⚪" for v in flat_values])

            print(f"  - {name} (Principal: {principal_str}, Interest: {rate_str})")
            print(f"    Outstanding Timeline: {timeline}")
            print(f"    Total Outstanding: ${credit_balance:.5f}")
            print(f"    Monthly Balances:")
            for i, val in enumerate(flat_values):
                display_val = f"${val:.5f}" if val > 0 else "—"
                print(f"      • Month {i:2d}: {display_val}")
            print("")

        print(f"💳 Total Credit Balance: ${total_credit_balance:.2f}\n")

        # --- Loans (month‐by‐month) ---
        print("📉 Loans:")
        total_loan_balance = 0.0
        for name, (loan_info, tensor) in self.state_obj.loans.items():
            flat_values = tensor.cpu().numpy().flatten()
            loan_balance = np.sum(flat_values)
            total_loan_balance += loan_balance

            rate_str = f"{loan_info[2].item() * 100:.2f}%"
            principal_str = f"${loan_info[0].item():.2f}"

            if loan_balance == 0:
                print(f"  - {name} (Principal: {principal_str}, Interest: {rate_str}) — No Outstanding Balance")
                continue

            timeline = "".join(["🟢" if v > 0 else "⚪" for v in flat_values])

            print(f"  - {name} (Principal: {principal_str}, Interest: {rate_str})")
            print(f"    Outstanding Timeline: {timeline}")
            print(f"    Total Outstanding: ${loan_balance:.5f}")
            print(f"    Monthly Balances:")
            for i, val in enumerate(flat_values):
                display_val = f"${val:.5f}" if val > 0 else "—"
                print(f"      • Month {i:2d}: {display_val}")
            print("")

        print(f"💳 Total Loan Balance: ${total_loan_balance:.2f}")
        print("=" * 30)

    

    def get_current_metrics(self):
        metrics = {
            "total_cash": self.state_obj.total_cash,
            "unrealized_bonds": self.state_obj.unrealized_bonds(),
            "unrealized_loans": self.state_obj.unrealized_loans(),
            "unpaid_credits": self.state_obj.unpaid_credit(),
        }

        if hasattr(self, "last_reward") and self.last_reward is not None:
            metrics["reward"] = self.last_reward

        if hasattr(self, "last_action") and self.last_action is not None:
            action_bonds = self.last_action[:9]
            action_credits = self.last_action[9:12]
            action_loans = self.last_action[12:]

            bond_keys = ["Bubill_0.5","Schnatz_1","OAT_2",
                "OAT_3","Bonos_3","Bonos_4","DSL_4","BTP_2","Schatz_2"]
            bond_allocations = {
                f"alloc/{key}": float(action_bonds[i]) for i, key in enumerate(bond_keys)
            }
            metrics.update(bond_allocations)

            credit_keys = ["credit_1", "credit_2", "credit_3"]
            credit_decisions = {
                f"credit/{key}": float(action_credits[i]) for i, key in enumerate(credit_keys)
            }
            metrics.update(credit_decisions)

            loan_keys = ["loan_1", "loan_2", "loan_3"]
            loan_decisions = {
                f"loan/{key}": float(action_loans[i]) for i, key in enumerate(loan_keys)
            }
            metrics.update(loan_decisions)


        return metrics
