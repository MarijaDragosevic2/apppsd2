import torch

class PortfolioState:
    def __init__(self, total_cash, return_rates, credit_costs, loan_costs, T=12):
        self.total_cash = total_cash
        self.initial_cash = total_cash  
        self.T = T
        self.return_rates = return_rates  
        self.credit_costs = credit_costs
        self.total_credit_drawn = 0.0
        self.total_loans_given = 0.0

        self.assets = {
            "Bubill_0.5": (return_rates["Bubill_0.5"], torch.zeros((1, T), dtype=torch.float32)),
            "Schnatz_1": (return_rates["Schnatz_1"], torch.zeros((1, T), dtype=torch.float32)),
            "OAT_2": (return_rates["OAT_2"], torch.zeros((1, T), dtype=torch.float32)),
            "OAT_3":      (return_rates["OAT_3"], torch.zeros((1, T), dtype=torch.float32)),
            "Bonos_3":      (return_rates["Bonos_3"], torch.zeros((1, T), dtype=torch.float32)),
            "Bonos_4":     (return_rates["Bonos_4"], torch.zeros((1, T), dtype=torch.float32)),
            "DSL_4":  (return_rates["DSL_4"], torch.zeros((1, T), dtype=torch.float32)),
            "BTP_2":  (return_rates["BTP_2"], torch.zeros((1, T), dtype=torch.float32)),
            "Schatz_2": (return_rates["Schatz_2"], torch.zeros((1, T), dtype=torch.float32))
        }
        self.credits = {
            "credit_1": (credit_costs["credit_1"], torch.zeros((1, T), dtype=torch.float32)),
            "credit_2": (credit_costs["credit_2"], torch.zeros((1, T), dtype=torch.float32)),
            "credit_3": (credit_costs["credit_3"], torch.zeros((1, T), dtype=torch.float32))
        }

        self.loans = {
            "loan_1": (loan_costs["loan_1"], torch.zeros((1, T), dtype=torch.float32)),
            "loan_2": (loan_costs["loan_2"], torch.zeros((1, T), dtype=torch.float32)),
            "loan_3": (loan_costs["loan_3"], torch.zeros((1, T), dtype=torch.float32))
        }


    def unrealized_bonds(self):

        total_fmv = 0.0
        maturity_mapping = {
            "0.5": 3,   
            "1": 6,    
            "2": 12,   
            "3": 18,   
            "4": 24     
        }

        for key, (rate_tensor, holdings) in self.assets.items():
            r = rate_tensor.item()  
            M = self.T 
            for cat, m_months in maturity_mapping.items():
                if cat in key:
                    M = m_months
                    break

            n_cols = holdings.shape[1]
            for j in range(n_cols):
                col_units = holdings[:, j]
                principal = col_units.sum().item()
                if principal <= 0:
                    continue

                earned_fraction = (j + 1) / float(M)
                unit_value = 1.0 + r * earned_fraction
                total_fmv += principal * unit_value

        return total_fmv
    
    def unrealized_loans(self):
        total = 0.0
        for key, (rate, tensor) in self.loans.items():
            total += tensor.sum().item()
        return total

    def unpaid_credit(self):
        """Returns the sum of all credit obligations."""
        total = 0.0
        for key, (credit, tensor) in self.credits.items():
            total += tensor.sum().item()
        return total

    def shift(self):
        """
        Simulates the passage of time by shifting all asset and credit tensors
        one time-step to the right (i.e. column j becomes column j+1).
        For assets, the fixed rate remains unchanged.
        """
        for key, (rate, tensor) in self.assets.items():
            new_tensor = torch.zeros_like(tensor)
            new_tensor[:, 1:] = tensor[:, :-1]
            self.assets[key] = (rate, new_tensor)

        for key, (credit, tensor) in self.credits.items():
            new_tensor = torch.zeros_like(tensor)
            new_tensor[:, 1:] = tensor[:, :-1]
            self.credits[key] = (credit, new_tensor)

        for key, (loan, tensor) in self.loans.items():
            new_tensor = torch.zeros_like(tensor)
            new_tensor[:, 1:] = tensor[:, :-1]
            self.loans[key] = (loan, new_tensor)

    def add_cash(self, amount):
        """Adds the specified amount of cash to the portfolio."""
        self.total_cash += amount


    def apply_action(self, action_bonds, action_credits, action_loans, liquidity_reserve=0.2):
        bonds_keys = ["Bubill_0.5","Schnatz_1","OAT_2",
            "OAT_3","Bonos_3","Bonos_4","DSL_4","BTP_2","Schatz_2"]

        min_liquid = self.initial_cash * liquidity_reserve
        investable_cash = max(self.total_cash - min_liquid, 0.0)

        if not isinstance(action_bonds, torch.Tensor):
            action_bonds = torch.tensor(action_bonds, dtype=torch.float32)
        if action_bonds.shape[0] != 9:
            raise ValueError("Bond action vector must have 9 elements.")

        total_requested = torch.sum(action_bonds).item()
        if total_requested > investable_cash and total_requested > 0:
            scaling_factor = investable_cash / total_requested
            #scaling_factor = 1.0
            action_bonds = action_bonds * scaling_factor
            total_requested = torch.sum(action_bonds).item()

        for i, key in enumerate(bonds_keys):
            rate, tensor = self.assets[key]
            tensor[:, 0] = action_bonds[i]
            self.assets[key] = (rate, tensor)
        self.total_cash -= total_requested

        # Process credits.
        credit_keys = ["credit_1", "credit_2", "credit_3"]
        fixed_credit_amounts = {"credit_1": 4.0, "credit_2": 2.0, "credit_3": 3.0}
        for i, key in enumerate(credit_keys):
            if action_credits[i]:
                value = self.credits[key]
                if not isinstance(value, tuple):
                    raise ValueError(f"Expected tuple for credit {key}, got {value}")
                credit_info, tensor = value
                tensor[:, 0] = fixed_credit_amounts[key]
                self.total_credit_drawn += tensor[:,0].item()
                self.credits[key] = (credit_info, tensor)
                self.total_cash += fixed_credit_amounts[key]

        # Process loans
        loan_keys = ["loan_1", "loan_2", "loan_3"]
        fixed_loan_amounts = {"loan_1": 4.0, "loan_2": 1.0, "loan_3": 2.0}
        for i, key in enumerate(loan_keys):
            if action_loans[i]:
                value = self.loans[key]
                if not isinstance(value, tuple):
                    raise ValueError(f"Expected tuple for credit {key}, got {value}")
                loan_info, tensor = value
                tensor[:, 0] = fixed_loan_amounts[key]
                self.total_loans_given += tensor[:,0].item()
                self.loans[key] = (loan_info, tensor)
                self.total_cash -= fixed_loan_amounts[key]

   
    
    def mature_bonds(self):
        matured_cash = 0.0

        maturity_mapping = {
            "0.5": 3,   
            "1": 6,    
            "2": 12,   
            "3": 18, 
            "4": 24    
        }

        for key, (rate_tensor, holdings) in self.assets.items():
            fixed_rate = rate_tensor.item()  

            maturity_months = self.T  
            for cat, m_months in maturity_mapping.items():
                if cat in key:
                    maturity_months = m_months
                    break

            n_cols = holdings.shape[1]
            for j in range(n_cols):
                units_in_col_j = holdings[:, j]
                if torch.sum(units_in_col_j).item() == 0:
                    continue
                month_counter = j + 1
                if month_counter <= maturity_months:
                    monthly_coupon_rate = fixed_rate / 12
                    coupon_amount = torch.sum(units_in_col_j * monthly_coupon_rate).item()
                    matured_cash += coupon_amount

                if month_counter == maturity_months:
                    principal_amount = torch.sum(units_in_col_j).item()
                    matured_cash += principal_amount
                    holdings[:, j] = 0.0
                    self.assets[key] = (rate_tensor, holdings)

        self.total_cash += matured_cash
        return matured_cash


    def mature_loans(self):

        total_cost = 0.0

        for key, (loan_info, tensor) in self.loans.items():
            loan_amount = loan_info[0].item()
            duration = int(loan_info[1].item())
            interest_rate = loan_info[2].item()

            for j in range(min(self.T, duration) + 1):
                outstanding = tensor[:, j].item()
                if outstanding > 0:
                    monthly_principal = loan_amount / duration
                    interest_payment = outstanding * interest_rate/12
                    payment = monthly_principal + interest_payment
                    total_cost += payment

                    new_outstanding = max(outstanding - monthly_principal, 0.0)
                    tensor[:, j] = new_outstanding

            self.loans[key] = (loan_info, tensor)

        self.total_cash += total_cost
        return total_cost

    def mature_credits(self):
        total_cost = 0.0

        for key, (credit_info, tensor) in self.credits.items():
            loan_amount = credit_info[0].item()
            duration = int(credit_info[1].item())
            interest_rate = credit_info[2].item()

            for j in range(min(self.T, duration)+1):
                outstanding = tensor[:, j].item()
                if outstanding > 0:
                    # Monthly principal repayment
                    monthly_principal = loan_amount / duration
                    # Interest on current outstanding balance
                    interest_payment = outstanding * interest_rate/12
                    payment = monthly_principal + interest_payment
                    total_cost -= payment

                    # Update balance after repayment
                    new_outstanding = max(outstanding - monthly_principal, 0.0)
                    tensor[:, j] = new_outstanding

            self.credits[key] = (credit_info, tensor)

        self.total_cash += total_cost
        return total_cost

    def get_state(self):
        """
        Returns a dictionary representing the current state,
        cloning asset tensors (with their fixed rates) and credit tensors.
        """
        assets_state = { key: (rate, tensor.clone()) for key, (rate, tensor) in self.assets.items() }
        credits_state = { key: (credit_info, tensor.clone()) for key, (credit_info, tensor) in self.credits.items() }
        loans_state = { key: (loan_info, tensor.clone()) for key, (loan_info, tensor) in self.loans.items() }

        return {
            "total_cash": self.total_cash,
            "assets": assets_state,
            "credits": credits_state,
            "loans": loans_state
        }
