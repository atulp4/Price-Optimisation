import numpy as np
import pandas as pd
from scipy.optimize import minimize


class ConjointOptimizer:
    """Optimizer for conjoint models.

    Parameters
    ----------
    con_simulator : `con_opt_lib.ConjointSimulator`
        To simulate the opportunity for different price sets.
    cfg_algo : dict
        Optimizer parameters like rho, no.of iterations etc..
    cfg_constraints : dict
        Scenario constraint details like Max price allowed, Min price allowed,Max GC loss etc..
    ladder_rules_raw : list
        -List of price hierarchical rules(E.g Double cheeseburger should always greater than Cheeseburger)
        -This has to be created using item id as reference in the below format.
        -["14 >= 13","15 >= 4","20 >= 25","20 >= 23","29 >= 28","30 >= 29","16 >= 5 + 32 + 45","17 >= 6 + 32 + 45","19 >= 2 + 32 + 45","20 >= 1 + 32 + 45","21 >= 8 + 32 + 45","22 >= 10 + 32 + 45","23 >= 7 + 32 + 45","24 >= 9 + 32 + 45","25 >= 3 + 32 + 45","26 >= 11 + 32 + 45","27 >= 12 + 32 + 45","28 >= 13 + 32 + 45","29 >= 14 + 32 + 45","30 >= 4 + 32 + 45","16 <= 5","17 <= 6","19 <= 2","20 <= 1","21 <= 8","22 <= 10","23 <= 7","24 <= 9","25 <= 3","26 <= 11","27 <= 12","28 <= 13","29 <= 14","30 <= 4"]
    min_unit : float, default=0.05
        Minimum tolerance for inequality constraint check.
    verbose: int, default=2.
        when 0 doesn't print any intermediate results. When 2 prints all.
    """
    def __init__(
        self,
        con_simulator,
        input_opt,
        cfg_algo,
        cfg_constraints,
        vendor,
        rho,
        base,
        logger,
        ladder_rules_raw=[],
        item_hold=[],
        min_unit=0.05,
        verbose=2,
    ):
        # Inputs
        self.con_simulator = con_simulator
        self.vendor = vendor
        if self.vendor == 'SKIM':
            self.con_simulator.compute_scenario(base)
            self.base_unit = self.con_simulator.df_item.groupby(['item_name', 'Category', 'price_base'], as_index=False, sort=False).sum()['units_scenario']
            self.base_revenue = self.con_simulator.s_revenue_scenario
            self.base_gm = self.con_simulator.s_gm_scenario
        else:
            self.con_simulator.compute_scenario(base)
            self.base_unit = self.con_simulator.df_item.units_scenario
            self.base_revenue = self.con_simulator.s_revenue_scenario
            self.base_gm = self.con_simulator.s_gm_scenario
        self.logger = logger
        self.base = base
        self.min_unit = min_unit
        self.cfg_algo = cfg_algo
        self.rho = cfg_algo["RHO"][rho]
        self.maxiter = cfg_algo["MAXITER"]
        self._init_constraints(cfg_constraints)
        self.verbose = verbose
        if "LIMIT_TYPE" in self.cfg_constraints and self.cfg_constraints["LIMIT_TYPE"] == 'abs':
            self.set_bounds(
                items_to_hold=item_hold,
                limits_per=(
                    self.cfg_constraints["MIN_PRICE"],
                    self.cfg_constraints["MAX_PRICE"],
                ),
            limit_type = "abs"
            )
        else:
            self.set_bounds(
                items_to_hold=item_hold,
                limits_per=(
                    self.cfg_constraints["MIN_PRICE"],
                    self.cfg_constraints["MAX_PRICE"],
                )
            )
        self.ladder_rules_raw = ladder_rules_raw['LADDER_RULES']
        self.line_price = ladder_rules_raw['LINE_PRICE']
        self.line_price_list=[]
        #print(self.line_price)
        self._line_price()
        self._init_ladder_rules()
        self.Nfev = 1
        self.rlist = []
        self.vol_penalty = self.base_revenue * 10
        self.wap_penalty = self.base_revenue * 10
        self.gc_penalty = self.base_revenue * 10
        self.price_penalty = self.base_revenue * 10
        self.index = input_opt['index']
        self.opt = self.run_optimizer(input_opt['price_base'])
        self.scen_output = self.scenario_result(self.opt.x)

    def __str__(self):
        return self.scen_output

    def _init_constraints(self, cfg_constraints):
        self.cfg_constraints = cfg_constraints
        if self.cfg_constraints["MIN_PRICE"] is None:
            self.cfg_constraints["MIN_PRICE"] = -np.inf
        if self.cfg_constraints["MAX_PRICE"] is None:
            self.cfg_constraints["MAX_PRICE"] = np.inf
            
    def _line_price(self):
        for line in self.line_price:
            self.line_price_list.append([eval(i) for i in line.split('=')])

    def set_bounds(
        self, items_to_hold=[], limits_per=(-np.inf, np.inf), limit_type="per"
    ):
        """Set item price bounds.

        Parameters
        ----------
        items_to_hold : list, default=[]
            List of items whose prices needs to be fixed at base
        limits_per : tuple, default=(-np.inf, np.inf)
            Tuple having lower and upper price limit
        limit_type : str, default='per'
            Describes the type of limit defined. Possible inputs - ["per", "abs"]
            When per, computes a percentage limits.
            When "abs", computes a absolute limits.
        Returns
        -------
        """
        if self.vendor == 'SKIM':
            self.bounds = self.con_simulator.df_item.groupby(['item_name', 'Category'], as_index=False, sort=False).max()[
                ['item_name'] + self.con_simulator.price_cols
            ]
        else:
            self.bounds = self.con_simulator.df_item[
                ["item_id"] + self.con_simulator.price_cols
            ]

        self.bounds['price_base'] = self.base
        if self.vendor == 'SKIM':
            self.bounds["study_price_min"] = self.con_simulator.df_item.groupby(['item_name', 'Category'], as_index=False, sort=False).max()[
                self.con_simulator.price_cols
            ].min(axis=1)
            self.bounds["study_price_max"] = self.con_simulator.df_item.groupby(['item_name', 'Category'], as_index=False, sort=False).max()[
                self.con_simulator.price_cols
            ].max(axis=1)
        else:
            self.bounds["study_price_min"] = self.con_simulator.df_item[
                self.con_simulator.price_cols
            ].min(axis=1)
            self.bounds["study_price_max"] = self.con_simulator.df_item[
                self.con_simulator.price_cols
            ].max(axis=1)

        self.bounds['item_id'] = self.bounds.index.values
        limit_lower, limit_upper = limits_per
        if limit_type == "per":
            self.bounds["lb"] = self.bounds.price_base * (1 - limit_lower)
            self.bounds["ub"] = self.bounds.price_base * (1 + limit_upper)
        elif limit_type == "abs":
            self.bounds["lb"] = self.bounds.price_base - limit_lower
            self.bounds["ub"] = self.bounds.price_base + limit_upper
        # Items to hold
        if len(items_to_hold) != 0:
            fil_ = self.bounds.item_id.isin(items_to_hold)
            self.bounds.loc[fil_, "lb"] = self.bounds.loc[fil_, "price_base"]
            self.bounds.loc[fil_, "ub"] = self.bounds.loc[fil_, "price_base"]
        self._round_bounds(limit_lower)

    def _round_bounds(self, limit_lower):
        if np.isinf(limit_lower):
            self.bounds["lb"] = self.bounds[["lb", "study_price_min"]].min(axis=1)
        else:
            self.bounds["lb"] = self.bounds[["lb", "study_price_min"]].max(axis=1)
        self.bounds["ub"] = self.bounds[["ub", "study_price_max"]].min(axis=1)
        #self.bounds.to_csv('bound.csv')

    def _init_ladder_rules(self):
        df = pd.DataFrame({"con_str": self.ladder_rules_raw})
        df.con_str = df.con_str.str.replace(" ", "")
        df["ineq_type"] = df.con_str.apply(lambda s: "less" if "<" in s else "greater")
        df["strict"] = df.con_str.apply(lambda s: False if "=" in s else True)
        df["temp"] = (
            df.con_str.str.replace("=", "").str.replace(">", ",").str.replace("<", ",")
        )
        df["lhs"] = df.temp.apply(lambda x: (x.split(",")[0]).split("+") if '+' in x.split(",")[0] else (x.split(",")[0]).split(","))
        df["rhs"] = df.temp.apply(lambda x: (x.split(",")[1]).split("+") if '+' in x.split(",")[1] else (x.split(",")[1]).split(","))
        self.df_ladder_rules = df
        pri = self.con_simulator.df_item.groupby(['item_name', 'Category'], as_index=False, sort=False).max()['price_base']
        violation_at_base = self.eval_ladder_rules(
            np.array(pri)
        )
        if np.sum(np.array(violation_at_base) < 0) > 0:
            print("Ladder rules are violated at base: {(vioation_at_base < 0).sum()}")
            if self.verbose > 1:
                res = [idx for idx, val in enumerate(violation_at_base) if val < 0]
                for i in res:
                    print(self.ladder_rules_raw[i])
            self.df_ladder_rules = self.df_ladder_rules.iloc[res, :]

    def objective_function(self, prices):
        """Objective function to maximize revenue.

        Parameters
        ----------
        prices: List of input prices

        Returns
        -------
        res: Simulated revenue for given input prices
        """
        dummy_x = [x for _, x in sorted(zip(self.index, prices))]
        

        '''Upcharge handling'''

        if self.cfg_algo['UPCHARGE']['AVAIL']:
            upgrade = []
            for i in self.cfg_algo['UPCHARGE']['UPCHARGED_ITEM_ID']:
                upgrade.append(dummy_x[i])
            if (min(upgrade) - self.cfg_algo['UPCHARGE']['UPCHARGE_BASE_PRICE']) < 0:
                up = min(upgrade)
            else:
                up = max(upgrade)
            for i in self.cfg_algo['UPCHARGE']['UPCHARGED_ITEM_ID']:
                dummy_x[i] = up
        
        #print(dummy_x)
        #self.logger.info(f'dummy_x ---: {dummy_x}')
        for line in self.line_price_list:
            #print(line)
            line.sort()
            #print(dummy_x)
            dummy_x=np.array(dummy_x)
            #print(dummy_x)
            dummy_x[line[1:]]=dummy_x[line[0]]
            dummy_x=dummy_x.tolist()
        self.logger.info(f'dummy_x ---: {dummy_x}')
        self.con_simulator.compute_scenario(self.eval_g_eq(dummy_x))
        x = list(self.eval_g_eq(dummy_x)).copy()
        print(x)
        wap_flag = sum(self.eval_wap_constraints()) if self.cfg_constraints["MAX_WAP_HIKE"] is not None else 1
        gc_flag = sum(self.eval_gc_constraints()) if self.cfg_constraints["MAX_GC_DROP"] is not None else 1
        vol_flag1 = sum(self.eval_units_constraints()) if self.cfg_constraints["MAX_UNITS_DROP"] is not None else 1
        vol_flag2 = np.sum(np.array(self.eval_item_units_constraints()) < 0) if self.cfg_constraints["MAX_UNITS_DROP_ITEM"] is not None else 1
        price_flag1 = np.sum(np.array(self.eval_ladder_rules(x)) < 0)  # ladder rule
        price_flag2 = np.sum(np.array(self.price_bounds_constraints(x)) < 0)
        # compute scenario revenue
        if self.cfg_algo['METRIC'] != 'GM':
            rev = self.con_simulator.s_revenue_scenario * -1
        else:
            rev = self.con_simulator.s_gm_scenario * -1

        if ((wap_flag < 0) | (gc_flag < 0) | (vol_flag1 < 0) | (vol_flag2 > 0) | (price_flag1 > 0) | (price_flag2 > 0)):
            res = rev + self.vol_penalty
        else:
            res = rev
        print('Iteration - ', self.Nfev, res, rev, wap_flag, gc_flag, vol_flag1, vol_flag2, price_flag1, price_flag2)
        if self.Nfev == 1:
            self.logger.info(f'Rho ---: {self.rlist}')
        self.logger.info('Iteration - %s,%s,%s,%s,%s,%s,%s,%s,%s', self.Nfev, res, rev, wap_flag, gc_flag, vol_flag1, vol_flag2, price_flag1, price_flag2)
        self.Nfev += 1
        return res

    def eval_g_eq(self, input_arr):
        """Rounds price to market needs.

        Parameters
        ----------
        input_arr : input price before round off

        Returns
        -------
        rounded_prices : list of prices after rounding with defined precision
        """
        a = np.array(input_arr)
        #print(a)
        rounded_prices = np.around(np.around(a / 0.05, 2) * 0.05, 2)
        #print(rounded_prices)
        return rounded_prices

    def _optimization_function(self, prices):
        res = minimize(
            self.objective_function,
            prices,
            method="COBYLA",
            options={
                "rhobeg": self.rho,
                "disp": True,
                "maxiter": self.m
            },
        )

        return res

    def run_optimizer(self, prices):
        """Function to run optimizer.

        Parameters
        ----------
        prices : starting price for optimizer

        Returns
        -------
        df_output : final optimized price
        """
        self.m = self.cfg_algo["MAXITER"]
        rho_list = self.rho
        self.rlist = rho_list
        n = 1
        for i in rho_list:
            self.rho = i
            print("rho_list : ", rho_list)
            print("current_rho:", self.rho)
            print("iteration:", self.m)
            if n == 1:
                df_output = self._optimization_function(prices)
                n = 0
            else:
                df_output = self._optimization_function(df_output.x)
                n = 0
        return df_output

    def eval_wap_constraints(self):
        """Evaluate WAP constraint.

        Parameters
        ----------

        Returns
        -------
        evaluated wap numbers against constrained limit.
        """
        return [
            self.cfg_constraints["MAX_WAP_HIKE"]
            + 1
            - (self.con_simulator.s_wap_scenario / self.con_simulator.s_wap_base)
        ]

    def eval_gc_constraints(self):
        """Evaluate GC constraint.

        Parameters
        ----------

        Returns
        -------
        Evaluated GC numbers against constrained limit.
        """
        if self.vendor == 'SKIM':
            return [
                (self.con_simulator.s_gc_scenario - 1) + self.cfg_constraints["MAX_GC_DROP"]
            ]
        else:
            return [
                (self.con_simulator.s_gc_scenario / self.con_simulator.s_gc_base) - 1 + self.cfg_constraints["MAX_GC_DROP"]
            ]

    def eval_units_constraints(self):
        """Evaluate overall unit constraint.

        Returns
        -------
        Evaluated overall unit change against constrained limit.
        """
        return [
            (
                self.con_simulator.df_item.units_scenario.sum()
                / self.base_unit.sum()
            )
            - 1
            + self.cfg_constraints["MAX_UNITS_DROP"]
        ]

    def eval_item_units_constraints(self):
        """Evaluate individual unit constraint.

        Returns
        -------
        Evaluated item level unit change against constrained limit.
        """
        if self.vendor == 'SKIM':
            self.scen_unit = self.con_simulator.df_item.groupby(['item_name', 'Category', 'price_base'], as_index=False, sort=False).sum()['units_scenario']
        else:
            self.scen_unit = self.con_simulator.df_item.units_scenario
        return (
            (
                self.scen_unit
                / self.base_unit
            ) - 1
            + self.cfg_constraints["MAX_UNITS_DROP_ITEM"]
        ).to_list()

    def eval_ladder_rules(self, X):
        """Evaluate price hierarchy rules.

        all rules are x-y >= 0
        For rules is (lhs<=rhs) the coding is rhs - lhs >= 0
        For rules is (lhs>=rhs) the coding is lhs - rhs >= 0

        Parameters
        ----------
        X : test prices

        Returns
        -------
        list of price constrained violations
        """

        return_cons = []
        for index, row in self.df_ladder_rules.iterrows():
            val = -1 * self.min_unit if row["strict"] else 0
            if row["ineq_type"] == "less":
                lhs = row["rhs"]
                rhs = row["lhs"]
            else:
                lhs = row["lhs"]
                rhs = row["rhs"]
            for i in lhs:
                if '*' in i:
                    item_number = int(i.split('*')[0])
                    multiply_value = int(i.split('*')[1])
                    val += X[item_number] * multiply_value
                elif '/' in i:
                    item_number = int(i.split('/')[0])
                    multiply_value = int(i.split('/')[1])
                    val += X[item_number] / multiply_value
                else:
                    val += X[int(i)]
            for i in rhs:
                if '*' in i:
                    item_number = int(i.split('*')[0])
                    multiply_value = int(i.split('*')[1])
                    val -= X[item_number] * multiply_value
                elif '/' in i:
                    item_number = int(i.split('/')[0])
                    multiply_value = int(i.split('/')[1])
                    val -= X[item_number] / multiply_value
                else:
                    val -= X[int(i)]
            return_cons.append(val)
        return return_cons

    def price_bounds_constraints(self, X):
        """Evalute constraints at given price.

        Parameters
        ----------
        X : list, list of item prices

        Returns
        -------
        list of price bounds violations as inequality constraint check
        """
        lower = []
        upper = []
        #print(X)
        df_bounds = self.bounds[['lb', 'ub']]
        #df_bounds.to_csv('bounds.csv')
        df_bounds['test'] = X
        df_bounds['lb_check'] = df_bounds.test - df_bounds.lb
        df_bounds['ub_check'] = df_bounds.ub - df_bounds.test
        df_bounds['ub_check']=df_bounds['ub_check'].round(2)
        lower = df_bounds.lb_check.to_list()
        upper = df_bounds.ub_check.to_list()
        #print(lower + upper)
        return lower + upper

    def scenario_result(self, x_opt):
        """Compute scenario impact.

        Parameters
        ----------
        x_opt : Final optimized price

        """
        #GM=pd.read_excel(r"E:\SKIM\Brazil\Delivery\Input_Files\GM.xlsx")
        print(x_opt)
        p_x = [round(x,1) for _, x in sorted(zip(self.index, x_opt))]
        self.con_simulator.compute_scenario(pd.Series(p_x), bygrain=False)
        df = self.con_simulator.df_item.copy()
        df.rename(columns={'price_scenario': 'Input', 'units_scenario': 'Units', 'item_name': 'Name'}, inplace=True)
        df['Revenue'] = self.con_simulator.s_revenue_scenario_item  # (df['Input']*df['Units'])/1000
        #df['GM'] = (df['Revenue']/GM['GM'])-((df['Units']/1000)*df['fp_cost'])

        if self.vendor == 'SKIM':
            df = df.groupby(['Name', 'Category', 'Input'], as_index=False, sort=False).sum()
            df['GC'] = self.con_simulator.s_gc_scenario * 2300514 # Base GC value
        else:
            df['GC'] = self.con_simulator.s_gc_scenario
        df['GM']=self.con_simulator.s_gm_scenario_item
        df = df[['Category', 'Name', 'Input', 'Revenue', 'Units', 'GM', 'GC']]
        unit_change = (df['Units'].sum() / self.base_unit.sum()) - 1
        rev_change = (df['Revenue'].sum() / self.base_revenue) - 1
        gm_change =  (df['GM'].sum() / self.base_gm) - 1
        if self.vendor == 'SKIM':
            gc_change = self.con_simulator.s_gc_scenario - 1
        else:
            gc_change = (self.con_simulator.s_gc_scenario / self.con_simulator.s_gc_base) - 1

        print('% change in units from base: ', unit_change * 100)
        print('% change in revenue from base: ', rev_change * 100)
        print('% change in GM from base: ', gm_change * 100)
        print('% change in GC from base: ', gc_change * 100)
        return df
