import numpy as np
import pandas as pd
import datatable as dt
import re
import time


class ConjointSimulator:
    """Simlator for conjoint models.

    Parameters
    ----------
    df_item : pd.DataFrame
        should have: item_id, price_base, fp_cost, units_base, prices('p_1','p_2','p_3','p_4','p_5'..), weight_item, item_calib_factor
        may have: item_cat, item_channel, item_name, old, is_combo
        if weight_item is blank, a default of 1 is assumed
        if item_calib_factor is blank, a default of 1 is assumed
    df_resp : pd.DataFrame
        should have: resp_id, basket_size, weight_resp, gc_coeff(for KSNR)
        if weight_resp is blank, a default of 1 is assumed
    df_model : pd.DataFrame
        should have:
        resp_id, item_id, model_category, estimate_value, estimate_type (ASC, partworth-<id>,log-linear), cross_effect_item_id,wt,model_calib_factor
        may have: pw_item_id
        if model_category is blank, a items are assumed to be the same model category
        if wt is blank, a default of 1 is assumed
    df_none : pd.DataFrame
        resp_id, item_id, model_category, estimate_value, estimate_type (log-linear), cross_effect_item_id
    scale : str, default="log-log"
        cross effect component sclaing method(log-log , norm, linear)
    vendor : str, default="KSNR"
        name of vendor model(KSNR, Tiger, SKIM)
    """

    def __init__(
        self,
        df_item,
        df_resp,
        df_model,
        gc_base,
        sum_resp_wt,
        scale="log-log",
        vendor="KSNR",
        excl_cat=[],
        df_nobuy=None,
        df_avail=None,
        avail_identifier=999,
        dict_estimate_type={
            'asc': 0,
            'log-linear': 1,
            'partworth-2': 2,
            'partworth-3': 3,
            'partworth-4': 4,
            'partworth-5': 5,
            'partworth-6': 6,
            'partworth-7': 7,
            'av': 8,
            'pos':9

        },
        dict_gc_none=None,
        df_for_gc_ll=None, #flag gc related data frame assignment
        verbose=1
    ):
        # Inputs
        self.s_scale = scale
        self.gc_map = dict_gc_none
        self.df_avail = df_avail
        self.availibility_identifier = avail_identifier
        self.excluded_category = excl_cat
        self._init_item(df_item)
        self._init_resp(df_resp)
        self.s_gc_base = gc_base
        self.s_vendor = vendor
        self.dict_estimate = dict_estimate_type
        self.df_for_gc_ll=df_for_gc_ll #flag gc related data frame assignment
        self._init_model(df_model, df_nobuy)
        self.sum_resp_wt=sum_resp_wt
        
        self.verbose = verbose

        if self.s_vendor != 'SKIM':
            # Base computations
            self.s_revenue_base = self.compute_revenue("price_base", "units_base")
            self.s_wap_base = self.compute_wap("price_base", "units_base")
            self._set_scenario_prices_(self.df_item.price_base)
            self._compute_partworth_weights()
            df_prob = self.compute_prob(self.df_item.price_base.to_list())[0]
            df_prob["prob_resp_base"] = df_prob.groupby(["resp_id"])["prob"].transform(
                "sum"
            )

            df_prob = df_prob.merge(
                self.df_item[["item_id", "price_base", "weight_item"]],
                on=["item_id"],
                how="inner",
                validate="m:1",
            )
            df_prob["gc_fact"] = df_prob["prob"]
            df_prob["gc_dem"] = df_prob.price_base * df_prob["gc_fact"]

            df_prob = df_prob.merge(
                self.df_resp, on=["resp_id"], how="inner", validate="m:1"
            )
            self.df_item_resp_static = df_prob[
                [
                    "resp_id",
                    "item_id",
                    "model_category",
                    "prob_resp_base",
                    "gc_dem",
                    "weight_item",
                    "basket_size",
                    "gc_resp",
                    "weight_resp",
                    "gc_coeff",
                    "gc_fact",
                ]
            ]

    def _init_item(self, df_item):
        if "is_combo" in df_item.columns:
            fil_ = df_item.is_combo != 1
            self.df_item = df_item.loc[fil_, :]
            self.df_combo_item = df_item[['item_id', 'combo_map']]
            self.df_combo_item = self.df_combo_item.explode('combo_map')
            self.df_combo_item.combo_map = (np.where(
                self.df_combo_item.combo_map.isna(),
                self.df_combo_item.item_id,
                self.df_combo_item.combo_map)
            )
            self.df_combo_item.combo_map = self.df_combo_item.combo_map.astype(int)

        else:
            self.df_item = df_item.copy()
            self.df_combo_item = pd.DataFrame()

        self.price_cols = [i for i in self.df_item.columns if re.match("p_[0-9]+", i)]
        df_temp = self.df_item[self.price_cols]

        self.dict_item_prices_base = dict(zip(self.df_item.item_id, self.df_item.price_base))

        if self.s_scale == "log-log":
            df_temp1 = self.df_item[["item_id"] + self.price_cols]
            df_temp2 = df_temp1.melt(id_vars="item_id")
            df_temp2["value"] = np.log(df_temp2["value"])
            dict_log_linear_num_term2 = (
                df_temp2.groupby("item_id").value.mean().to_dict()
            )
            self.df_item["log_linear_dem"] = np.log(
                df_temp.max(axis=1) / df_temp.min(axis=1)
            )
            self.df_item["log_linear_num_term2"] = self.df_item.item_id.map(
                dict_log_linear_num_term2
            )
        elif self.s_scale == 'norm':
            self.df_item["log_linear_dem"] = (
                df_temp.max(axis=1) - df_temp.min(axis=1)
            ) / 2
            self.df_item["log_linear_num_term2"] = (
                df_temp.max(axis=1) + df_temp.min(axis=1)
            ) / 2
        elif self.s_scale == 'linear':
            df_temp1 = self.df_item[['item_id'] + self.price_cols]
            df_temp2 = df_temp1.melt(id_vars='item_id')
            # df_temp2.to_csv('df_temp2.csv')
            df_temp2 = df_temp2[~df_temp2.value.isna()]
            dict_log_linear_num_term2 = (
                df_temp2.groupby('item_id').value.mean().to_dict()
            )
            self.df_item["log_linear_dem"] = (
                df_temp.max(axis=1) - df_temp.min(axis=1)
            )
            
            self.df_item["log_linear_num_term2"] = (
                self.df_item.item_id.map(dict_log_linear_num_term2)
            )
            # self.df_item.to_csv('df_item.csv')

        if not ("weight_item" in self.df_item.columns):
            self.df_item["weight_item"] = 1

        if not ("item_calib_factor" in self.df_item.columns):
            self.df_item["item_calib_factor"] = 1
        else:
            self.df_item["item_calib_factor"] = self.df_item[
                "item_calib_factor"
            ].fillna(1)

        self.dict_old_items = None
        if "old" in self.df_item.columns:
            self.dict_old_items = dict(zip(self.df_item.item_id, self.df_item.old))

        df_temp = self.df_item[self.price_cols + ["item_id"]]
        
        df_temp = df_temp.melt(
            id_vars="item_id", var_name="price_level", value_name="price"
        )
        
        df_temp["estimate_type"] = df_temp.price_level.str.replace("p_", "").astype(int)
        df_temp.sort_values(["item_id", "price_level"], inplace=True)
        
        if self.df_avail is not None:
            for key in self.df_avail['availability'][self.df_avail['availability'] == False].index.values + 1:
                df_temp.loc[df_temp['item_id'] == key, 'price'] = df_temp.loc[df_temp['item_id'] == key, 'price'].fillna(self.availibility_identifier)

        df_temp["price_next"] = df_temp.groupby(["item_id"]).price.shift(-1)
        df_temp["price_previous"] = df_temp.groupby(["item_id"]).price.shift(1)
        df_temp["pricediff_previous"] = df_temp["price"] - df_temp["price_previous"]
        df_temp["pricediff_next"] = df_temp["price_next"] - df_temp["price"]
        df_temp = df_temp[df_temp.price_level != "p_1"]
        self.df_price_ranges = df_temp.drop(["price_level"], axis=1)

    def _init_resp(self, df_resp):
        """Initilize Respondent level data.

        Returns
        -------
            Respondent level data with missing weights(if any)
        """
        self.df_resp = df_resp.copy()
        if "weight_resp" not in self.df_resp.columns:
            self.df_resp["weight_resp"] = 1

    def _init_model(self, df_model, df_nobuy):
        """Initilize Model data.

        Returns
        -------
            Model data with all one time computations pre computed
        """
        self.df_item_resp = df_model.copy()
        self.nobuy = False

        # Append nobuy dataframe
        if df_nobuy is not None:
            self.df_item_resp = self.df_item_resp.append(df_nobuy)
        
            self.nobuy = True

        # Check if estimate type not int
        if self.df_item_resp.estimate_type.dtype == object:
            self.df_item_resp.estimate_type = (
                self.df_item_resp.estimate_type.map(self.dict_estimate)
            )

        # partworth cols
        self.pw_list = list(np.arange(2, len(self.price_cols) + 1))

        # Set category
        if not ("model_category" in self.df_item_resp.columns):
            self.df_item_resp["model_category"] = 1
        if not ("wt" in self.df_item_resp.columns):
            self.df_item_resp["wt"] = 1

        # Initiate a dummy column which will have the price of respective item for dot product
        self.df_item_resp["estimate_mul"] = 1
        

        # Divide log-linear coeff with log-linear denominator constant
        fil_ = self.df_item_resp.estimate_type == self.dict_estimate["log-linear"]
        dict_log_linear_dem = dict(
            zip(self.df_item.item_id, self.df_item.log_linear_dem)
        )
        # print('dict_log_linear_dem',dict_log_linear_dem)
        
        dict_log_linear_num_term2 = dict(
            zip(self.df_item.item_id, self.df_item.log_linear_num_term2)
        )
        # print('dict_log_linear_num_term2',dict_log_linear_num_term2)
        # self.df_item_resp.to_csv('df_item_resp_.csv')
        self.df_item_resp.loc[fil_, "estimate_value"] = self.df_item_resp.loc[
            fil_, "estimate_value"
        ] / self.df_item_resp.loc[fil_, "cross_effect_item_id"].map(dict_log_linear_dem)
        self.df_item_resp.estimate_value = self.df_item_resp.estimate_value.replace(
            [-np.inf, np.inf], 0
        )
        df_temp = self.df_item_resp[fil_]
        # print(df_temp['estimate_type'].unique(),self.dict_estimate["asc"])
        df_temp["estimate_type"] = self.dict_estimate["asc"]
        df_temp.estimate_value = (
            df_temp.estimate_value
            * df_temp.cross_effect_item_id.map(dict_log_linear_num_term2)
        )
        df_temp = (
            df_temp.groupby(["item_id", "resp_id", "estimate_type", "model_category"])
            .estimate_value.sum()
            .reset_index()
            .rename(columns={"estimate_value": "estimate_value_temp"})
        )
        # df_temp.to_csv('df_temp.csv')
        self.df_item_resp = self.df_item_resp.merge(
            df_temp, on=["item_id", "resp_id", "estimate_type", "model_category"], how="left"
        )
        self.df_item_resp.estimate_value = (
            self.df_item_resp.estimate_value
            - self.df_item_resp.estimate_value_temp.fillna(0)
        )
        self.df_item_resp.drop("estimate_value_temp", axis=1, inplace=True)
        # print('self.df_item_resp.columns',self.df_item_resp.columns)
        if "pw_item_id" in self.df_item_resp.columns:
            self.df_item_resp.pw_item_id = np.where(
                self.df_item_resp.pw_item_id.isna(), self.df_item_resp.item_id, self.df_item_resp.pw_item_id
            )
            fil_ = self.df_item_resp.estimate_type == self.dict_estimate["log-linear"]
            self.df_item_resp["estimate_mul_gc"] = 1
            # print('self.gc_map',self.gc_map)
            if self.gc_map == None:
                temp_prices = self.dict_item_prices_base
                # print('self.df_avail',self.df_avail)
                if self.df_avail is not None:
                    temp2 = self.df_item[self.price_cols]
                    for key in self.df_avail['availability'][self.df_avail['availability'] == False].index.values:
                        temp_prices[key + 1] = temp2.iloc[key].max()
                self.df_item_resp.loc[fil_, "estimate_mul_gc"] = self.df_item_resp[
                    fil_].cross_effect_item_id.map(temp_prices)
            # Handling exception in case of none share calculation
            else:
                self.df_item_resp.loc[fil_, "estimate_mul_gc"] = self.df_item_resp[
                    fil_].cross_effect_item_id.map(self.gc_map)
        else:
            self.df_item_resp['pw_item_id'] = self.df_item_resp['item_id']
            
        

        
        self.df_item_resp_pw  = dt.Frame(self.df_item_resp[self.df_item_resp.estimate_type.isin(self.pw_list)].reset_index(drop=True))
        self.df_item_resp_asc = dt.Frame(self.df_item_resp[self.df_item_resp.estimate_type == self.dict_estimate["asc"]].reset_index(drop=True))
        self.df_item_resp_ll  = dt.Frame(self.df_item_resp[self.df_item_resp.estimate_type == self.dict_estimate["log-linear"]].reset_index(drop=True))
        self.df_item_resp_av  = dt.Frame(self.df_item_resp[self.df_item_resp.estimate_type == self.dict_estimate["av"]].reset_index(drop=True))
        self.df_item_resp_pos  = dt.Frame(self.df_item_resp[self.df_item_resp.estimate_type == self.dict_estimate["pos"]].reset_index(drop=True))
        
        self.df_ll=self.df_item_resp_ll.copy()
        #self.df_ll=dt.Frame(self.df_ll)
        for model_category in self.df_for_gc_ll.Model_category.unique():
            fil_1=self.df_for_gc_ll['Model_category']==model_category
            for item_id in self.df_for_gc_ll.loc[fil_1,'Item_id'].unique():
                fil_2=(self.df_for_gc_ll['Model_category']==model_category) & (self.df_for_gc_ll['Item_id']==item_id)
                for cross_item in self.df_for_gc_ll.loc[fil_2,'Cross_Effect_Item_id']:
                    fil_3=(self.df_for_gc_ll['Model_category']==model_category) & (self.df_for_gc_ll['Item_id']==item_id) &(self.df_for_gc_ll['Cross_Effect_Item_id']==cross_item)
                    val=self.df_for_gc_ll.loc[fil_3,'Estimated_mul']
                    self.df_ll[ (dt.f.model_category ==model_category)  & (dt.f.item_id == item_id) & (dt.f.cross_effect_item_id==cross_item), dt.update(estimate_mul = val)]
                    self.df_ll[ (dt.f.model_category ==model_category)  & (dt.f.item_id == item_id) & (dt.f.cross_effect_item_id==cross_item), dt.update(estimate_mul_gc = val)]
                    
        self.df_ll= dt.Frame(self.df_ll)    

    def compute_revenue(self, price_col, units_col, bygrain=False):
        """Compute revenue as desired granularity.

        Parameters
        ----------
        price_col : str
            column name of price. Should be 'price_base' or 'price_scenario'
        units_col : str
            column name of units. Should be 'units_base' or 'units_scenario'
        bygrain : bool, default=False
            When True, price is computed at granualar level like cluster/item otherwise computes at overall level.

        Returns
        -------
            Scenario/Base revenue
        """
        if bygrain:
            return ((self.df_item[units_col] * self.df_item[price_col]))
        else:
            return (((self.df_item[units_col] * self.df_item[price_col]).sum()))
    def compute_gm(self, bygrain=False):
        """Compute GM as desired granularity.
    
        Parameters
        ----------
        bygrain : bool, default=False
            When True, price is computed at granualar level like cluster/item otherwise computes at overall level.
    
        Returns
        -------
            Scenario/Base GM
        """
        'GM variable is market specific--------'
        #GM=pd.read_excel(r"C:\Users\ashutosh.anand\Desktop\Markets\Delivery\Input_Files\GM.xlsx")
        df_gm=pd.DataFrame()
        df_gm['item_name'] = self.df_item['item_name']
        df_gm['revenue_scenario'] = self.s_revenue_scenario_item
        df_gm['units_scenario'] = self.df_item['units_scenario']
        #df_gm.to_csv('df_gm.csv')
        #fc=pd.read_excel(r'C:\Users\ashutosh.anand\Desktop\Markets\Philippines\Instore\Code\Input_Files\F&Pcost.xlsx')
        df_gm['fp'] = self.df_item['fp_cost']
        #df_gm=df_gm.iloc[1:,:]
        #print(fc['FP Cost'])
        if self.s_vendor == 'SKIM':
            df_gm=df_gm.groupby(['item_name'], as_index=False, sort=False).sum()
            fp=self.df_item.groupby(['item_name', 'Category'], as_index=False, sort=False).max()['fp_cost']
        
        if bygrain:
            if self.s_vendor == 'KSNR':
                return (df_gm['revenue_scenario']) - (df_gm['units_scenario']*df_gm['fp'])
            else:
                return (df_gm['revenue_scenario']) - ((df_gm['units_scenario'])*fp)#return self.df_by_grain[price_col] * self.df_by_grain[units_col]
        else:
            if self.s_vendor == 'KSNR':
                return ((df_gm['revenue_scenario']) - (df_gm['units_scenario']*df_gm['fp'])).sum()
            else:
                return ((df_gm['revenue_scenario']) - ((df_gm['units_scenario'])*fp)).sum()

    def compute_wap(self, price_col, units_col):
        """Compute WAP.

        Parameters
        ----------
        price_col : str
            column name of price. Should be 'price_base' or 'price_scenario'
        units_col : str
            column name of units. Should be 'units_base' or 'units_scenario'

        Returns
        -------
            Scenario/Base WAP
        """
        return (self.df_item[units_col] * self.df_item[price_col]).sum() / self.df_item[
            units_col
        ].sum()

    def _compute_partworth_weights(self):
        """Compute partworth weights.

        Returns
        -------
            Partworth weights by item & test price
        """
        temp_prices = self.dict_item_prices
        if self.df_avail is not None:
            for key in self.df_avail['availability'][self.df_avail['availability'] == False].index.values + 1:
                temp_prices[key] = 999.0

        self.df_price_ranges["current_price"] = self.df_price_ranges.item_id.map(
            temp_prices
        )
        fil_1 = (
            self.df_price_ranges.current_price >= self.df_price_ranges.price_previous
        ) & (self.df_price_ranges.current_price <= self.df_price_ranges.price)
        fil_2 = (self.df_price_ranges.current_price > self.df_price_ranges.price) & (
            self.df_price_ranges.current_price <= self.df_price_ranges.price_next
        )
        self.df_price_ranges.loc[fil_1, "pw_weight"] = (
            self.df_price_ranges[fil_1].current_price
            - self.df_price_ranges[fil_1].price_previous
        ) / self.df_price_ranges[fil_1].pricediff_previous
        self.df_price_ranges.loc[fil_2, "pw_weight"] = (
            self.df_price_ranges[fil_2].price_next
            - self.df_price_ranges[fil_2].current_price
        ) / self.df_price_ranges[fil_2].pricediff_next
        self.df_price_ranges.loc[~(fil_1 | fil_2), "pw_weight"] = 0
        return self.df_price_ranges[["item_id", "estimate_type", "pw_weight"]]

    def compute_prob(self, item_prices, ignore_crosseff=False):
        """Compute Prob.

        Parameters
        ----------
        item_prices : list
            list of scenario prices of all items
        ignore_crosseff : bool
            set as True if cross effects needs to be ignored defaults to False

        Returns
        -------
        df_prob : DataFrame
            Respondent * Item level probabilities along with model category information
        df_prob_nobuy : DataFrame
            None if ignore_crosseff is False
            else returns Respondent level nobuy probabilities of all categories
        """
        st = time.time()
        self._set_scenario_prices_(item_prices)
        if self.verbose >= 1:
            print("set_price : ", time.time() - st)
        st = time.time()

        # Log Linears
        if self.s_scale == "log-log":
            self._set_log_transform_num()
            ll = dt.Frame(pd.DataFrame.from_dict(
                self.dict_log_transform, orient='index').reset_index())
        else:
            temp_prices = self.dict_item_prices
            # print('df_avail',self.df_avail)
            if self.df_avail is not None:
                temp2 = self.df_item[self.price_cols]
                for key in self.df_avail['availability'][self.df_avail['availability'] == False].index.values:
                    temp_prices[key + 1] = temp2.iloc[key].max()
            # print(temp_prices)
            ll = dt.Frame(pd.DataFrame.from_dict(
                temp_prices, orient='index').reset_index())
        ll.names = ('cross_effect_item_id', 'temp')
        ll.key = 'cross_effect_item_id'
        self.df_item_resp_ll['estimate_mul'] = (
            (self.df_item_resp_ll)[:, :, dt.join(dt.Frame(ll))]['temp'])
        # df_item_resp_ll[ (dt.f.model_category ==1)  & (dt.f.item_id == -99) , dt.update(estimate_mul = dt.f.utility * 0.3)]
        if self.verbose >= 1 :
            print("log linear assign : ", time.time() - st)
        st = time.time()

        # Availability AV
        self.df_item_resp_av['estimate_mul'] = 0
        self.df_item_resp_av['estimate_mul_gc'] = 0
        
        #Pos
        self.df_item_resp_pos['estimate_mul'] = 0
        self.df_item_resp_pos['estimate_mul_gc'] = 0


        # Parthworths
        self.item_price_levels = self._compute_partworth_weights()
        self.item_price_levels.rename(columns={'item_id': 'pw_item_id'}, inplace=True)
        dt_pw = dt.Frame(self.item_price_levels)
        #print(dt_pw)
        dt_pw.key = ['pw_item_id', 'estimate_type']
        self.df_item_resp_pw['estimate_mul'] = (
            self.df_item_resp_pw[:, :, dt.join(dt_pw)]['pw_weight'].to_pandas()['pw_weight']
        )
        if self.verbose >= 1:
            print("PW assign : ", time.time() - st)
        st = time.time()

        # Ignore crosseffects for GC model
        print('ignore_crosseff:',ignore_crosseff)
        if ignore_crosseff == True:
            self.df_item_resp_pw['estimate_mul_gc'] = self.df_item_resp_pw['estimate_mul']
            if self.verbose >= 1 :
                print("log linear gc assign : ", time.time() - st)
            df_prob, df_prob_nobuy = self._compute_utility(ignore_crosseff=True)
            df_prob_nobuy_gc = self._compute_utility_gc(ignore_crosseff=True) # Flag calling utility function which returns share required for gc calculation 

           
        # Only Share model
        else:
            df_prob, df_prob_nobuy = self._compute_utility()
        return (df_prob, df_prob_nobuy,df_prob_nobuy_gc)
    
    '''
    Flag: New Function is added for calculation of shares and None Shares for gc.
    '''
    
    def _compute_utility_gc(self, ignore_crosseff=False):
        """Compute utilities and return probabilities.

        Returns
        -------
        df_prob_gc : DataFrame
            Respondent * Item level probabilities along with model category information
        df_prob_nobuy : DataFrame
            None if ignore_crosseff is False
            else returns Respondent level nobuy probabilities of all categorie
        """
        st = time.time()

       
        
        #self.df_item_resp_gc['estimate_mul']=self.df_item_resp_gc['estimate_mul'].astype('int64')
        

        if self.verbose >= 1 :
            print("concat data : ", time.time() - st)
        st = time.time()
        # flag = 1
        # try:
        #     self.df_item_resp_gc=pd.read_csv('df_item_resp_gc.csv')
        #     self.df_item_resp_gc=dt.Frame(self.df_item_resp_gc)
        #     print('Respondent GC file present')
        #     flag = 0
        # except:  
        # self.df_ll=self.df_item_resp_ll.copy()
        # #self.df_ll=dt.Frame(self.df_ll)
        # for model_category in self.df_for_gc_ll.Model_category.unique():
        #     fil_1=self.df_for_gc_ll['Model_category']==model_category
        #     for item_id in self.df_for_gc_ll.loc[fil_1,'Item_id'].unique():
        #         fil_2=(self.df_for_gc_ll['Model_category']==model_category) & (self.df_for_gc_ll['Item_id']==item_id)
        #         for cross_item in self.df_for_gc_ll.loc[fil_2,'Cross_Effect_Item_id']:
        #             fil_3=(self.df_for_gc_ll['Model_category']==model_category) & (self.df_for_gc_ll['Item_id']==item_id) &(self.df_for_gc_ll['Cross_Effect_Item_id']==cross_item)
        #             val=self.df_for_gc_ll.loc[fil_3,'Estimated_mul']
        #             self.df_ll[ (dt.f.model_category ==model_category)  & (dt.f.item_id == item_id) & (dt.f.cross_effect_item_id==cross_item), dt.update(estimate_mul = val)]
        #             self.df_ll[ (dt.f.model_category ==model_category)  & (dt.f.item_id == item_id) & (dt.f.cross_effect_item_id==cross_item), dt.update(estimate_mul_gc = val)]
         
        #self.df_ll= dt.Frame(self.df_ll)          
        self.df_item_resp_gc = dt.rbind([
            self.df_item_resp_pw,
            self.df_item_resp_asc,
            self.df_ll,
            self.df_item_resp_av,
            self.df_item_resp_pos])
        # if flag:
        #     self.df_item_resp_gc.to_csv('df_item_resp_gc.csv')
        # Compute Utility for GC and share models
        if ignore_crosseff:

            df_prob_gc = self.df_item_resp_gc[:, {
                "utility"   : dt.math.exp(dt.sum(dt.f.estimate_value * dt.f.estimate_mul)),
                "utility_gc": dt.math.exp(dt.sum(dt.f.estimate_value * dt.f.estimate_mul_gc))},
                dt.by("item_id", "resp_id", "model_category")]
            #df_prob_gc.to_csv('df_prob_gc_new.csv')
            if self.verbose >= 1 :
                print("utilities compute : ", time.time() - st)
            st = time.time()
            # print('nobuy',self.nobuy)
            if self.nobuy == True:
                df_prob_gc = df_prob_gc[:, dt.f[:].extend({
                    "utility_resp"   : dt.sum(dt.f.utility),
                    "utility_resp_gc": dt.sum(dt.f.utility_gc)}),
                    dt.by("resp_id", "model_category")]
            else:
                df_prob_gc = df_prob_gc[:, dt.f[:].extend({
                    "utility_resp"   : 1 + dt.sum(dt.f.utility),
                    "utility_resp_gc": 1 + dt.sum(dt.f.utility_gc)}),
                    dt.by("resp_id", "model_category")]
            df_prob_gc = df_prob_gc[:, dt.f[:].extend({
                "prob"       : dt.f.utility / dt.f.utility_resp,
                "nobuy_prob" : dt.f.utility_gc / dt.f.utility_resp_gc})
            ]
            
            df_nobuy_prob_gc = (
                df_prob_gc[dt.f.item_id == -99, :][:,
            dt.mean(dt.f.nobuy_prob), dt.by('resp_id', 'model_category')]).to_pandas()
            
            
            df_prob_gc = df_prob_gc[dt.f.item_id != -99, :]

            if self.verbose >= 1 :
                print("probablities compute : ", time.time() - st)

        # Non GC models like KSNR and Tiger
        else:
            df_prob_gc = (dt.Frame(self.df_item_resp_gc)[:, {
                "utility": dt.math.exp(
                    dt.sum(dt.f.estimate_value * dt.f.estimate_mul) *
                    dt.mean(dt.f.model_calib_factor))},
                dt.by("item_id", "resp_id", "model_category")]
            )
            if self.nobuy == True:
                df_prob_gc = dt.Frame(df_prob_gc)[:, dt.f[:].extend({
                    "utility_resp"   : dt.sum(dt.f.utility)}),
                    dt.by("resp_id", "model_category")]
                df_prob_gc = df_prob_gc[dt.f.item_id != -99, :]
            else:
                df_prob_gc = dt.Frame(df_prob_gc)[:, dt.f[:].extend({
                    "utility_resp"   : 1 + dt.sum(dt.f.utility)}),
                    dt.by("resp_id", "model_category")]
            df_prob_gc = df_prob_gc[:, dt.f[:].extend({
                "prob"       : dt.f.utility / dt.f.utility_resp})]
            df_nobuy_prob_gc = None

        df_prob_gc = df_prob_gc[:, ["item_id", "resp_id", "model_category", "prob"]].to_pandas()
        self.df_item_resp_gc = self.df_item_resp_gc.to_pandas()


        # redistributing combo share to parent item
        df_prob_gc = self._distribute_combo_item(df_prob_gc)

        
        # excuding new item prob
        # print('self.dict_old_items',self.dict_old_items)
        if self.dict_old_items is not None:
            df_prob_gc = df_prob_gc.merge(
                self.df_item_resp_gc.groupby(["item_id", "resp_id"], as_index=False)[
                    "wt"
                ].mean(),
                on=["item_id", "resp_id"],
                how="inner",
            )
            df_prob_gc["old"] = df_prob_gc.item_id.map(self.dict_old_items)
            df_prob_gc["prob"] = df_prob_gc["prob"] * df_prob_gc["wt"] * df_prob_gc["old"]
        

        return df_nobuy_prob_gc
    def _compute_utility(self, ignore_crosseff=False):
        """Compute utilities and return probabilities.

        Returns
        -------
        df_prob : DataFrame
            Respondent * Item level probabilities along with model category information
        df_prob_nobuy : DataFrame
            None if ignore_crosseff is False
            else returns Respondent level nobuy probabilities of all categorie
        """
        st = time.time()

        # self.df_item_resp.to_csv('df_item_resp.csv')
        self.df_item_resp = dt.rbind([
            self.df_item_resp_pw,
            self.df_item_resp_ll,
            self.df_item_resp_asc,
            self.df_item_resp_av,
            self.df_item_resp_pos])
        if self.verbose >= 1 :
            print("concat data : ", time.time() - st)
        st = time.time()
        # Compute Utility for GC and share models
        if ignore_crosseff:

            df_prob = self.df_item_resp[:, {
                "utility"   : dt.math.exp(dt.sum(dt.f.estimate_value * dt.f.estimate_mul)),
                "utility_gc": dt.math.exp(dt.sum(dt.f.estimate_value * dt.f.estimate_mul_gc))},
                dt.by("item_id", "resp_id", "model_category")]
            #df_prob.to_csv('df_prob_cu_gc.csv')
            #df_prob[ (dt.f.model_category ==8)  & (dt.f.item_id == -99) , dt.update(utility = dt.f.utility )] #Market Specific
            #df_prob[ (dt.f.model_category ==8)  & (dt.f.item_id == -99) , dt.update(utility_gc = dt.f.utility_gc )] #Market Specific
            # df_prob.to_csv('df_prob.csv') #change

            if self.verbose >= 1 :
                print("utilities compute : ", time.time() - st)
            st = time.time()
            # print('nobuy',self.nobuy)
            if self.nobuy == True:
                df_prob = df_prob[:, dt.f[:].extend({
                    "utility_resp"   : dt.sum(dt.f.utility),
                    "utility_resp_gc": dt.sum(dt.f.utility_gc)}),
                    dt.by("resp_id", "model_category")]
            else:
                df_prob = df_prob[:, dt.f[:].extend({
                    "utility_resp"   : 1 + dt.sum(dt.f.utility),
                    "utility_resp_gc": 1 + dt.sum(dt.f.utility_gc)}),
                    dt.by("resp_id", "model_category")]
            df_prob = df_prob[:, dt.f[:].extend({
                "prob"       : dt.f.utility / dt.f.utility_resp,
                "nobuy_prob" : dt.f.utility_gc / dt.f.utility_resp_gc})
            ]
            #df_prob.to_csv('df_prob2.csv')
            #df_prob[df_prob['resp_id'] in (self.resp_id_list),'prob']=0
            #df_prob.to_csv('df_prob2.csv')
            df_nobuy_prob = (
                df_prob[dt.f.item_id == -99, :][:,
            dt.mean(dt.f.nobuy_prob), dt.by('resp_id', 'model_category')]).to_pandas()
            
            
            df_prob = df_prob[dt.f.item_id != -99, :]
            # df_prob.to_csv('df_prob.csv')
            # df_nobuy_prob.to_csv('df_nobuy_prob.csv')
            # df_prob.to_csv("df_prob_com_prob_1.csv")
            if self.verbose >= 1 :
                print("probablities compute : ", time.time() - st)

        # Non GC models like KSNR and Tiger
        else:
            df_prob = (dt.Frame(self.df_item_resp)[:, {
                "utility": dt.math.exp(
                    dt.sum(dt.f.estimate_value * dt.f.estimate_mul) *
                    dt.mean(dt.f.model_calib_factor))},
                dt.by("item_id", "resp_id", "model_category")]
            )
            if self.nobuy == True:
                df_prob = dt.Frame(df_prob)[:, dt.f[:].extend({
                    "utility_resp"   : dt.sum(dt.f.utility)}),
                    dt.by("resp_id", "model_category")]
                df_prob = df_prob[dt.f.item_id != -99, :]
            else:
                df_prob = dt.Frame(df_prob)[:, dt.f[:].extend({
                    "utility_resp"   : 1 + dt.sum(dt.f.utility)}),
                    dt.by("resp_id", "model_category")]
            df_prob = df_prob[:, dt.f[:].extend({
                "prob"       : dt.f.utility / dt.f.utility_resp})]
            df_nobuy_prob = None

        df_prob = df_prob[:, ["item_id", "resp_id", "model_category", "prob"]].to_pandas()
        self.df_item_resp = self.df_item_resp.to_pandas()
        # df_prob.to_csv('df_prob.csv')

        # redistributing combo share to parent item
        df_prob = self._distribute_combo_item(df_prob)
        # df_prob.to_csv('df_prob_com_prob_2.csv')
        
        # excuding new item prob
        # print('self.dict_old_items',self.dict_old_items)
        if self.dict_old_items is not None:
            df_prob = df_prob.merge(
                self.df_item_resp.groupby(["item_id", "resp_id"], as_index=False)[
                    "wt"
                ].mean(),
                on=["item_id", "resp_id"],
                how="inner",
            )
            df_prob["old"] = df_prob.item_id.map(self.dict_old_items)
            df_prob["prob"] = df_prob["prob"] * df_prob["wt"] * df_prob["old"]
        
        # df_prob.to_csv('df_prob_com_prob_3.csv') #change

        return df_prob[["item_id", "resp_id", "model_category", "prob"]], df_nobuy_prob

    def _distribute_combo_item(self, df_prob):
        """Add combo probabilities to individual items.

        Returns
        -------
        df_prob : DataFrame
            prob dataframe after adding combo probabilities if any
        """
        # print('self.df_combo_item.empty',self.df_combo_item.empty)
        # df_prob.to_csv("df_prob_before.csv")
        # self.df_combo_item.to_csv('df_combo_item.csv')
        if not (self.df_combo_item.empty):
            st = time.time()
            df_prob = (
                self.df_combo_item.merge(df_prob,
                    left_on='item_id', right_on='item_id', how='inner').drop(
                    columns='item_id').rename(columns={'combo_map': 'item_id'})
            )
            # df_prob.to_csv("df_prob_after.csv")
            if self.verbose >= 1 :
                print("merge combos : ", time.time() - st)
            st = time.time()
            df_prob1 = (
                dt.Frame(df_prob)[:,
                dt.sum(dt.f.prob), dt.by("item_id", "resp_id", "model_category")]).to_pandas()
            if self.verbose >= 1:
                print("add combo items : ", time.time() - st)
            df_prob = df_prob1
            # df_prob.to_csv("df_prob_last_Check1.csv")
        return df_prob

    def _set_scenario_prices_(self, prices, mul_quant=False):
        """Create dict based on input prices.

        Returns
        -------
            dict with item prices maped to item ids
        """
        if mul_quant:
            dict_item_old = dict(
                zip(self.df_item.item_id_old, prices)
            )
            self.df_item["price_scenario"] = self.df_item.item_id_old.map(dict_item_old)
        else:
            self.df_item["price_scenario"] = prices
        
        self.dict_item_prices = dict(
            zip(self.df_item.item_id, self.df_item.price_scenario)
        )

    def _set_log_transform_num(self):
        """Compute log values

        Returns
        -------
            dict with item prices maped to item ids
        """
        df_temp = pd.DataFrame()
        df_temp = self.df_item[["item_id"] + self.price_cols]
        df_temp["Input"] = df_temp.item_id.map(self.dict_item_prices)

        def _log_interpolation(x):
            inp = x[-1]
            for i in range(1, len(x) - 1):
                if inp == x[i]:
                    return np.log(x[i])
                if inp > x[i] and inp < x[i + 1]:
                    cal = (
                        (np.log(x[i]) * (x[i + 1] - inp))
                        + (np.log(x[i + 1]) * (inp - x[i]))
                    ) / (x[i + 1] - x[i])
                    return cal

        df_temp["log_const"] = df_temp.apply(lambda x: _log_interpolation(x), axis=1)
        self.dict_log_transform = dict(zip(df_temp.item_id, df_temp.log_const))

    def compute_scenario(self, prices, multi=0, bygrain=False, mul_quant=True, verbose=1):
        """Compute overall gc,item level units and revenue for given set of prices

        Parameters
        ----------
        prices : list
            list of scenario prices of all items
        bygrain : bool, default = False
            When True, price is computed at granualar level like cluster/item otherwise computes at overall level.
        mul_quant : bool
            When True, multiple quantity items are grouped into single item

        Returns
        -------
        df_sops: DataFrame
            Respondent * Item level probabilities
        df_item: DataFrme
            units revenue and other item level metrics
        """
        self.verbose = verbose
        #prices[19]=prices[18]     #This rule is specifically for Guatemala Instore Market. Please comment it out for other markets.
        #prices[20]=prices[18]     #This rule is specifically for Guatemala Instore Market. Please comment it out for other markets.
        if (self.s_vendor == "KSNR") or (self.s_vendor == "Tiger"):
            self.s_wap_scenario = self.compute_wap("price_scenario", "units_base")
            df_probs, df_prob_nobuy = self.compute_prob(prices)
            
            df_probs = df_probs.merge(
                self.df_item_resp_static,
                on=["item_id", "resp_id", "model_category"],
                how="inner",
            )
            df_probs = df_probs.merge(
                self.df_item[["item_id", "price_scenario"]], on=["item_id"], how="inner"
            )
            df_probs["share"] = df_probs.prob / df_probs.prob_resp_base
            df_probs["gc_num"] = df_probs.gc_fact * df_probs.price_scenario
            df_probs["gc_resp_new"] = df_probs.gc_resp * (
                1
                + (
                    df_probs.gc_coeff
                    * (
                        df_probs.groupby(["resp_id"]).gc_num.transform("sum")
                        / df_probs.groupby(["resp_id"]).gc_dem.transform("sum")
                        - 1
                    )
                )
            )
            self.s_gc_scenario = (
                df_probs.gc_resp_new * df_probs.weight_resp
            ).sum() / df_probs["item_id"].nunique()

        else:
            print('Multi:',multi)# This is SKIM
            if not multi:   
                quant = pd.DataFrame(self.df_item.groupby(['item_id_old'],as_index=False,sort=False)['mul'].max())
                #print(quant)
                prices_ = pd.Series()
                #print(prices)
                for i in range(len(prices)):
                    for j in range(int(quant.iloc[i]['mul'])):
                        prices_ = prices_.append(pd.Series(prices[i]))
                prices = prices_.reset_index(drop=True)
            df_probs, df_prob_nobuy,df_prob_nobuy_gc = self.compute_prob(prices, ignore_crosseff=True) #change
            # df_probs.to_csv('df_probs.csv')
            st = time.time()
            df_probs = df_probs.merge(
                self.df_resp, on=['resp_id'], how='left')
            df_prob_nobuy_gc = df_prob_nobuy_gc.merge(
                self.df_resp[['resp_id', 'basket_size']], on=[
                    'resp_id'], how='left'
            ) # flag df_prob_nobuy_gc is used for gc related calculation
            if self.verbose >= 1 :
                print('mege resp data : ', time.time() - st)
            st = time.time()
            #temp=df_prob_nobuy.loc[df_prob_nobuy['model_category']==10] #change
            #del_gc=temp.apply(lambda row: row['basket_size'] if row['nobuy_prob'] > 0.5 else 0,axis=1).mean()#change
            #print("Delivery GC: ",del_gc)
            
            if self.excluded_category:
                df_prob_nobuy_gc.loc[df_prob_nobuy_gc['model_category'].isin(self.excluded_category), 'nobuy_prob'] = 999
            # df_prob_nobuy.to_csv("df_prob_nobuy.csv")    
            # df_prob_nobuy_8 = df_prob_nobuy_gc[df_prob_nobuy['model_category']!=8]
            # df_prob_nobuy_gc.groupby(['resp_id'])[
            #     'nobuy_prob', 'basket_size'].min().apply(lambda row: row['basket_size'
            # ] if row['nobuy_prob'] < 0.5 else 0, axis=1).to_csv('nobuy.csv')
            
            
            gc_fact = df_prob_nobuy_gc.groupby(['resp_id'])[
                 'nobuy_prob', 'basket_size'].min().apply(lambda row: row['basket_size'
             ] if row['nobuy_prob'] < 0.5 else 0, axis=1).sum()/self.sum_resp_wt # flag df_prob_nobuy_gc is used for gc calculation.
             
                                                                         
                                                                         
            # gc_fact = df_prob_nobuy_gc.groupby(['resp_id'])[
            #     'nobuy_prob', 'basket_size'].min().apply(lambda row: row['basket_size'
            # ] if row['nobuy_prob'] < 0.5 else 0, axis=1).mean(
            # ) # flag df_prob_nobuy_gc is used for gc calculation.
            
            print("\ngc_base: ",self.s_gc_base)
            print("gc_fact: ",gc_fact)

            #gc_fact=(gc_fact+del_gc)/2 #change
            gc_fact=gc_fact/self.s_gc_base #change
            print("Final GC: ",gc_fact,"\n")

            self.s_gc_scenario = gc_fact #change
            df_probs["gc_resp_new"] = self.s_gc_scenario
            df_probs['share'] = df_probs['prob']
            df_probs['weight_item'] = 1
            if self.verbose >= 1 :
                print('GC compute : ', time.time() - st)

        self.df_sops = df_probs
        self.df_prob_nobuy = df_prob_nobuy

        if bygrain == False:
            st = time.time()
            #df_probs.loc[df_probs['resp_id'].isin(self.resp_id_list),'basket_size']=0
            self.df_item["units_scenario"] = self.df_item.item_id.map(
                df_probs.eval("col = gc_resp_new* share* basket_size* weight_item")
                .groupby("item_id")
                .col.sum()
            )
            self.df_item["units_scenario"] = (
                self.df_item["units_scenario"] * self.df_item["item_calib_factor"] * self.df_item["mul"]
            )
            # self.df_item.to_csv("df_item.csv")
            #self.df_item.loc[self.df_item.item_id==56,'units_scenario'] = 0 #change
            self.s_revenue_scenario = self.compute_revenue(
                "price_scenario", "units_scenario"
            )
            self.s_revenue_scenario_item = self.compute_revenue(
                "price_scenario", "units_scenario",bygrain=True
            )
            self.s_gm_scenario = self.compute_gm()
            self.s_gm_scenario_item = self.compute_gm(bygrain=True)
            if not mul_quant:
                self.df_item_act = (self.df_item.groupby(
                    ['item_id_old'], as_index=False)['units_scenario', 'revenue_scenario'].sum()
                )
                self.df_item_act = (self.df_item_act.merge(self.df_item[
                    ['item_id_old', 'price_base', 'item_name']], by='item_id_old').rename({'item_id_old': "item_id"})
                )
            if self.verbose >= 1 :
                print('units compute : ', time.time() - st)
        else:
            self._compute_units_gc_bygrain()
            
        #df_probs.to_csv("df_prob.csv")

    def compute_delta(
        self, stat_list=["wap", "gc", "revenue", "units"], delta_type="abs"
    ):
        """Compute Prob.

        Parameters
        ----------
        stat_list : list
            list of metrics for which delta needs to be computed
        delta_type : string, default = 'abs'
            deta type for metric computation

        Returns
        -------
        out : dict
            dict with computed detas
        """
        out = {}
        for stat_ in stat_list:
            if stat_ == "wap":
                if delta_type == "abs":
                    out[stat_] = self.s_wap_scenario - self.s_wap_base
                if delta_type == "per":
                    out[stat_] = self.s_wap_scenario / self.s_wap_base - 1
            elif stat_ == "gc":
                if delta_type == "abs":
                    out[stat_] = self.s_gc_scenario - self.s_gc_base
                if delta_type == "per":
                    out[stat_] = self.s_gc_scenario / self.s_gc_base - 1
            elif stat_ == "revenue":
                if delta_type == "abs":
                    out[stat_] = self.s_revenue_scenario - self.s_revenue_base
                if delta_type == "per":
                    out[stat_] = self.s_revenue_scenario / self.s_revenue_base - 1
            elif stat_ == "units":
                if delta_type == "abs":
                    out[stat_] = np.array(
                        self.df_item["units_scenario"] - self.df_item["units_base"]
                    )
                if delta_type == "per":
                    out[stat_] = (
                        np.array(
                            self.df_item["units_scenario"] / self.df_item["units_base"]
                        )
                        - 1
                    )
        return out

    def _compute_units_gc_bygrain(
        self, group_level=["item_id", "cluster", "model_category"]
    ):
        """Compute units and GC by grainular level

        Returns
        -------
            Dataframe with units Revenue GC metrics at desired granular level
        """
        # Get Demographic and cluster
        df_temp_grain = self.df_sops
        cols = [col for col in group_level if col not in df_temp_grain]
        df_temp_grain = self.df_sops.merge(
            self.df_resp[cols + ["resp_id"]],
            on="resp_id",
            how="inner",
            validate="m:1",
        )
        df_temp_grain["gc"] = (
            df_temp_grain["gc_resp_new"] * df_temp_grain["weight_resp"]
        )
        df_temp = pd.DataFrame(
            df_temp_grain.assign(
                col=df_temp_grain.gc_resp_new
                * df_temp_grain.share
                * df_temp_grain.basket_size
                * df_temp_grain.weight_item
            )
            .groupby(group_level)
            .col.sum()
        )
        df_temp = df_temp.merge(
            pd.DataFrame(df_temp_grain.groupby(group_level)["gc"].sum()),
            left_index=True,
            right_index=True,
            validate="1:1",
        )
        df_temp.rename(columns={"col": "units"}, inplace=True)
        df_temp["price_scenario"] = df_temp.index.get_level_values("item_id").map(
            self.dict_item_prices
        )
        df_temp = df_temp.reset_index()
        self.df_by_grain = df_temp
        self.df_by_grain["revenue"] = self.compute_revenue(
            "price_scenario", "units", bygrain=True
        )