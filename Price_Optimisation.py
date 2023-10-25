# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 18:35:12 2022

@author: atul.poddar
"""

import os
import re
import math
import pandas as pd
import numpy as np
from openpyxl import load_workbook
import warnings
from ast import literal_eval
from collections import OrderedDict
from conjoint_simulator import ConjointSimulator
# from conjoint_simulator2 import ConjointSimulator2
from conjoint_optimiser import ConjointOptimizer
import time
import sys
from con_opt_lib import utils as ut
#import data_preprocessing as dp
import logging
import logging.handlers
from datetime import datetime
from datetime import timedelta
from Laddering_rule_v2 import LADDER_RULES 
warnings.filterwarnings("ignore")
def readF(param):
    global cfg
    return cfg[param].format(**cfg)


def readC(param):
    global cfg
    return cfg[param]

def get_log(market,Segment):
    now = datetime.now()
    newDirName = now.strftime("%Y%m%d_%H%M")
    if not os.path.exists("../Outputs/Log/"):
        os.makedirs("../Outputs/Log/")
    filename = "../Outputs/Log/" + market + "_" + Segment + "_" + newDirName + ".log"
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s , %(message)s", datefmt="%Y-%m-%d  %H:%M:%S")
    handler = logging.handlers.RotatingFileHandler(filename, mode="w", backupCount=5, delay=True)
    should_roll_over = os.path.isfile(filename)
    if should_roll_over:  # log already exists, roll over!
        handler.doRollover()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    return logger
tjob=0
cfg = ut.load_config_from_path(r"D:\Guatemala_Client\Codes\config_skim1.yml")
gc=(cfg["CALIB_GC"]/cfg["SUM_RESP_WT"])  #(GC used for calibration*GC factor)/sumof resp
base_gc=cfg["BASE_GC"]
excl_cat= cfg["EXCLUDED_MODEL_CATEGORY"]
#%%
df_item = pd.read_csv(cfg["INPUT_DF_ITEM"].format(**cfg),engine='python')
df_item_gc=df_item.copy(deep=True)
price_cols=[i for i in df_item.columns if re.match("p_[0-9]+",i)]
df_item=df_item.iloc[1:].reset_index(drop=True)
df_item.columns = ["item_id_old","item_id","include/exclude","item_name","Category","price_base"]+price_cols+["units_base","item_calib_factor","fp_cost","is_combo","combo_item_id","combo_map","base_GC"]
df_item['item_calib_factor'] = df_item['item_calib_factor']*gc
df_item['combo_map'] = df_item.combo_map.apply(lambda x: literal_eval(str(x)) if pd.notnull(x) else print('',end=''))
df_item['combo_map'] = df_item['combo_map'].fillna(value=np.nan)
df_item[price_cols]=df_item[price_cols].replace(999,np.nan)
#%%Appending multiple quantities to df_item #flag
df_item['mul']=df_item.groupby('item_name').cumcount()+1

#%%
# Respondent
df_resp = pd.read_csv(cfg["INPUT_DF_RESP"].format(**cfg))
df_resp.columns = ["resp_id",'basket_size','Segment']
df_resp['resp_id'] = df_resp['resp_id'].astype(int)
df_resp['gc_coeff'] = 0
df_resp['gc_resp'] =1

# Model
df_model = pd.read_csv(cfg["INPUT_DF_MODEL"].format(**cfg))
df_model.item_id.replace({0:-99},inplace=True)
df_model.dropna(subset=['resp_id'],inplace=True)
df_model_none=df_model.loc[df_model['item_id']==-99,:]
#df_model_none['item_id']=-99
'Asc update for GC----------'
respondent=df_model.resp_id.unique()
models=df_model.model_category.unique()
for model in models:
    df_asc=pd.DataFrame()
    df_asc['resp_id']=respondent
    df_asc['estimate_type']='asc'
    df_asc['estimate_value']=0
    df_asc['cross_effect_item_id']=np.nan
    df_asc['pw_item_id']=np.nan
    df_asc['model_category']=model
    df_asc['item_id']=-99
    df_asc['model_calib_factor']=1
    df_model_none=df_model_none.append(df_asc,ignore_index=True)

df_model=df_model.loc[df_model['item_id']!=-99,:]
df_model.loc[:,'estimate_type']=df_model.estimate_type.str.replace(" IV",'')
df_model.estimate_type.replace({'partworth-2 AV':'av'},inplace=True)
df_model_none.estimate_type.replace({'partworth-2 AV':'av'},inplace=True)
df_model=df_model.loc[df_model['estimate_value']!=0,:].reset_index(drop=True)
df_model.drop_duplicates(inplace=True)
# df_model.to_csv('df_model.csv')
# df_none_all.to_csv('df_none_all_old.csv')
#%%
# df_none_all_new= pd.read_csv('D:\Guatemala_2022\Delivery\Codes\df_none_all.csv')
# df_model_none = pd.read_csv(r'D:\Guatemala_2022\Delivery\Codes\None_model.csv')
#%%
'''
Data Preparation for Estimate Multiplier Calculation for GC
This block is required for calculation of estimate multiplier
'''
#df_item_gc = pd.read_csv(cfg["INPUT_DF_ITEM"].format(**cfg),engine='python')
df_item_gc=df_item_gc.iloc[1:]
df_for_gc_ll=pd.read_csv(cfg["INPUT_DF_MODEL_GC"].format(**cfg),engine='python')
df_for_gc_ll=df_for_gc_ll[df_for_gc_ll['estimate_type']!='partworth-2 AV']
df_for_gc_ll.drop(columns='estimated_mul',inplace=True)
df_for_gc_ll['item_id'].replace({0:-99},inplace=True)
df_for_gc_ll.rename(columns={'model_category':'Model_category','cross_effect_item_id':'Cross_Effect_Item_id','estimate_type':'Estimate_type','item_id':'Item_id'},inplace=True)
df_for_gc_ll['Cross_Effect_Item_id']=df_for_gc_ll['Cross_Effect_Item_id'].astype(float)
df_for_gc_ll['Model_category']=df_for_gc_ll['Model_category'].astype(float)
df_for_gc_ll['Item_id']=df_for_gc_ll['Item_id'].astype(object)

#%%
df_item_gc=df_item_gc[df_item_gc['is_combo']!=1]
df_item_gc[price_cols]=df_item_gc[price_cols].replace(999,np.nan)
df_item_gc['AveragePrice']=df_item_gc[price_cols].mean(axis=1)
df_item_gc=df_item_gc[['item_id']+price_cols+['AveragePrice']]
df_item_gc['Max-Min']=df_item_gc[price_cols].max(axis=1)-df_item_gc[price_cols].min(axis=1)
dict_min_max=df_item_gc.groupby('item_id')['Max-Min'].mean().to_dict()
dict_average=df_item_gc.groupby('item_id')['AveragePrice'].mean().to_dict()
df_for_gc_ll['Max-Min']=df_for_gc_ll.Cross_Effect_Item_id.map(dict_min_max)
df_for_gc_ll['AveragePrice']=df_for_gc_ll.Cross_Effect_Item_id.map(dict_average)
df_for_gc_ll['Estimated_mul']=(df_for_gc_ll['Max-Min']*df_for_gc_ll['Multiplier'])+df_for_gc_ll['AveragePrice']
df_for_gc_ll['Estimated_mul']=df_for_gc_ll['Estimated_mul'].astype('float64')
df_for_gc_ll.drop_duplicates(subset=['Model_category', 'Estimate_type', 'Cross_Effect_Item_id',
        'Multiplier','Item_id'],keep='first',inplace=True)
#%%
start = time.time()
df_for_gc_ll['Estimated_mul']=df_for_gc_ll['Estimated_mul'].astype('int64')
Segment=readC("SUB_SEGMENT")
print(Segment)
if Segment in df_resp['Segment'].unique():
    resp_id_list = list(df_resp.loc[df_resp['Segment'] == Segment , 'resp_id'])
    #df_resp.loc[df_resp['resp_id'].isin(resp_id_list),'basket_size']=0
    df_resp = df_resp[df_resp['resp_id'].isin(resp_id_list)]
    df_model = df_model[df_model.item_id.notnull()]
    df_model = df_model[df_model['resp_id'].isin(resp_id_list)]
    df_model_none = df_model_none[df_model_none['resp_id'].isin(resp_id_list)]
    


obj = ConjointSimulator(df_item, df_resp, df_model, base_gc,cfg["SUM_RESP_WT"], 'linear', 'SKIM', excl_cat, df_model_none,df_for_gc_ll=df_for_gc_ll) #flag = Passing gc related data frame
# obj = ConjointSimulator2(df_item, df_resp, df_model, 0.649694501, 'linear', 'SKIM', excl_cat, df_none_all_new)
print(time.time()-start)
#%%
#%%
if readC("START_OPTIMIZER"):
    time2 = time.time()
    logger = get_log(cfg['MARKET'], cfg['SEGMENT'])
    try:
        ladd_rule=ut.load_config_from_path(r'toyaml.yml')
        print("Ladder Rule File present")
    except:
        ladder=pd.read_excel(readF("LADDER_LOC"))
        LADDER_RULES(ladder)
        ladd_rule=ut.load_config_from_path(r'toyaml.yml')
        print("Ladder rule File not present")
    def path_exist(csvpath):
        return os.path.exists(csvpath)

    def _scen_order(cfg):
        indepen = {}
        scen_order = []
        for i in cfg:
            if len(cfg[i]['DEPENDENCY']) == 0:
                scen_order.append(i)
            else:
                indepen[i] = cfg[i]['DEPENDENCY']
        print('-------------Hi-------------')
        print(scen_order)
        r = dict(reversed(list(indepen.items())))
        print(r)

        for key, val in indepen.items():
            if set(val).issubset(set(scen_order)):
                scen_order.append(key)
        for key, val in r.items():
            if set(val).issubset(set(scen_order)):
                scen_order.append(key)
        return list(OrderedDict.fromkeys(scen_order))

    scen_order = _scen_order(readC("CFG_CONSTRAINTS"))
    [scen_order.append(i) for i in readC("CFG_CONSTRAINTS") if i not in scen_order]
    order = readC("ORDER")
    if readC("VENDOR") == 'SKIM':
        base = obj.df_item.groupby(['item_name', 'Category', 'price_base'], as_index=False, sort=False).sum()['price_base'].copy(deep=True)
    else:
        base = obj.df_item['price_base'].copy(deep=True)
    # base.to_csv('base.csv')
    #print(base)
    inp_opt = pd.read_excel(readF('INPUT_OPT'))
    print(scen_order)
    logger.info('Optimization Method --- : --- %s', order)
    for scen in [1]:
        print('===========================================================================================================\n')
        print('scenario' + " " + str(scen) + " " + 'started ------')
        logger.info("Scenario %s started", str(scen))
        rho = 'R' + str(readC("CFG_CONSTRAINTS")[scen]['RHO'])
        item_hold = readC("CFG_CONSTRAINTS")[scen]['ITEMS_TO_HOLD']
        if len(readC("CFG_CONSTRAINTS")[scen]['DEPENDENCY']) > 0:
            best_rev = {}
            for j in readC("CFG_CONSTRAINTS")[scen]['DEPENDENCY']:
                filepath = os.path.join('../Outputs', order, str(j) + ".csv")
                if path_exist(filepath):
                    latest_rev = pd.read_csv(filepath)
                    best_rev[j] = latest_rev['Revenue'].sum()
                else:
                    best_rev[j] = 0.0
            best = max(best_rev.items(), key=lambda k : k[1])
            print('-----------------Hi2-------------------')
            print(best)
            if best[1] > 0:
                filepath = os.path.join('../Outputs', order, str(best[0]) + ".csv")
                if path_exist(filepath):
                    inp = pd.read_csv(filepath)
                    if readC("VENDOR") == 'SKIM':
                        data = {}
                        ind = inp['Input'].size
                        index = [*range(1, ind + 1)]
                        for i in index:
                            data[i] = inp['Input'][i - 1]
                        item_old = obj.df_item['item_id_old']
                        pri = item_old.map(data)
                        obj.df_item.loc[:, 'price_base'] = pri
                    else:
                        obj.df_item.loc[:, 'price_base'] = inp['Input']
                else:
                    obj.df_item.loc[:, "input"] = obj.df_item.loc[:, "price_base"]
            else:
                obj.df_item.loc[:, "input"] = obj.df_item.loc[:, "price_base"]
        else:
            obj.df_item.loc[:, "input"] = obj.df_item.loc[:, "price_base"]
        if readC("VENDOR") == 'SKIM':
            input_optimiser = obj.df_item.groupby(['item_name', 'Category', 'price_base'], as_index=False, sort=False).sum()[['item_name', 'price_base']].copy(deep=True)
        else:
            input_optimiser = obj.df_item[['item_name', 'price_base']].copy(deep=True)
        input_optimiser.insert(loc=0, column='index', value=input_optimiser.index.values)
        input_optimiser['Revenue'] = inp_opt['Revenue']
        input_optimiser['Elasticity'] = inp_opt['Elasticity']
        print('Optimization Method --- : --- ', order)
        if order == 'Revenue':
            input_optimiser.sort_values(by=order, inplace=True, ascending=False)
        elif order == 'Elasticity':
            input_optimiser.sort_values(by=order, inplace=True, ascending=False)
        elif order == 'Custom':
            # if path_exist(readF('CUSTOM_ORDER_LIST')):
            if path_exist('../Metrics/Overall_Metrics/Spectrum.csv'):
                # item_list=pd.read_csv(readF('CUSTOM_ORDER_LIST'))
                item = pd.read_csv('../Metrics/Overall_Metrics/Spectrum.csv')
                item = item[(item['%_Change_in_Units'] != 0) & (item['%_Change_in_Revenue'] > 0)]
                item['%_Change_in_Rev/Units'] = item['%_Change_in_Revenue'] / item['%_Change_in_Units']
                item.sort_values(by=['%_Change_in_Rev/Units'], inplace=True)
                item_list = pd.DataFrame()
                l1 = pd.Series(item_list['Product_Name'].unique())
                l1 = l1.tolist()
                miss_item1 = (set(input_optimiser['Name'])).difference(set(l1))
                l1 = l1 + list(miss_item1)
                item_list['Product name'] = l1
                miss_item = (set(input_optimiser['item_name'])).difference(set(item_list['Product name']))
                if len(miss_item):
                    print('Missing or misspelled items in Custom list : ', miss_item)
                    sys.exit()
                else:
                    input_optimiser['item_name'] = input_optimiser['item_name'].astype("category")
                    input_optimiser['item_name'].cat.set_categories(item_list['Product name'].tolist(), inplace=True)
                    input_optimiser.sort_values(['item_name'], inplace=True)
            else:
                print('=========================================================================+')
                print("Please Provide custom item list file")
                sys.exit()
        opt = ConjointOptimizer(obj, input_optimiser, readC("CFG_ALGO"), readC("CFG_CONSTRAINTS")[scen], readC("VENDOR"), rho, base, logger, ladd_rule,item_hold) #ladd_rule["LADDER_RULES"]
        titer = readC("CFG_ALGO")["MAXITER"] * len(readC("CFG_ALGO")["RHO"][rho])
        tjob = tjob + titer
        logger.info('Total Iteration for this scenario: %s', str(titer))
        logger.info("Scenario %s Completed\n", str(scen))
        if not os.path.exists("../Outputs/" + str(order) + "/"):
            os.makedirs("../Outputs/" + str(order) + "/")
        opt.scen_output.to_csv(os.path.join('../Outputs', order, str(scen) + ".csv"))
    logger.info('-------------***************-----------------------')
    logger.info('Total Optimisation Job : %s', str(tjob))
    logger.info(f"Total Duration : {timedelta(seconds = time.time()-time2)}")
    logger.info("Server log path : " + readC('MARKET') + '\\' + readC('SEGMENT') + '\\' + 'Outputs' + '\\' + 'Log')
    sys.exit()
#%%
def cross_effect(sk,dict_nameid):

    temp = pd.DataFrame()
    temp["item_id"] = sk['item_id'].map(dict_nameid)
    temp['cross'] = sk["cross_effect_item_id"].map(dict_nameid)
    temp = temp.drop_duplicates(keep="first")
    temp = temp[pd.notnull(temp['cross'])]


    cross_effect_pair = list(temp.itertuples(index=False, name=None))
    return cross_effect_pair
dict_itemid = dict(zip(df_item.item_name, df_item.item_id))
dict_nameid = dict(zip(dict_itemid.values(), dict_itemid.keys()))
# del dict_nameid[260]
temp = obj.df_item_resp
sk = temp[temp.item_id<100]
cross_effect_pair = cross_effect(sk,dict_nameid)
print("Cross effect pairs created.")
#del cross_effect_pair[-1]
res = [t for t in cross_effect_pair if not any(isinstance(n, float) and math.isnan(n) for n in t)]
cross_effect_pairs = [k for k in res if "Choice" not in k[0]]
#%%
def OBJECTIVE_FUNCTION(price): #flag
    global obj, df
    obj.compute_scenario(price,bygrain=False)
    df=obj.df_item.copy()
    #print(df.columns)
    df.rename(columns={'price_scenario':'Input','units_scenario':'Units','item_name':'Name'},inplace=True)
    df['Revenue'] = (df['Input'] * df['Units']) / 1000
    df['GM'] = (df['Revenue']  - (df['fp_cost'] * df['Units'])/1000 )
    df=df.groupby(['Name','Category','Input'], as_index=False, sort=False).sum()
    df['GC'] = obj.s_gc_scenario *cfg["CALIB_GC"]
    #print(df.columns)
    #print(temp)
    return df[['Name','Category','Input','Revenue','Units','GM','GC']]
#%%
start = time.time()
df=test=pd.read_excel(r"D:/Guatemala_Client/Input_Files/Test.xlsx",engine='openpyxl')
summary = OBJECTIVE_FUNCTION(df['Input'].fillna(0))
a=pd.Series(df_item.loc[df_item['is_combo']!=1,'price_base'])
#summary.to_csv('Base.csv')
print(time.time()-start)
print("Units: {} \nRevenue: {}\nGM: {}\nGC: {}".format(summary["Units"].sum(),summary["Revenue"].sum(),summary["GM"].sum(),summary["GC"].iloc[0]))
