# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob 

import numpy as np 
import pandas as pd 

from datetime import datetime

from eevalue.definitions import ROOT_DIR



def fillna(data, method=None):
    if np.any(data.isna()):
        if method is None:
            data = (data.groupby([lambda x: x.hour, lambda x: x.dayofweek])
                        .transform(lambda group: group.fillna(group.median())))
        else:
            data = data.fillna(method=method)
    return data 


match_dates     = lambda data, dates: pd.DataFrame(index=dates).join(data, how='left')
match_and_fill  = lambda data, dates, method=None: data.pipe(match_dates, dates).pipe(fillna, method=method)


def build_availability_factors(country, period):
    avail_factors = None
    path = os.path.join(ROOT_DIR, f"DataSets/Capacity/{country}")
    _, _, filenames = next(os.walk(path))

    for name in filenames:
        x = pd.read_csv(os.path.join(path, name), index_col=0, parse_dates=[0])
        x = x['Available capacity [MW]'] / x['Nominal capacity [MW]']
        x = x.to_frame(name.split('.')[0])
        avail_factors = pd.concat([avail_factors, match_and_fill(x, period, method='ffill')], axis=1) 
    
    return avail_factors


def build_committed_capacity(country, period, include_res_gen=True):
    res_generation = None
    committed_capacity = None
    
    path = os.path.join(ROOT_DIR, f"DataSets/Generation/{country}")
    _, _, filenames = next(os.walk(path))
    
    for name in filenames:
        if 'RES' in name:
            if include_res_gen:
                res_generation = pd.read_csv(os.path.join(path, name), index_col=0, parse_dates=[0]) 
                res_generation = match_and_fill(res_generation, period).iloc[:, 0]
                continue
            else:
                continue
        x = pd.read_csv(os.path.join(path, name), index_col=0, parse_dates=[0])
        x = x.rename(columns={'Capacity committed [MW]': name.split('.')[0]})
        committed_capacity = pd.concat([committed_capacity, match_and_fill(x, period)], axis=1) 
    
    return res_generation, committed_capacity


def build_available_capacity(country, period):
    avail_capacity = None
    path = os.path.join(ROOT_DIR, f"DataSets/Capacity/{country}")
    _, _, filenames = next(os.walk(path))
    
    for name in filenames:
        x = pd.read_csv(os.path.join(path, name), index_col=0, parse_dates=[0])
        x = x.drop('Nominal capacity [MW]', axis=1)                                 \
                .rename(columns={'Available capacity [MW]': name.split('.')[0]})
        avail_capacity = pd.concat([avail_capacity, match_and_fill(x, period, method='ffill')], axis=1) 

    return avail_capacity 


def build_demand(country, period):
    path = os.path.join(ROOT_DIR, f"DataSets/Load/{country}")
    load = pd.read_csv(os.path.join(path, 'load.csv'), index_col=0, parse_dates=[0])
    if load.columns[0] != 'DA':
        load = load.rename(columns={load.columns[0]: "DA"})
    load = match_and_fill(load, period)

    reserves_file = glob.glob(os.path.join(path, "reserve") + '*')
    if reserves_file:
        resv = pd.read_csv(reserves_file[0], index_col=0, parse_dates=[0])
        resv_2U = match_and_fill(resv['2U'], period)
        resv_2D = match_and_fill(resv['2D'], period)
        demand = pd.concat([load, resv_2U, resv_2D], axis=1)
    else:
        resv = (load['DA'].groupby(lambda x: x.date).apply(lambda x: np.sqrt(10*x.max()+22500)-150))
        resv = match_and_fill(resv, period, method='ffill')
        demand = pd.concat([load, 
                            resv.rename(columns={resv.columns[0]: "2U"}), 
                            resv.multiply(0.5).rename(columns={resv.columns[0]: "2D"})], 
                        axis=1) 
    return demand



def compile_dataset(config, period, include=None, exclude=None):
    country = config['Country']

    if (include is not None) and not isinstance(include, list):
        include = [include]
    elif include is None:
        include = ['availability_factors', 'committed_capacity', 'res_generation', 
                   'available_capacity', 'demand', 'net_imports', 'plants',
                   'water_level', 'fuel_price', 'permit_price', 'no_load_cost']
    
    if (exclude is not None) and not isinstance(exclude, list):
        exclude = [exclude]
    elif exclude is None:
        exclude = []
    
    include = [x for x in include if x not in exclude]
    
    dataset = dict()
    dataset['simulation_period'] = period 

    # Availability factors
    if 'availability_factors' in include:
        dataset['availability_factors'] = build_availability_factors(country, period)
    
    # Historical data for generation except renewables
    if 'committed_capacity' in include:
        if 'res_generation' in include:
            res_generation, committed_capacity = build_committed_capacity(country, period)
            dataset['res_generation'] = res_generation
            dataset['committed_capacity'] = committed_capacity
        else:
            _, committed_capacity = build_committed_capacity(country, period, include_res_gen=False)
            dataset['committed_capacity'] = committed_capacity
     
    # Historical data for generation from renewables
    if ('res_generation' in include) and ('committed_capacity' not in include):
        path = os.path.join(ROOT_DIR, f"DataSets/Generation/{country}")
        res_generation = pd.read_csv(os.path.join(path, 'RES.csv'), index_col=0, parse_dates=[0])
        dataset['res_generation'] = match_and_fill(res_generation, period).iloc[:, 0]
    
    # Historical data for available capacity
    if 'available_capacity' in include:
        dataset['available_capacity'] = build_available_capacity(country, period)

    # Demand data
    if 'demand' in include:
        dataset['demand'] = build_demand(country, period)

    # Net imports
    if 'net_imports' in include:
        path = os.path.join(ROOT_DIR, f"DataSets/NetImports/{country}")
        imports = pd.read_csv(os.path.join(path, 'imports.csv'), index_col=0, parse_dates=[0])
        dataset['net_imports'] = match_and_fill(imports, period).iloc[:, 0]
    
    # Water levels
    if 'water_level' in include:
        path = os.path.join(ROOT_DIR, f"DataSets/Hydro/{country}")
        water_level = pd.read_csv(os.path.join(path, "level_coefficient_avg.csv"), 
                                    index_col=0, parse_dates=[0], na_values=0)
        dataset['water_level'] = match_and_fill(water_level, period, method='ffill').iloc[:, 0]
    
    # Plant clusters
    if 'plants' in include:
        path = os.path.join(ROOT_DIR, f"DataSets/PowerPlants/{country}")
        clustered_plants = pd.read_csv(os.path.join(path, 'clustered.csv'))
        dataset['clustered_plants'] = clustered_plants.drop('Units', axis=1)
        
        reserve_techs = dict()
        for tech in clustered_plants['Technology'].unique().tolist():
            reserve_techs.update({tech: 1 if tech in config['ReserveParticipation'] else 0})
        dataset['reserve_technologies'] = reserve_techs
    
    if 'fuel_price' in include:
        fuel_price = pd.DataFrame.from_dict(config['FuelPrice'], orient='index')
        fuel_price.index = fuel_price.index.map(lambda x: datetime(x, 1, 1 , 0))
        dataset['fuel_price'] = match_and_fill(fuel_price, period, method='ffill') 
    
    if 'permit_price' in include:
        permit_price = pd.DataFrame.from_dict(config['PermitPrice'], orient='index')
        permit_price.index = permit_price.index.map(lambda x: datetime(x, 1, 1 , 0))
        dataset['permit_price'] = match_and_fill(permit_price, period, method='ffill').iloc[:, 0] 
    
    if 'no_load_cost' in include:
        dataset['no_load_cost'] = config['NoLoadCost']
    
    return dataset

