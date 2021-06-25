# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from types import SimpleNamespace

from eevalue.postprocessing import pyomo_to_pandas
from eevalue.data_handlers.build import compile_dataset
from eevalue.gradient_boosting import BoostedTreeRegressor
from eevalue.simulation import SimData, create_ucm, run_solver



class Market:
    def __init__(self):
        pass 


    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError










def create_features(net_load, available_capacity, water_level=None, alpha=0.85):
    X = []

    total_available = available_capacity.sum(axis=1)
    X.append(net_load.divide(total_available, axis=0).to_frame('Margin'))

    for tech in available_capacity.columns:
        X.append(available_capacity[tech].divide(total_available, axis=0)
                                         .to_frame(f'{tech}_capacity'))
   
    ramps = net_load - (net_load.shift(1) + net_load.shift(2) + net_load.shift(3))/3
    ramp_up = ramps[ramps >=ramps[ramps>0].quantile(alpha)]
    ramp_down = ramps[ramps <=ramps[ramps<0].quantile(1-alpha)]

    ramps = ramps.to_frame('Ramping')
    ramps['RU'] = 0
    ramps['RD'] = 0
    ramps.loc[ramps.index.isin(ramp_up.index), 'RU'] = 1
    ramps.loc[ramps.index.isin(ramp_down.index), 'RD'] = 1
    X.append(ramps.drop('Ramping', axis=1))
    
    if water_level is not None:
        if isinstance(water_level, pd.Series):
            X.append(water_level.to_frame('WaterLevel'))
        else:
            X.append(water_level.rename(columns={water_level.columns[0]: 'WaterLevel'}))

    X = pd.concat(X, axis=1)
    return X


def create_target(dataset, committed_capacity, available_capacity, solver):
    if isinstance(available_capacity, pd.DataFrame):
        available_capacity = available_capacity.sum(axis=1)
    
    period = dataset['simulation_period']
    sim_data = SimData(calibration=True, **dataset) 
    sets, parameters = sim_data.build_sim_data()
    instance = create_ucm(sets, parameters, calibration=True)
    instance = run_solver(instance, solver=solver)

    predicted_generation_cl = pyomo_to_pandas(instance, 'Power', dates=period)
    plants = dataset['clustered_plants']
    combinations = set(zip(plants['Technology'], plants['Fuel'])) 
    
    predicted_generation = None
    for item in combinations:
        t, f = item
        clusters = plants[(plants['Technology']==t) & (plants['Fuel']==f)]['Cluster'].tolist()
        pred = pd.Series(0, index=period)
        for cl in clusters:
            pred += predicted_generation_cl[cl]
        predicted_generation = pd.concat([predicted_generation, pred.to_frame(f'{t}_{f}')], axis=1)

    error = predicted_generation - committed_capacity
    return error.divide(available_capacity, axis=0).astype('float')


def create_input_data(config, period, has_hydro=True):
    solver = config['Solver']
    exclude = ['water_level'] if not has_hydro else None
    dataset = compile_dataset(config, period, exclude=exclude)
    committed_capacity = dataset.pop('committed_capacity')
    available_capacity = dataset.pop('available_capacity')
    net_load = dataset['demand']['DA'] - dataset['res_generation']

    water_level = None 
    if has_hydro:
        water_level = dataset.pop('water_level')

    return SimpleNamespace(dataset=dataset, 
                           committed_capacity=committed_capacity,
                           available_capacity=available_capacity,
                           net_load=net_load,
                           water_level=water_level,
                           solver=solver,
    )
    

def learn_model_error(X, y, n_trials=10, verbose=False):
    loss_function = 'MultiRMSE' if y.ndim > 1 else 'RMSE'
    model = BoostedTreeRegressor(loss_function=loss_function, cat_features=['RU', 'RD'])
    opt_results = model.optimize(X, y, n_trials=n_trials, verbose=verbose)
    model = model.fit(X, y, iterations=opt_results.best_iter)
    pred = model.predict(X)
    return SimpleNamespace(actual=y, predicted=pred, model=model)