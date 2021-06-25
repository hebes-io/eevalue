# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import dataclasses

import numpy as np 
import pandas as pd 

from typing import Optional
from itertools import product
from pydantic import validator
from pydantic.dataclasses import dataclass
from datetime import datetime, time, timedelta 

from pyomo.environ import SolverFactory
from pyomo.opt import TerminationCondition
from pyomo.environ import Objective, Param, Set, Var
from pyomo.environ import ConcreteModel, Constraint, minimize, NonNegativeReals


class ValidationConfig:
    arbitrary_types_allowed = True
    underscore_attrs_are_private = True


@dataclass(config=ValidationConfig)
class SimData:
    # Required time data
    simulation_period    : pd.DatetimeIndex
    
    # Required market data
    availability_factors : pd.DataFrame
    clustered_plants     : pd.DataFrame
    demand               : pd.DataFrame
    net_imports          : pd.Series
    reserve_technologies : dict
    
    # Required cost components
    fuel_price           : pd.DataFrame
    no_load_cost         : float
    permit_price         : pd.Series
    
    # Optional cost components
    markup               : Optional[pd.DataFrame] = None
    voll                 : Optional[float] = None
    
    # Renewables
    avail_factor_pv      : Optional[pd.Series] = None
    avail_factor_wind    : Optional[pd.Series] = None
    capacity_pv          : Optional[float] = None
    capacity_wind        : Optional[float] = None
    cost_curtailment     : Optional[float] = None
    res_generation       : Optional[pd.Series] = None
    
    # Initial state
    committed_initial    : Optional[dict] = None
    power_initial        : Optional[dict] = None
    storage_initial      : Optional[float] = None
    
    # Storage
    storage_charge_ef    : Optional[float] = None
    storage_discharge_ef : Optional[float] = None
    storage_capacity     : Optional[float] = None
    storage_charge_cap   : Optional[float] = None
    storage_min          : Optional[float] = None
    storage_final_min    : Optional[float] = None

    # Other
    calibration          : dataclasses.InitVar[bool] = False
    max_load_shaping     : Optional[float] = None
    
    
    def __post_init__(self, calibration):
        if calibration:
            if self.res_generation is None:
                raise ValueError('RES generation must be provided for calibration') 
        else:
            if self.avail_factor_pv is None:
                raise ValueError('Solar PV availability must be provided for what-if analysis')
            if self.avail_factor_wind is None:
                raise ValueError('Wind availability must be provided for what-if analysis')
            if self.capacity_pv is None:
                raise ValueError('Solar PV capacity must be provided for what-if analysis')
            if self.capacity_wind is None:
                raise ValueError('Wind capacity must be provided for what-if analysis')

    
    @validator('simulation_period')
    def check_no_dates_missing(cls, data):
        time_col = data.to_series()    
        time_step = time_col.diff().min()
        
        if time_step != pd.Timedelta('1H'):
            raise ValueError('`simulation_period` must have hourly time step')
        if len(data) < 24:
            raise ValueError('The `simulation_period` range should span at least one full day')
        
        full_index = pd.date_range(
                start=datetime.combine(time_col.min().date(), time(0, 0)), 
                end=datetime.combine(time_col.max().date() + timedelta(days=1), time(0, 0)),
                freq=time_step)[:-1]
        if len(full_index) > len(data):
            raise ValueError('Missing hours in `simulation_period`')
        return data
    

    @validator('availability_factors', 'avail_factor_pv', 'avail_factor_wind')
    def check_availability_factor(cls, data):
        if data is not None:
            if (data.values < 0).any() or (data.values > 1).any():
                raise ValueError('The availability factor\'s values are outside of the [0,1] interval')
        return data
    
    
    @validator('availability_factors', 'avail_factor_pv', 'avail_factor_wind', 'demand', 
               'res_generation', 'markup', 'net_imports', 'fuel_price', 'permit_price')
    def check_time_index(cls, data, values):         
        if data is not None:
            data.index = data.index.map(pd.to_datetime)
            time_step = data.index.to_series().diff().min()
            
            if time_step != pd.Timedelta('1H'):
                raise ValueError('The index of the dataframe has not an hourly time step')
            if not data.index.is_monotonic_increasing:
                raise ValueError('The index of the dataframe is not monotonic increasing')
            if not set(values['simulation_period']).issubset(data.index):
                raise ValueError('The `simulation_period` range is not a subset of the dataframe\'s index')
        return data

    
    @validator('clustered_plants')
    def check_expected_fields(cls, data): 
        expected_fields = ['Cluster', 'Nunits', 'PowerCapacity', 'PowerMinStable', 'Efficiency', 
                           'RampStartUpRate', 'CO2Intensity', 'MinDownTime', 'MinUpTime', 
                           'RampUpRate', 'RampDownRate', 'CostRampUp', 'Technology', 'Fuel']
        
        if not set(expected_fields).issubset(data.columns):
            raise ValueError('Missing fields from plant cluster dataframe')
        return data 
    

    def __post_init_post_parse__(self, calibration):
        self._sets = dict()
        self._sets['mk'] = ['DA', '2U', '2D']
        self._sets['cl'] = self.clustered_plants['Cluster'].tolist()
        self._sets['f']  = self.clustered_plants['Fuel'].unique().tolist()
        self._sets['t']  = self.clustered_plants['Technology'].unique().tolist()
        self._sets['h']  = (np.array((self.simulation_period - self.simulation_period[0].to_pydatetime())
                                .total_seconds()/3600).astype('int').tolist())
            
    @property
    def sets(self):
        return self._sets
    
    
    def build_cluster_data(self, parameters=None):
        if parameters is None:
            parameters = dict() 

        plants = self.clustered_plants.set_index('Cluster')

        parameters['Nunits'] = plants['Nunits'].to_dict()
        parameters['PowerCapacity'] = plants['PowerCapacity'].to_dict()
        parameters['PowerMinStable'] = plants['PowerMinStable'].to_dict()
        parameters['Efficiency'] = plants['Efficiency'].to_dict()
        parameters['EmissionRate'] = plants['CO2Intensity'].to_dict()
        parameters['TimeDownMinimum'] = plants['MinDownTime'].to_dict()
        parameters['TimeUpMinimum'] = plants['MinUpTime'].to_dict()        
        parameters['RampUpMaximum'] = plants['RampUpRate'].to_dict()
        parameters['RampDownMaximum'] = plants['RampDownRate'].to_dict()
        parameters['RampStartUpMaximum'] = plants['RampStartUpRate'].to_dict()
        parameters['CostRampUp'] = plants['CostRampUp'].to_dict()
        
        parameters['RampShutDownMaximum'] = (
            {cl: 0 for cl in self.sets['cl']} if 'RampShutDownRate' not in plants.columns else 
            plants['RampStartUpRate'].to_dict()
        )
        parameters['CostRampDown'] = (
            {cl: 0 for cl in self.sets['cl']} if 'CostRampDown' not in plants.columns else 
            plants['CostRampDown'].to_dict()
        )
        parameters['Technology'] = {
            (cl, t): 1 if plants.loc[cl,'Technology'] == t else 0
                for (cl, t) in product(self.sets['cl'], self.sets['t'])
        }
        parameters['Fuel'] = {
            (cl, f): 1 if plants.loc[cl,'Fuel'] == f else 0
                for (cl, f) in product(self.sets['cl'], self.sets['f'])
        }
        return parameters


    def build_market_data(self, parameters=None):
        if parameters is None:
            parameters = dict()

        parameters['Reserve'] = self.reserve_technologies
        parameters['NoLoadCost'] = self.no_load_cost
        parameters['ValueOfLostLoad'] = self.voll or 100E3
        parameters['MaxLoadShaping'] = self.max_load_shaping or 0
        return parameters 


    def build_time_series(self, parameters=None):
        if parameters is None:
            parameters = dict()

        def align_time_index(data):
            data = data.loc[self.simulation_period].reset_index(drop=True)
            if set(data.index).difference(self.sets['h']):
                raise RuntimeError('Dataframe index cannot match `h` index')
            return data

        plants = self.clustered_plants
        combinations = set(zip(plants['Technology'], plants['Fuel']))

        avail_factors = align_time_index(self.availability_factors)
        parameters['AF'] = {}
        for item in combinations:
            t, f = item
            clusters = plants[(plants['Technology']==t) & (plants['Fuel']==f)]['Cluster'].tolist()
            for cl in clusters:
                parameters['AF'].update({
                    (cl, h): val for h, val in avail_factors['_'.join([t, f])].to_dict().items()
                }) 

        demand = align_time_index(self.demand)
        parameters['Demand'] = {}
        parameters['Demand'].update({('DA', h): val for h, val in demand['DA'].to_dict().items()})
        parameters['Demand'].update({('2U', h): val for h, val in demand['2U'].to_dict().items()})
        parameters['Demand'].update({('2D', h): val for h, val in demand['2D'].to_dict().items()})
        
        net_imports = align_time_index(self.net_imports)
        parameters['NetImports'] = net_imports.to_dict()
        
        parameters['FuelPrice'] = dict()
        fuel_price = align_time_index(self.fuel_price)
        for fuel in self.sets['f']:
            parameters['FuelPrice'].update(
                {(fuel, h): val for h, val in fuel_price[fuel].to_dict().items()}
            ) 

        permit_price = align_time_index(self.permit_price)
        parameters['PermitPrice'] = permit_price.to_dict()

        parameters['Markup'] = dict()
        if self.markup is not None:
            markup = align_time_index(self.markup)
            for item in combinations:
                t, f = item
                clusters = plants[(plants['Technology']==t) & (plants['Fuel']==f)]['Cluster'].tolist()
                for cl in clusters:
                    parameters['Markup'].update({
                        (cl, h): val for h, val in markup['_'.join([t, f])].to_dict().items()
                    }) 
        else:
            for cl in self.sets['cl']:
                parameters['Markup'].update({(cl, h): 0 for h in self.sets['h']}) 
        
        if self.avail_factor_pv is not None:
            avail_factor_pv = align_time_index(self.avail_factor_pv)
            parameters['AF_Solar'] = avail_factor_pv.to_dict()
            
        if self.avail_factor_wind is not None:
            avail_factor_wind = align_time_index(self.avail_factor_wind)
            parameters['AF_Wind'] = avail_factor_wind.to_dict()
        
        if self.res_generation is not None:
            res_generation = align_time_index(self.res_generation)
            parameters['RES_Generation'] = res_generation.to_dict()

        return parameters
                

    def build_storage(self, parameters=None):
        if parameters is None:
            parameters = dict()

        parameters['StorageChargingEfficiency'] = self.storage_charge_ef or 1
        parameters['StorageDischargeEfficiency'] = self.storage_discharge_ef or 1
        parameters['StorageCapacity'] = self.storage_capacity or 0
        parameters['StorageChargingCapacity'] = self.storage_charge_cap or 0
        parameters['StorageMinimum'] = self.storage_min or 0
        parameters['StorageFinalMin'] = self.storage_final_min or 0
        
        return parameters


    def build_renewables(self, parameters=None):
        if parameters is None:
            parameters = dict()

        parameters['CostCurtailment'] = self.cost_curtailment or 0
        parameters['SolarCapacity'] = self.capacity_pv or 0
        parameters['WindCapacity'] = self.capacity_wind or 0
        
        return parameters
        
    
    def build_initial_state(self, parameters):
        parameters['CommittedInitial'] = (
            self.committed_initial if self.committed_initial is not None else
            {cl:val for cl, val in parameters['Nunits'].items()}
        )
        parameters['PowerInitial'] = (
            self.power_initial if self.power_initial is not None else  
            {cl: parameters['PowerCapacity'][cl] * parameters['Nunits'][cl] for cl in self.sets['cl']}
        )
        parameters['StorageInitial'] = self.storage_initial or 0
        
        return parameters


    def build_sim_data(self):
        parameters = self.build_cluster_data()
        parameters = self.build_market_data(parameters=parameters)
        parameters = self.build_time_series(parameters=parameters)
        parameters = self.build_storage(parameters=parameters)
        if not self.calibration:
            parameters = self.build_renewables(parameters=parameters)
        parameters = self.build_initial_state(parameters)
        return self.sets, parameters
        


def create_ucm(sets, params, load_shape_mode=None, calibration=False):
    """
    Creates a concrete Pyomo model for optimization. 
    
    Parameters
    ----------
    sets: Dictionary containing the model's sets (defined as a list of strings or integers).
    params: Dictionary containing the parameters of the optimization problem
    load_shape_mode: String indicating the presence and type of the load modifying resources.
        It can be None, 'savings', 'flex' 
    calibration: Boolean. Indicates whether the model will be used for calibration or what-if
        analysis
    
    Returns
    -------
    The Pyomo optimization model instance.

    Adapted from https://github.com/energy-modelling-toolkit/Dispa-SET/tree/v2.3
    
    """

    # Definition of model:
    model = ConcreteModel('UCM')

    ##############################################################################################
    ############################################ Definition of the sets ##########################
    ##############################################################################################

    model.h     = Set(initialize=sets['h'])   # Hours
    model.cl    = Set(initialize=sets['cl'])  # Generation clusters
    model.t     = Set(initialize=sets['t'])   # Generation technologies
    model.f     = Set(initialize=sets['f'])   # Fuel types
    model.mk    = Set(initialize=sets['mk'])  # Markets (DA: Day-Ahead, 2U: Reserve up, 2D: Reserve Down)
    

    ###############################################################################################
    ######################################### Definition of the Parameters ########################
    ###############################################################################################

    model.AF = Param(model.cl, model.h, initialize=params['AF']) # [%] Availability factor
    model.CommittedInitial = Param(model.cl, initialize=params['CommittedInitial']) 
    model.CostRampUp = Param(model.cl, initialize=params['CostRampUp']) # [EUR\MW] Ramp-up costs
    model.CostRampDown = Param(model.cl, initialize=params['CostRampDown']) # [EUR\MW] Ramp-down costs
    model.Demand = Param(model.mk, model.h, initialize=params['Demand']) # [MW] Demand
    model.Efficiency = Param(model.cl, initialize=params['Efficiency'])  # [%] Efficiency
    model.EmissionRate = Param(model.cl, initialize=params['EmissionRate'])  # [tCO2/MWh] CO2 emission rate
    model.Fuel = Param(model.cl, model.f, initialize=params['Fuel'])  # [n.a.] Fuel type {1 0}
    model.FuelPrice = Param(model.f, model.h, initialize=params['FuelPrice'])  # [€/MWh] Fuel price
    model.Markup = Param(model.cl, model.h, initialize=params['Markup'])  # [€/MW] EUR/MW
    model.MaxLoadShaping = Param(initialize=params['MaxLoadShaping']) # [MW]
    model.NetImports = Param(model.h, initialize=params['NetImports']) # [MW] Net imports
    model.NoLoadCost = Param(initialize=params['NoLoadCost']) # [€/MWh]
    model.Nunits = Param(model.cl, initialize=params['Nunits']) # Number of units inside the cluster
    model.PermitPrice = Param(model.h, initialize=params['PermitPrice']) # [EUR/ton] CO2 emission permit price
    model.PowerCapacity = Param(model.cl, initialize=params['PowerCapacity']) # [MW] Installed capacity
    model.PowerInitial = Param(model.cl, initialize=params['PowerInitial']) # [MW] Power output before initial period
    model.PowerMinStable = Param(model.cl, initialize=params['PowerMinStable'])  # [MW] Minimum power for stable generation
    model.RampDownMaximum = Param(model.cl, initialize=params['RampDownMaximum'])  # [MW/h] Ramp down limit
    model.RampUpMaximum = Param(model.cl, initialize=params['RampUpMaximum']) # [MW\h] Ramp up limit
    model.RampStartUpMaximum = Param(model.cl, initialize=params['RampStartUpMaximum']) # [MW\h] Start-up ramp limit
    model.RampShutDownMaximum = Param(model.cl, initialize=params['RampShutDownMaximum']) # [MW\h] Shut-down ramp limit
    model.Reserve = Param(model.t, initialize=params['Reserve'])  # [n.a.] Reserve technology {1 0}
    model.StorageChargingEfficiency = Param(initialize=params['StorageChargingEfficiency']) 
    model.StorageCapacity = Param(initialize=params['StorageCapacity'])
    model.StorageChargingCapacity = Param(initialize=params['StorageChargingCapacity']) 
    model.StorageDischargeEfficiency = Param(initialize=max(0.0001, params['StorageDischargeEfficiency']))
    model.StorageFinalMin = Param(initialize=params['StorageFinalMin'])
    model.StorageInitial = Param(initialize=params['StorageInitial'])
    model.StorageMinimum = Param(initialize=params['StorageMinimum']) 
    model.Technology = Param(model.cl, model.t, initialize=params['Technology'])  # [n.a] Technology type {1 0}
    model.TimeDownMinimum = Param(model.cl, initialize=params['TimeDownMinimum']) # [h] Minimum down time
    model.TimeUpMinimum = Param(model.cl, initialize=params['TimeUpMinimum']) # [h] Minimum up time
    model.ValueOfLostLoad = Param(initialize=params['ValueOfLostLoad'])
    
    def _var_cost_init(model, cl, h):
        return (
            model.Markup[cl, h]    
            + sum(model.Fuel[cl, f] * model.FuelPrice[f, h] / model.Efficiency[cl] for f in model.f)
            + model.PermitPrice[h] * model.EmissionRate[cl]
        )
    
    model.CostVariable = Param(model.cl, model.h, initialize=_var_cost_init) # [EUR\MWh] Variable costs
    
    if not calibration:
        model.AF_Solar = Param(model.h, initialize=params['AF_Solar']) # [%]
        model.AF_Wind = Param(model.h, initialize=params['AF_Wind']) # [%]
        model.CostCurtailment = Param(initialize=params['CostCurtailment']) # [EUR\MWh]
        model.SolarCapacity = Param(initialize=params['SolarCapacity']) # [MW] Installed capacity
        model.WindCapacity = Param(initialize=params['WindCapacity']) # [MW] Installed capacity
    else:
        model.RES_Generation = Param(model.h, initialize=params['RES_Generation']) # [MW] RES generation
        

    ################################################################################################
    ##########################################  Definition of variables ############################
    ################################################################################################

    # VARIABLES
    def _nunit_rule(model, cl, h):
        return (0, model.Nunits[cl])
    
    model.Committed = Var(model.cl, model.h, within=NonNegativeReals, bounds=_nunit_rule)
    model.CostRampUpH = Var(model.cl, model.h, within=NonNegativeReals) # [EUR/h] Cost of ramping up
    model.CostRampDownH = Var(model.cl, model.h, within=NonNegativeReals) # [EUR/h] Cost of ramping down
    model.LL_RampDown = Var(model.cl, model.h, within=NonNegativeReals) # [MW] Deficit in terms of ramping down 
    model.LL_RampUp = Var(model.cl, model.h, within=NonNegativeReals) # [MW] Deficit in terms of ramping up 
    model.LL_MaxPower = Var(model.h, within=NonNegativeReals) # [MW] Deficit in terms of maximum power
    model.LL_MinPower = Var(model.h, within=NonNegativeReals) # [MW] Power exceeding the demand
    model.LL_2U = Var(model.h, within=NonNegativeReals) # [MW] Deficit in reserve up
    model.LL_2D = Var(model.h, within=NonNegativeReals) # [MW] Deficit in reserve down
    model.LoadMod_U = Var(model.h, within=NonNegativeReals) # [MW] Positive difference from the baseline
    model.LoadMod_D = Var(model.h, within=NonNegativeReals) # [MW] Negative difference from the baseline
    model.Power = Var(model.cl, model.h, within=NonNegativeReals) # [MW] Power output
    model.Reserve_2U = Var(model.cl, model.h, within=NonNegativeReals) # [MW] Spinning reserve up
    model.Reserve_2D = Var(model.cl, model.h, within=NonNegativeReals) # [MW] Spinning reserve down
    model.StartUp = Var(model.cl, model.h, within=NonNegativeReals, bounds=_nunit_rule)
    model.ShutDown = Var(model.cl, model.h, within=NonNegativeReals, bounds=_nunit_rule)
    model.StorageLevel = Var(model.h, within=NonNegativeReals) # [MWh] Storage level of charge
    model.StorageInput = Var(model.h, within=NonNegativeReals) # [MWh] Charging input for storage units
    model.StorageOutput = Var(model.h, within=NonNegativeReals) # [MWh] Discharging output of storage units
    model.SystemCost = Var(model.h, within=NonNegativeReals) # [EUR] Hourly system cost

    if not calibration:
        model.CurtailedPower = Var(model.h, within=NonNegativeReals) # [MW] Curtailed power
        model.Power_RES = Var(model.h, within=NonNegativeReals) # [MW] Power consumed from RES

    
    ###############################################################################################
    ############################################### EQUATIONS #####################################
    ###############################################################################################
    
    def EQ_Commitment(model, cl, h):
        if h == 0:
            return (
                model.Committed[cl, h] - model.CommittedInitial[cl] == 
                        model.StartUp[cl, h] - model.ShutDown[cl, h]
            )
        else:
            return (
                model.Committed[cl, h] - model.Committed[cl, h-1] == 
                         model.StartUp[cl, h] - model.ShutDown[cl, h]
            )
    
    
    def EQ_MinUpTime(model, cl, h):
        for h in model.h:
            if (h > model.TimeUpMinimum[cl]):
                return (
                    model.Committed[cl, h] >= 
                        sum(model.StartUp[cl, h-i] for i in range(model.TimeUpMinimum[cl]))
                )
            else:
                return Constraint.Skip
    

    def EQ_MinDownTime(model, cl, h):
        for h in model.h:
            if (h > model.TimeDownMinimum[cl]):
                return (
                    model.Nunits[cl] - model.Committed[cl, h] >= 
                        sum(model.ShutDown[cl, h-i] for i in range(model.TimeDownMinimum[cl]))
                )
            else:
                return Constraint.Skip
    

    # Power output with respect to power output in the previous period:
    def EQ_RampUp_TC(model, cl, h):
        if h == 0:
            return (
                model.Power[cl, h] - model.PowerInitial[cl] <= 
                    (model.Committed[cl, h] - model.StartUp[cl, h]) * model.RampUpMaximum[cl]
                    + model.StartUp[cl, h] * model.RampStartUpMaximum[cl]
                    - model.PowerMinStable[cl] * model.AF[cl, h] * model.ShutDown[cl, h]
                    + model.LL_RampUp[cl, h]
            )
        else:
            return (
                model.Power[cl, h] - model.Power[cl, h-1] <=  
                    (model.Committed[cl, h] - model.StartUp[cl, h]) * model.RampUpMaximum[cl]
                    + model.StartUp[cl, h] * model.RampStartUpMaximum[cl]
                    - model.PowerMinStable[cl] * model.AF[cl, h] * model.ShutDown[cl, h]
                    + model.LL_RampUp[cl, h]
            )
    

    def EQ_RampDown_TC(model, cl, h):
        if h == 0:
            return (
                model.PowerInitial[cl] - model.Power[cl, h] <= 
                        (model.Committed[cl, h] - model.StartUp[cl, h]) * model.RampDownMaximum[cl] 
                        + model.ShutDown[cl, h] * model.RampShutDownMaximum[cl]
                        - model.PowerMinStable[cl] * model.AF[cl, h] * model.StartUp[cl, h]
                        + model.LL_RampDown[cl, h]
            )
        else:
            return (
                model.Power[cl, h-1] - model.Power[cl, h] <= 
                         (model.Committed[cl, h] - model.StartUp[cl, h]) * model.RampDownMaximum[cl] 
                        + model.ShutDown[cl, h] * model.RampShutDownMaximum[cl]
                        - model.PowerMinStable[cl] * model.AF[cl, h] * model.StartUp[cl, h]
                        + model.LL_RampDown[cl, h]
            )
    

    def EQ_CostRampUp(model, cl, h):
        if model.CostRampUp[cl] != 0:
            if h == 0:
                return (
                    model.CostRampUpH[cl, h] >= 
                            model.CostRampUp[cl] * (model.Power[cl, h] - model.PowerInitial[cl])
                )
            else:
                return (
                    model.CostRampUpH[cl, h] >= 
                            model.CostRampUp[cl] * (model.Power[cl, h] - model.Power[cl, h-1])
                )
        else:
            return Constraint.Skip


    def EQ_CostRampDown(model, cl, h):
        if model.CostRampDown[cl] != 0:
            if h == 0:
                return (
                    model.CostRampDownH[cl, h] >= 
                            model.CostRampDown[cl] * (model.PowerInitial[cl] - model.Power[cl, h])
                )
            else:
                return (
                    model.CostRampDownH[cl, h] >= 
                            model.CostRampDown[cl] * (model.Power[cl, h-1] - model.Power[cl, h])
                )
        else:
            return Constraint.Skip


    def EQ_Reserve_2U_capability(model, cl, h):
        return (
            model.Reserve_2U[cl, h] <= 
                model.PowerCapacity[cl] * model.AF[cl, h] * model.Committed[cl, h]
                - model.Power[cl, h]
        )


    def EQ_Reserve_2D_capability(model, cl, h):
        return (
            model.Reserve_2D[cl, h] <= 
                    model.Power[cl, h] 
                    - model.PowerMinStable[cl] * model.AF[cl, h] * model.Committed[cl, h]
                    + model.StorageChargingCapacity - model.StorageInput[h]
        )


    def EQ_Power_must_run(model, cl, h):
        return (
            model.PowerMinStable[cl] * model.AF[cl, h] * model.Committed[cl, h] <= 
                    model.Power[cl, h]
        )


    def EQ_Power_available(model, cl, h):
        return (
            model.Power[cl, h] <= model.PowerCapacity[cl] * model.AF[cl, h] * model.Committed[cl, h]
        )


    def EQ_Force_Decommitment(model, cl, h):
        if model.AF[cl, h] == 0:
            return model.Committed[cl, h] == 0
        else:
            return Constraint.Skip


    def EQ_Demand_balance_DA(model, h):
        if calibration:
            return (
                sum(model.Power[cl, h] for cl in model.cl) 
                + model.RES_Generation[h]
                + model.StorageOutput[h] 
                + model.NetImports[h] == 
                    model.Demand['DA', h] + model.StorageInput[h]
                    + model.LoadMod_U[h] - model.LoadMod_D[h]
                    - model.LL_MaxPower[h] + model.LL_MinPower[h]
            ) 
        else:
            return (
                sum(model.Power[cl, h] for cl in model.cl) 
                + model.Power_RES[h] 
                + model.StorageOutput[h] 
                + model.NetImports[h] == 
                        model.Demand['DA', h] + model.StorageInput[h]
                        + model.LoadMod_U[h] - model.LoadMod_D[h]
                        - model.LL_MaxPower[h] + model.LL_MinPower[h]
            )


    def EQ_Demand_balance_2U(model, h):
        return (
            sum(model.Reserve_2U[cl, h] * model.Technology[cl, t] * model.Reserve[t] 
                    for cl in model.cl for t in model.t) >=
                    model.Demand['2U', h] - model.LL_2U[h]
        )


    def EQ_Demand_balance_2D(model, h):
        return (
            sum(model.Reserve_2D[cl, h] * model.Technology[cl, t] * model.Reserve[t] 
                    for cl in model.cl for t in model.t) >=
                    model.Demand['2D', h] - model.LL_2D[h]
        )
    
    def EQ_Storage_minimum(model, h):
        return model.StorageLevel[h] >= model.StorageMinimum


    def EQ_Storage_level(model, h):
        return model.StorageLevel[h] <= model.StorageCapacity


    def EQ_Storage_input(model, h):
        return model.StorageInput[h] <= model.StorageChargingCapacity


    def EQ_Storage_MaxDischarge(model, h):
        return (model.StorageOutput[h] <= 
                    model.StorageDischargeEfficiency * model.StorageLevel[h]
        )


    def EQ_Storage_MaxCharge(model, h):
        return (model.StorageInput[h] * model.StorageChargingEfficiency <= 
                    model.StorageCapacity - model.StorageLevel[h]
        )


    def EQ_Storage_balance(model, h):
        if h == 0:
            return (model.StorageInitial + model.StorageInput[h] * model.StorageChargingEfficiency ==
                    model.StorageLevel[h] + model.StorageOutput[h] / model.StorageDischargeEfficiency
            )
        else:
            return (model.StorageLevel[h-1] + model.StorageInput[h] * model.StorageChargingEfficiency ==
                    model.StorageLevel[h] + model.StorageOutput[h] / model.StorageDischargeEfficiency
            )


    def EQ_Storage_boundaries(model, h):
        if h == len(model.h):
            return model.StorageLevel[h] >= model.StorageFinalMin
        else:
            return Constraint.Skip


    def EQ_CurtailedPower(model, h):
        return (
            model.AF_Solar[h] * model.SolarCapacity 
            + model.AF_Wind[h] * model.WindCapacity 
            - model.Power_RES[h] <=
                 model.CurtailedPower[h]
        )

    # OBJECTIVE
    def EQ_Objective_function(model):
        return sum(model.SystemCost[h] for h in model.h)


    def EQ_SystemCost(model, h):
        if calibration:
            return model.SystemCost[h] == (   
                sum(model.NoLoadCost * model.Committed[cl, h] for cl in model.cl) 
                + sum(model.CostVariable[cl, h] * model.Power[cl, h] for cl in model.cl)
                + sum(model.CostRampUpH[cl, h] + model.CostRampDownH[cl, h] for cl in model.cl)
                + model.ValueOfLostLoad * (model.LL_MaxPower[h] + model.LL_MinPower[h])
                + 0.8 * model.ValueOfLostLoad * (model.LL_2U[h] + model.LL_2D[h])
                + 0.7 * model.ValueOfLostLoad * sum(model.LL_RampUp[cl, h] + model.LL_RampDown[cl, h] 
                                                        for cl in model.cl)
            )
        else:
            return model.SystemCost[h] == (    
                sum(model.NoLoadCost * model.Committed[cl, h] for cl in model.cl)
                + sum(model.CostVariable[cl, h] * model.Power[cl, h] for cl in model.cl)
                + sum(model.CostRampUpH[cl, h] + model.CostRampDownH[cl, h] for cl in model.cl)
                + model.CostCurtailment * model.CurtailedPower[h]
                + model.ValueOfLostLoad * (model.LL_MaxPower[h] + model.LL_MinPower[h])
                + 0.8 * model.ValueOfLostLoad * (model.LL_2U[h] + model.LL_2D[h])
                + 0.7 * model.ValueOfLostLoad * sum(model.LL_RampUp[cl, h] + model.LL_RampDown[cl, h] 
                                                        for cl in model.cl)
            )

    
    ##########################################################################################
    ######################################## Definition of model #############################
    ##########################################################################################

    model.EQ_Objective_Function       = Objective(rule=EQ_Objective_function, sense=minimize)

    model.EQ_Commitment               = Constraint(model.cl, model.h, rule=EQ_Commitment)
    model.EQ_CostRampUp               = Constraint(model.cl, model.h, rule=EQ_CostRampUp)
    model.EQ_CostRampDown             = Constraint(model.cl, model.h, rule=EQ_CostRampDown)
    model.EQ_Demand_balance_DA        = Constraint(model.h, rule=EQ_Demand_balance_DA)
    model.EQ_MinUpTime                = Constraint(model.cl, model.h, rule=EQ_MinUpTime)
    model.EQ_MinDownTime              = Constraint(model.cl, model.h, rule=EQ_MinDownTime)
    model.EQ_RampUp_TC                = Constraint(model.cl, model.h, rule=EQ_RampUp_TC)
    model.EQ_RampDown_TC              = Constraint(model.cl, model.h, rule=EQ_RampDown_TC)
    model.EQ_Reserve_2U_capability    = Constraint(model.cl, model.h, rule=EQ_Reserve_2U_capability)
    model.EQ_Reserve_2D_capability    = Constraint(model.cl, model.h, rule=EQ_Reserve_2D_capability)
    model.EQ_Power_must_run           = Constraint(model.cl, model.h, rule=EQ_Power_must_run)
    model.EQ_Power_available          = Constraint(model.cl, model.h, rule=EQ_Power_available)
    model.EQ_Force_Decommitment       = Constraint(model.cl, model.h, rule=EQ_Force_Decommitment)
    model.EQ_Demand_balance_2U        = Constraint(model.h, rule=EQ_Demand_balance_2U)
    model.EQ_Demand_balance_2D        = Constraint(model.h, rule=EQ_Demand_balance_2D)
    model.EQ_Storage_minimum          = Constraint(model.h, rule=EQ_Storage_minimum)
    model.EQ_Storage_level            = Constraint(model.h, rule=EQ_Storage_level)
    model.EQ_Storage_input            = Constraint(model.h, rule=EQ_Storage_input)
    model.EQ_Storage_MaxDischarge     = Constraint(model.h, rule=EQ_Storage_MaxDischarge)
    model.EQ_Storage_MaxCharge        = Constraint(model.h, rule=EQ_Storage_MaxCharge)
    model.EQ_Storage_balance          = Constraint(model.h, rule=EQ_Storage_balance)
    model.EQ_Storage_boundaries       = Constraint(model.h, rule=EQ_Storage_boundaries)
    model.EQ_SystemCost               = Constraint(model.h, rule=EQ_SystemCost)
    
    if not calibration:
        model.EQ_CurtailedPower       = Constraint(model.h, rule=EQ_CurtailedPower)
    
    if load_shape_mode is None:
        model.EQ_MaxLoadShaping = Constraint(rule=lambda model: 
                sum(model.LoadMod_U[h] + model.LoadMod_D[h] for h in model.h) == 0)
    elif load_shape_mode.lower() == 'savings':
        model.EQ_MaxLoadShaping_up = Constraint(rule=lambda model: 
                sum(model.LoadMod_U[h] for h in model.h) == 0)
        model.EQ_MaxLoadShaping_down = Constraint(rule=lambda model: 
                sum(model.LoadMod_D[h] for h in model.h) <= model.MaxLoadShaping)
    elif load_shape_mode.lower() == 'flex':
        model.EQ_MaxLoadShaping = Constraint(rule=lambda model: 
                sum(model.LoadMod_U[h] + model.LoadMod_D[h] for h in model.h) <= model.MaxLoadShaping)
         
        
    return model


def run_solver(instance, solver="glpk"):
    solver = SolverFactory(solver)
    solver.options['OutputFlag'] = False
    results = solver.solve(instance, tee=True)
    
    if results.solver.termination_condition != TerminationCondition.optimal:
        # something went wrong
        logging.warn("Solver: %s" % results.solver.termination_condition)
        logging.debug(results.solver)
    else:
        logging.info("Solver: %s" % results.solver.termination_condition)
    
    instance.solutions.load_from(results)
    return instance