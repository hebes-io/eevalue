# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Literal, Optional, Union

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Param,
    Reals,
    Set,
    SolverFactory,
    Var,
    minimize,
)
from pyomo.opt import TerminationCondition


def create_model(
    sets: Dict[str, List[Union[str, int]]],
    params: Dict,
    load_shape_mode: Optional[Literal["up", "down", "flex"]] = None,
    action: Literal["calibrate", "backtest", "forward"] = "backtest",
) -> ConcreteModel:
    """Create a concrete Pyomo model for optimization.

    Adapted from https://github.com/energy-modelling-toolkit/Dispa-SET/tree/v2.3

    Args:
        sets (dict): Dictionary containing the model's sets (each set is defined as a list
            of strings or integers).
        params (dict): Dictionary containing the parameters of the optimization problem.
        load_shape_mode ({'up', 'down', 'flex'}, optional): Optional string indicating the
            presence and type of the load modifying resources. It can be 'up', 'down' or 'flex'.
            Defaults to None.
        action ({"calibrate", "backtest", "forward"}, optional): Indicates whether the model
            will be used for calibration, backtesting (simulation using historical data to
            compare actual and predicted results) or forward scenario modelling. Defaults to
            "backtest".

    Returns:
        pyomo.environ.ConcreteModel: The Pyomo optimization model instance.
    """

    # Definition of Unit Commitment Model:
    model = ConcreteModel("UCM")

    ##############################################################################################
    # Definition of the sets
    ##############################################################################################

    model.h = Set(initialize=sets["h"])  # Hours
    model.cl = Set(initialize=sets["cl"])  # Generation unit clusters
    model.t = Set(initialize=sets["t"])  # Generation technologies
    model.f = Set(initialize=sets["f"])  # Fuel types
    model.mk = Set(
        initialize=sets["mk"]
    )  # Markets (DA: Day-Ahead, 2U: Reserve up, 2D: Reserve Down)

    ###############################################################################################
    # Definition of the Parameters
    ###############################################################################################

    model.AF = Param(
        model.cl, model.h, initialize=params["AF"]
    )  # [%] Availability factor per generation cluster (percentage of nominal capacity available)
    model.CommittedInitial = Param(
        model.cl, initialize=params["CommittedInitial"]
    )  # [-] Number of units initially committed per cluster
    model.CostRampUp = Param(
        model.cl, initialize=params["CostRampUp"]
    )  # [EUR/MW] Ramp-up costs
    model.CostRampDown = Param(
        model.cl, initialize=params["CostRampDown"]
    )  # [EUR/MW] Ramp-down costs
    model.Demand = Param(model.mk, model.h, initialize=params["Demand"])  # [MW] Demand
    model.Efficiency = Param(
        model.cl, initialize=params["Efficiency"]
    )  # [%] Efficiency
    model.EmissionRate = Param(
        model.cl, initialize=params["EmissionRate"]
    )  # [tCO2/MWh] CO2 emission rate
    model.Fuel = Param(
        model.cl, model.f, initialize=params["Fuel"]
    )  # [n.a.] Fuel type per cluster {1 0}
    model.FuelPrice = Param(
        model.f, model.h, initialize=params["FuelPrice"]
    )  # [€/MWh] Fuel price
    model.Markup = Param(
        model.cl, model.h, initialize=params["Markup"]
    )  # [€/MWh] Markup
    model.Nunits = Param(
        model.cl, initialize=params["Nunits"]
    )  # Number of units inside the cluster
    model.PermitPrice = Param(
        model.h, initialize=params["PermitPrice"]
    )  # [EUR/ton] CO2 emission permit price
    model.PowerCapacity = Param(
        model.cl, initialize=params["PowerCapacity"]
    )  # [MW] Installed capacity per unit
    model.PowerInitial = Param(
        model.cl, initialize=params["PowerInitial"]
    )  # [MW] Power output before initial period
    model.PowerMinStable = Param(
        model.cl, initialize=params["PowerMinStable"]
    )  # [MW] Minimum power for stable generation
    model.RampDownMaximum = Param(
        model.cl, initialize=params["RampDownMaximum"]
    )  # [MW/h] Ramp down limit
    model.RampUpMaximum = Param(
        model.cl, initialize=params["RampUpMaximum"]
    )  # [MW/h] Ramp up limit
    model.RampStartUpMaximum = Param(
        model.cl, initialize=params["RampStartUpMaximum"]
    )  # [MW/h] Start-up ramp limit
    model.RampShutDownMaximum = Param(
        model.cl, initialize=params["RampShutDownMaximum"]
    )  # [MW/h] Shut-down ramp limit
    model.Reserve = Param(
        model.t, initialize=params["Reserve"]
    )  # [n.a.] Reserve technology {1 0}
    model.StorageCapacity = Param(
        initialize=params["StorageCapacity"]
    )  # [MWh] Storage capacity
    model.StorageChargingCapacity = Param(
        initialize=params["StorageChargingCapacity"]
    )  # [MW] Maximum charging capacity
    model.StorageFinalMin = Param(
        initialize=params["StorageFinalMin"]
    )  # [MWh] Minimum storage level at the end of the optimization period
    model.StorageInitial = Param(
        initialize=params["StorageInitial"]
    )  # [MWh] Storage level before initial period
    model.StorageMinimum = Param(
        initialize=params["StorageMinimum"]
    )  # [MWh] Minimum storage level
    model.StorageChargingEfficiency = Param(
        initialize=params["StorageChargingEfficiency"]
    )  # [%] Charging efficiency
    model.StorageDischargeEfficiency = Param(
        initialize=params["StorageDischargeEfficiency"]
    )  # [%] Discharging efficiency
    model.Technology = Param(
        model.cl, model.t, initialize=params["Technology"]
    )  # [n.a] Technology type {1 0}
    model.TimeDownMinimum = Param(
        model.cl, initialize=params["TimeDownMinimum"]
    )  # [h] Minimum down time
    model.TimeUpMinimum = Param(
        model.cl, initialize=params["TimeUpMinimum"]
    )  # [h] Minimum up time
    model.VOLL = Param(initialize=params["VOLL"])  # [€/Mwh] Value of lost load

    def _var_cost_init(model, cl, h):
        return (
            model.Markup[cl, h]
            + sum(
                model.Fuel[cl, f] * model.FuelPrice[f, h] / model.Efficiency[cl]
                for f in model.f
            )
            + model.PermitPrice[h] * model.EmissionRate[cl]
        )

    model.CostVariable = Param(
        model.cl, model.h, initialize=_var_cost_init
    )  # [EUR/MWh] Variable costs

    if (action == "calibrate") or (action == "backtest"):
        model.NetImports = Param(
            model.h, initialize=params["NetImports"]
        )  # [MW] Net power imports
        model.PowerRES = Param(
            model.h, initialize=params["PowerRES"]
        )  # [MW] RES generation

    if action == "forward":
        model.AFSolar = Param(
            model.h, initialize=params["AFSolar"]
        )  # [%] Availability factor (percentage of nominal capacity available) for solar
        model.AFWind = Param(
            model.h, initialize=params["AFWind"]
        )  # [%] Availability factor (percentage of nominal capacity available) for wind
        model.CostCurtailment = Param(
            initialize=params["CostCurtailment"]
        )  # Cost of RES curtailment [EUR/MWh]
        model.ExportsMax = Param(
            model.h, initialize=params["ExportsMax"]
        )  # [MW] Upper bound for power exports
        model.LoadShapeMaximum = Param(
            initialize=params["LoadShapeMaximum"]
        )  # [MW] Total load shaping allowed
        model.ImportsMax = Param(
            model.h, initialize=params["ImportsMax"]
        )  # [MW] Upper bound for power imports
        model.SolarCapacity = Param(
            model.h, initialize=params["SolarCapacity"]
        )  # [MW] Installed capacity for solar
        model.WindCapacity = Param(
            model.h, initialize=params["WindCapacity"]
        )  # [MW] Installed capacity for wind

    ################################################################################################
    #  Definition of variables
    ################################################################################################

    def _nunit_rule(model, cl, h):
        return (0, model.Nunits[cl])

    model.Committed = Var(
        model.cl, model.h, within=NonNegativeReals, bounds=_nunit_rule
    )  # Number of units that are committed per cluster at hour h
    model.CostRampUpH = Var(
        model.cl, model.h, within=NonNegativeReals
    )  # [EUR/h] Cost of ramping up
    model.CostRampDownH = Var(
        model.cl, model.h, within=NonNegativeReals
    )  # [EUR/h] Cost of ramping down
    model.LLMaxPower = Var(
        model.h, within=NonNegativeReals
    )  # [MW] Deficit in terms of maximum power
    model.LLMinPower = Var(
        model.h, within=NonNegativeReals
    )  # [MW] Power exceeding the demand
    model.LLRampDown = Var(
        model.cl, model.h, within=NonNegativeReals
    )  # [MW] Deficit in terms of ramping down for each cluster
    model.LLRampUp = Var(
        model.cl, model.h, within=NonNegativeReals
    )  # [MW] Deficit in terms of ramping up for each cluster
    model.LL2U = Var(model.h, within=NonNegativeReals)  # [MW] Deficit in reserve up
    model.LL2D = Var(model.h, within=NonNegativeReals)  # [MW] Deficit in reserve down
    model.Power = Var(model.cl, model.h, within=NonNegativeReals)  # [MW] Power output
    model.Reserve2U = Var(
        model.cl, model.h, within=NonNegativeReals
    )  # [MW] Spinning reserve up
    model.Reserve2D = Var(
        model.cl, model.h, within=NonNegativeReals
    )  # [MW] Spinning reserve down
    model.StartUp = Var(
        model.cl, model.h, within=NonNegativeReals, bounds=_nunit_rule
    )  # Number of units in cluster cl that start up at hour h
    model.ShutDown = Var(
        model.cl, model.h, within=NonNegativeReals, bounds=_nunit_rule
    )  # Number of units in cluster cl that shut down at hour h
    model.StorageInput = Var(
        model.h, within=NonNegativeReals
    )  # [MWh] Charging input for storage
    model.StorageLevel = Var(
        model.h, within=NonNegativeReals
    )  # [MWh] Storage level of charge
    model.StorageOutput = Var(
        model.h, within=NonNegativeReals
    )  # [MWh] Discharging output of storage
    model.SystemCost = Var(model.h, within=NonNegativeReals)  # [EUR] Hourly system cost

    if action == "forward":
        model.CurtailedPower = Var(
            model.h, within=NonNegativeReals
        )  # [MW] Curtailed power
        model.Exports = Var(model.h, within=NonNegativeReals)  # [MW] Power exports
        model.Imports = Var(model.h, within=NonNegativeReals)  # [MW] Power imports
        model.LoadModeUp = Var(
            model.h, within=NonNegativeReals
        )  # [MW] Positive difference from the baseline
        model.LoadModeDown = Var(
            model.h, within=NonNegativeReals
        )  # [MW] Negative difference from the baseline
        model.NetImports = Var(model.h, within=Reals)  # [MW] Net power imports
        model.PowerRES = Var(
            model.h, within=NonNegativeReals
        )  # [MW] Power consumed from RES

    ###############################################################################################
    # EQUATIONS
    ###############################################################################################

    def EQ_Commitment(model, cl, h):
        if h == 0:
            return (
                model.Committed[cl, h] - model.CommittedInitial[cl]
                == model.StartUp[cl, h] - model.ShutDown[cl, h]
            )
        else:
            return (
                model.Committed[cl, h] - model.Committed[cl, h - 1]
                == model.StartUp[cl, h] - model.ShutDown[cl, h]
            )

    def EQ_MinUpTime(model, cl, h):
        for h in model.h:
            if h > model.TimeUpMinimum[cl]:
                return model.Committed[cl, h] >= sum(
                    model.StartUp[cl, h - i] for i in range(model.TimeUpMinimum[cl])
                )
            else:
                return Constraint.Skip

    def EQ_MinDownTime(model, cl, h):
        for h in model.h:
            if h > model.TimeDownMinimum[cl]:
                return model.Nunits[cl] - model.Committed[cl, h] >= sum(
                    model.ShutDown[cl, h - i] for i in range(model.TimeDownMinimum[cl])
                )
            else:
                return Constraint.Skip

    # Power output with respect to power output in the previous period:
    def EQ_RampUp_TC(model, cl, h):
        if h == 0:
            return (
                model.Power[cl, h] - model.PowerInitial[cl]
                <= (model.Committed[cl, h] - model.StartUp[cl, h])
                * model.RampUpMaximum[cl]
                + model.StartUp[cl, h] * model.RampStartUpMaximum[cl]
                - model.PowerMinStable[cl] * model.AF[cl, h] * model.ShutDown[cl, h]
                + model.LLRampUp[cl, h]
            )
        else:
            return (
                model.Power[cl, h] - model.Power[cl, h - 1]
                <= (model.Committed[cl, h] - model.StartUp[cl, h])
                * model.RampUpMaximum[cl]
                + model.StartUp[cl, h] * model.RampStartUpMaximum[cl]
                - model.PowerMinStable[cl] * model.AF[cl, h] * model.ShutDown[cl, h]
                + model.LLRampUp[cl, h]
            )

    def EQ_RampDown_TC(model, cl, h):
        if h == 0:
            return (
                model.PowerInitial[cl] - model.Power[cl, h]
                <= (model.Committed[cl, h] - model.StartUp[cl, h])
                * model.RampDownMaximum[cl]
                + model.ShutDown[cl, h] * model.RampShutDownMaximum[cl]
                - model.PowerMinStable[cl] * model.AF[cl, h] * model.StartUp[cl, h]
                + model.LLRampDown[cl, h]
            )
        else:
            return (
                model.Power[cl, h - 1] - model.Power[cl, h]
                <= (model.Committed[cl, h] - model.StartUp[cl, h])
                * model.RampDownMaximum[cl]
                + model.ShutDown[cl, h] * model.RampShutDownMaximum[cl]
                - model.PowerMinStable[cl] * model.AF[cl, h] * model.StartUp[cl, h]
                + model.LLRampDown[cl, h]
            )

    def EQ_CostRampUp(model, cl, h):
        if model.CostRampUp[cl] != 0:
            if h == 0:
                return model.CostRampUpH[cl, h] >= model.CostRampUp[cl] * (
                    model.Power[cl, h] - model.PowerInitial[cl]
                )
            else:
                return model.CostRampUpH[cl, h] >= model.CostRampUp[cl] * (
                    model.Power[cl, h] - model.Power[cl, h - 1]
                )
        else:
            return Constraint.Skip

    def EQ_CostRampDown(model, cl, h):
        if model.CostRampDown[cl] != 0:
            if h == 0:
                return model.CostRampDownH[cl, h] >= model.CostRampDown[cl] * (
                    model.PowerInitial[cl] - model.Power[cl, h]
                )
            else:
                return model.CostRampDownH[cl, h] >= model.CostRampDown[cl] * (
                    model.Power[cl, h - 1] - model.Power[cl, h]
                )
        else:
            return Constraint.Skip

    def EQ_Reserve_2U_capability(model, cl, h):
        return (
            model.Reserve2U[cl, h]
            <= model.PowerCapacity[cl] * model.AF[cl, h] * model.Committed[cl, h]
            - model.Power[cl, h]
        )

    def EQ_Reserve_2D_capability(model, cl, h):
        return (
            model.Reserve2D[cl, h]
            <= model.Power[cl, h]
            - model.PowerMinStable[cl] * model.AF[cl, h] * model.Committed[cl, h]
            + model.StorageChargingCapacity
            - model.StorageInput[h]
        )

    def EQ_Power_must_run(model, cl, h):
        return (
            model.PowerMinStable[cl] * model.AF[cl, h] * model.Committed[cl, h]
            <= model.Power[cl, h]
        )

    def EQ_Power_available(model, cl, h):
        return (
            model.Power[cl, h]
            <= model.PowerCapacity[cl] * model.AF[cl, h] * model.Committed[cl, h]
        )

    def EQ_Force_Decommitment(model, cl, h):
        if model.AF[cl, h] == 0:
            return model.Committed[cl, h] == 0
        else:
            return Constraint.Skip

    def EQ_Demand_balance_DA(model, h):
        if action == "forward":
            return (
                sum(model.Power[cl, h] for cl in model.cl)
                + model.PowerRES[h]
                + model.StorageOutput[h]
                - model.NetImports[h]
                == model.Demand["DA", h]
                + model.StorageInput[h]
                + model.LoadModeUp[h]
                - model.LoadModeDown[h]
                - model.LLMaxPower[h]
                + model.LLMinPower[h]
            )
        else:
            return (
                sum(model.Power[cl, h] for cl in model.cl)
                + model.PowerRES[h]
                + model.StorageOutput[h]
                - model.NetImports[h]
                == model.Demand["DA", h]
                + model.StorageInput[h]
                - model.LLMaxPower[h]
                + model.LLMinPower[h]
            )

    def EQ_Demand_balance_2U(model, h):
        return (
            sum(
                model.Reserve2U[cl, h] * model.Technology[cl, t] * model.Reserve[t]
                for cl in model.cl
                for t in model.t
            )
            >= model.Demand["2U", h] - model.LL2U[h]
        )

    def EQ_Demand_balance_2D(model, h):
        return (
            sum(
                model.Reserve2D[cl, h] * model.Technology[cl, t] * model.Reserve[t]
                for cl in model.cl
                for t in model.t
            )
            >= model.Demand["2D", h] - model.LL2D[h]
        )

    def EQ_Storage_minimum(model, h):
        return model.StorageLevel[h] >= model.StorageMinimum

    def EQ_Storage_level(model, h):
        return model.StorageLevel[h] <= model.StorageCapacity

    def EQ_Storage_input(model, h):
        return model.StorageInput[h] <= model.StorageChargingCapacity

    def EQ_Storage_MaxDischarge(model, h):
        return (
            model.StorageOutput[h]
            <= model.StorageDischargeEfficiency * model.StorageLevel[h]
        )

    def EQ_Storage_MaxCharge(model, h):
        return (
            model.StorageInput[h] * model.StorageChargingEfficiency
            <= model.StorageCapacity - model.StorageLevel[h]
        )

    def EQ_Storage_balance(model, h):
        if h == 0:
            return (
                model.StorageInitial
                + model.StorageInput[h] * model.StorageChargingEfficiency
                == model.StorageLevel[h]
                + model.StorageOutput[h] / model.StorageDischargeEfficiency
            )
        else:
            return (
                model.StorageLevel[h - 1]
                + model.StorageInput[h] * model.StorageChargingEfficiency
                == model.StorageLevel[h]
                + model.StorageOutput[h] / model.StorageDischargeEfficiency
            )

    def EQ_Storage_boundaries(model, h):
        if h == len(model.h):
            return model.StorageLevel[h] >= model.StorageFinalMin
        else:
            return Constraint.Skip

    def EQ_CurtailedPower(model, h):
        return (
            model.AFSolar[h] * model.SolarCapacity[h]
            + model.AFWind[h] * model.WindCapacity[h]
            - model.PowerRES[h]
            == model.CurtailedPower[h]
        )

    def EQ_MaxImports(model, h):
        return model.Imports[h] <= model.ImportsMax[h]

    def EQ_MaxExports(model, h):
        return model.Exports[h] <= model.ExportsMax[h]

    def EQ_NetImports(model, h):
        return model.NetImports[h] == model.Exports[h] - model.Imports[h]

    def EQ_LoadShapeMaximum_Down_Up(model, h):
        return model.LoadModeUp[h] == 0

    def EQ_LoadShapeMaximum_Down_Down(model):
        return sum(model.LoadModeDown[h] for h in model.h) <= model.LoadShapeMaximum

    def EQ_LoadShapeMaximum_Up_Up(model):
        return sum(model.LoadModeUp[h] for h in model.h) <= model.LoadShapeMaximum

    def EQ_LoadShapeMaximum_Up_Down(model, h):
        return model.LoadModeDown[h] == 0

    def EQ_LoadShapeMaximum_Flex(model):
        return (
            sum(model.LoadModeUp[h] + model.LoadModeDown[h] for h in model.h)
            <= model.LoadShapeMaximum
        )

    # OBJECTIVE
    def EQ_Objective_function(model):
        return sum(model.SystemCost[h] for h in model.h)

    def EQ_SystemCost(model, h):
        if (action == "calibrate") or (action == "backtest"):
            return model.SystemCost[h] == (
                sum(model.CostVariable[cl, h] * model.Power[cl, h] for cl in model.cl)
                + sum(
                    model.CostRampUpH[cl, h] + model.CostRampDownH[cl, h]
                    for cl in model.cl
                )
                + model.VOLL * (model.LLMaxPower[h] + model.LLMinPower[h])
                + 0.8 * model.VOLL * (model.LL2U[h] + model.LL2D[h])
                + 0.7
                * model.VOLL
                * sum(
                    model.LLRampUp[cl, h] + model.LLRampDown[cl, h] for cl in model.cl
                )
            )
        else:
            return model.SystemCost[h] == (
                sum(model.CostVariable[cl, h] * model.Power[cl, h] for cl in model.cl)
                + sum(
                    model.CostRampUpH[cl, h] + model.CostRampDownH[cl, h]
                    for cl in model.cl
                )
                + model.CostCurtailment * model.CurtailedPower[h]
                + model.VOLL * (model.LLMaxPower[h] + model.LLMinPower[h])
                + 0.8 * model.VOLL * (model.LL2U[h] + model.LL2D[h])
                + 0.7
                * model.VOLL
                * sum(
                    model.LLRampUp[cl, h] + model.LLRampDown[cl, h] for cl in model.cl
                )
            )

    ##########################################################################################
    # Definition of model
    ##########################################################################################

    model.EQ_Objective_Function = Objective(rule=EQ_Objective_function, sense=minimize)

    model.EQ_Commitment = Constraint(model.cl, model.h, rule=EQ_Commitment)
    model.EQ_CostRampUp = Constraint(model.cl, model.h, rule=EQ_CostRampUp)
    model.EQ_CostRampDown = Constraint(model.cl, model.h, rule=EQ_CostRampDown)
    model.EQ_Demand_balance_DA = Constraint(model.h, rule=EQ_Demand_balance_DA)
    model.EQ_MinUpTime = Constraint(model.cl, model.h, rule=EQ_MinUpTime)
    model.EQ_MinDownTime = Constraint(model.cl, model.h, rule=EQ_MinDownTime)
    model.EQ_RampUp_TC = Constraint(model.cl, model.h, rule=EQ_RampUp_TC)
    model.EQ_RampDown_TC = Constraint(model.cl, model.h, rule=EQ_RampDown_TC)
    model.EQ_Reserve_2U_capability = Constraint(
        model.cl, model.h, rule=EQ_Reserve_2U_capability
    )
    model.EQ_Reserve_2D_capability = Constraint(
        model.cl, model.h, rule=EQ_Reserve_2D_capability
    )
    model.EQ_Power_must_run = Constraint(model.cl, model.h, rule=EQ_Power_must_run)
    model.EQ_Power_available = Constraint(model.cl, model.h, rule=EQ_Power_available)
    model.EQ_Force_Decommitment = Constraint(
        model.cl, model.h, rule=EQ_Force_Decommitment
    )
    model.EQ_Demand_balance_2U = Constraint(model.h, rule=EQ_Demand_balance_2U)
    model.EQ_Demand_balance_2D = Constraint(model.h, rule=EQ_Demand_balance_2D)
    model.EQ_Storage_minimum = Constraint(model.h, rule=EQ_Storage_minimum)
    model.EQ_Storage_level = Constraint(model.h, rule=EQ_Storage_level)
    model.EQ_Storage_input = Constraint(model.h, rule=EQ_Storage_input)
    model.EQ_Storage_MaxDischarge = Constraint(model.h, rule=EQ_Storage_MaxDischarge)
    model.EQ_Storage_MaxCharge = Constraint(model.h, rule=EQ_Storage_MaxCharge)
    model.EQ_Storage_balance = Constraint(model.h, rule=EQ_Storage_balance)
    model.EQ_Storage_boundaries = Constraint(model.h, rule=EQ_Storage_boundaries)
    model.EQ_SystemCost = Constraint(model.h, rule=EQ_SystemCost)

    if action == "forward":
        model.EQ_CurtailedPower = Constraint(model.h, rule=EQ_CurtailedPower)
        model.EQ_MaxImports = Constraint(model.h, rule=EQ_MaxImports)
        model.EQ_MaxExports = Constraint(model.h, rule=EQ_MaxExports)
        model.EQ_NetImports = Constraint(model.h, rule=EQ_NetImports)

        if load_shape_mode == "up":
            model.EQ_LoadShapeMaximum_Up_Up = Constraint(rule=EQ_LoadShapeMaximum_Up_Up)
            model.EQ_LoadShapeMaximum_Up_Down = Constraint(
                model.h, rule=EQ_LoadShapeMaximum_Up_Down
            )
        elif load_shape_mode == "down":
            model.EQ_LoadShapeMaximum_Down_Up = Constraint(
                model.h, rule=EQ_LoadShapeMaximum_Down_Up
            )
            model.EQ_LoadShapeMaximum_Down_Down = Constraint(
                rule=EQ_LoadShapeMaximum_Down_Down
            )
        elif load_shape_mode == "flex":
            model.EQ_LoadShapeMaximum_Flex = Constraint(rule=EQ_LoadShapeMaximum_Flex)
        else:
            model.EQ_LoadShapeMaximum_Up_Down = Constraint(
                model.h, rule=EQ_LoadShapeMaximum_Up_Down
            )
            model.EQ_LoadShapeMaximum_Down_Up = Constraint(
                model.h, rule=EQ_LoadShapeMaximum_Down_Up
            )

    return model


def run_solver(model: ConcreteModel, solver: str = "highs"):
    opt = SolverFactory(solver)
    if solver == "highs":
        opt.options["run_crossover"] = False
        opt.options["feastol"] = 1e-04
        opt.options["dualfeastol"] = 1e-05

    results = opt.solve(model, tee=False)

    if results.solver.termination_condition != TerminationCondition.optimal:
        # something went wrong
        logging.warn("Solver: %s" % results.solver.termination_condition)
        logging.debug(results.solver)
    else:
        logging.info("Solver: %s" % results.solver.termination_condition)

    model.solutions.load_from(results)
    return model
