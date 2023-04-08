# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from itertools import product
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import validator
from pydantic.dataclasses import dataclass
from toolz.functoolz import pipe


class ValidationConfig:
    arbitrary_types_allowed = True
    underscore_attrs_are_private = True


@dataclass(config=ValidationConfig)
class SimData:
    simulation_period: Union[np.ndarray, pd.DatetimeIndex]
    # Required data
    avail_factors: pd.DataFrame
    clustered_plants: pd.DataFrame
    demand: pd.DataFrame
    fuel_price: pd.DataFrame
    permit_price: pd.Series
    reserve_technologies: list
    # Attributes that define when optional data is needed
    action: Literal["calibrate", "backtest", "forward"] = "backtest"
    load_shape_mode: Optional[Literal["up", "down", "flex"]] = None
    # Optional data
    avail_factor_pv: Optional[pd.Series] = None
    avail_factor_wind: Optional[pd.Series] = None
    capacity_pv: Optional[pd.Series] = None
    capacity_wind: Optional[pd.Series] = None
    cost_curtailment: Optional[float] = None
    markup: Optional[pd.DataFrame] = None
    max_exports: Optional[pd.Series] = None
    max_imports: Optional[pd.Series] = None
    net_imports: Optional[pd.Series] = None
    maximum_load_shaping: Optional[float] = None
    res_generation: Optional[pd.Series] = None
    voll: Optional[float] = None
    # Initial state
    committed_initial: Optional[dict] = None
    power_initial: Optional[dict] = None
    storage_initial: Optional[float] = None
    # Storage
    storage_charge_ef: Optional[float] = None
    storage_discharge_ef: Optional[float] = None
    storage_capacity: Optional[float] = None
    storage_charge_cap: Optional[float] = None
    storage_min: Optional[float] = None
    storage_final_min: Optional[float] = None

    def __post_init__(self):
        if (self.action == "calibrate") or (self.action == "backtest"):
            if self.res_generation is None:
                raise ValueError(
                    "RES generation data must be provided for "
                    "calibration or backtesting."
                )
            if self.net_imports is None:
                raise ValueError(
                    "Net imports data must be provided for "
                    "calibration or backtesting."
                )

        if self.action == "forward":
            if self.avail_factor_pv is None:
                raise ValueError(
                    "Solar PV availability must be provided for forward simulation"
                )
            if self.avail_factor_wind is None:
                raise ValueError(
                    "Wind availability must be provided for forward simulation"
                )
            if self.capacity_pv is None:
                raise ValueError(
                    "Solar PV capacity must be provided for forward simulation"
                )
            if self.capacity_wind is None:
                raise ValueError(
                    "Wind capacity must be provided for forward simulation"
                )
            if self.max_exports is None:
                raise ValueError(
                    "Upper bound for exports must be provided for forward simulation"
                )
            if self.max_imports is None:
                raise ValueError(
                    "Upper bound for imports must be provided for forward simulation"
                )
            if (self.load_shape_mode is not None) and (
                self.maximum_load_shaping is None
            ):
                raise ValueError(
                    "`maximum_load_shaping` must be provided if `load_shape_mode` is not None."
                )

    @validator("avail_factors", "avail_factor_pv", "avail_factor_wind")
    def check_availability_factor(cls, data):
        if data is not None:
            data = data.clip(0, 1)
        return data

    @validator("load_shape_mode")
    def check_load_shape_mode(cls, data):
        if data is not None:
            if data not in ("up", "down", "flex"):
                raise ValueError(
                    "`load_shape_mode` can be only one of 'up', 'down', 'flex'."
                )
        return data

    @validator("action")
    def check_action(cls, data):
        if data is not None:
            if data not in ("calibrate", "backtest", "forward"):
                raise ValueError(
                    "`action` can be only one of 'calibrate', 'backtest', 'forward'."
                )
        return data

    @validator(
        "avail_factors",
        "avail_factor_pv",
        "avail_factor_wind",
        "capacity_pv",
        "capacity_wind",
        "demand",
        "fuel_price",
        "markup",
        "max_exports",
        "max_imports",
        "net_imports",
        "permit_price",
        "res_generation",
    )
    def check_time_index(cls, data, values):
        if isinstance(values["simulation_period"], pd.DatetimeIndex) and (
            data is not None
        ):
            data.index = data.index.map(pd.to_datetime)
            time_step = data.index.to_series().diff().min()

            if time_step != pd.Timedelta("1H"):
                raise ValueError(
                    "The index of the dataframe has not an hourly time step"
                )
            if not data.index.is_monotonic_increasing:
                raise ValueError(
                    "The index of the dataframe is not monotonic increasing"
                )
            if not set(values["simulation_period"]).issubset(data.index):
                raise ValueError(
                    "The `simulation_period` range is not a subset of the dataframe's index"
                )
        return data

    @validator("clustered_plants")
    def check_expected_fields(cls, data):
        expected_fields = [
            "Cluster",
            "Nunits",
            "PowerCapacity",
            "PowerMinStable",
            "Efficiency",
            "RampStartUpRate",
            "CO2Intensity",
            "MinDownTime",
            "MinUpTime",
            "RampUpRate",
            "RampDownRate",
            "CostRampUp",
            "Technology",
            "Fuel",
        ]

        if not set(expected_fields).issubset(data.columns):
            raise ValueError("Missing fields from plant cluster dataframe")
        return data

    def __post_init_post_parse__(self):
        self._sets = dict()
        self._sets["mk"] = ["DA", "2U", "2D"]
        self._sets["cl"] = self.clustered_plants["Cluster"].tolist()
        self._sets["f"] = self.clustered_plants["Fuel"].unique().tolist()
        self._sets["t"] = self.clustered_plants["Technology"].unique().tolist()
        self._sets["h"] = list(range(len(self.simulation_period)))

    @property
    def sets(self):
        return copy.deepcopy(self._sets)

    def build_cluster_data(self, parameters=None):
        if parameters is None:
            parameters = dict()

        plants = self.clustered_plants.set_index("Cluster")
        parameters["Nunits"] = plants["Nunits"].to_dict()
        parameters["PowerCapacity"] = plants["PowerCapacity"].to_dict()
        parameters["PowerMinStable"] = plants["PowerMinStable"].to_dict()
        parameters["Efficiency"] = plants["Efficiency"].to_dict()
        parameters["EmissionRate"] = plants["CO2Intensity"].to_dict()
        parameters["TimeDownMinimum"] = plants["MinDownTime"].to_dict()
        parameters["TimeUpMinimum"] = plants["MinUpTime"].to_dict()
        parameters["RampUpMaximum"] = plants["RampUpRate"].to_dict()
        parameters["RampDownMaximum"] = plants["RampDownRate"].to_dict()
        parameters["RampStartUpMaximum"] = plants["RampStartUpRate"].to_dict()
        parameters["CostRampUp"] = plants["CostRampUp"].to_dict()

        parameters["RampShutDownMaximum"] = (
            {cl: 0 for cl in self._sets["cl"]}
            if "RampShutDownRate" not in plants.columns
            else plants["RampStartUpRate"].to_dict()
        )
        parameters["CostRampDown"] = (
            {cl: 0 for cl in self._sets["cl"]}
            if "CostRampDown" not in plants.columns
            else plants["CostRampDown"].to_dict()
        )
        parameters["Technology"] = {
            (cl, t): 1 if plants.loc[cl, "Technology"] == t else 0
            for (cl, t) in product(self._sets["cl"], self._sets["t"])
        }
        parameters["Fuel"] = {
            (cl, f): 1 if plants.loc[cl, "Fuel"] == f else 0
            for (cl, f) in product(self._sets["cl"], self._sets["f"])
        }
        return parameters

    def build_market_data(self, parameters=None):
        if parameters is None:
            parameters = dict()

        parameters["Reserve"] = {
            tech: 1 if tech in self.reserve_technologies else 0
            for tech in self._sets["t"]
        }
        parameters["VOLL"] = self.voll or 15000

        if self.action == "forward":
            parameters["LoadShapeMaximum"] = self.maximum_load_shaping or 0
            parameters["CostCurtailment"] = self.cost_curtailment or 0

        return parameters

    def build_time_series(self, parameters=None):
        if parameters is None:
            parameters = dict()

        def align_time_index(data):
            data = data.loc[self.simulation_period].reset_index(drop=True)
            if set(data.index).difference(self._sets["h"]):
                raise RuntimeError("Dataframe index cannot match `h` index")
            return data

        avail_factors = align_time_index(self.avail_factors)
        parameters["AF"] = {}
        for cl in self._sets["cl"]:
            parameters["AF"].update(
                {(cl, h): val for h, val in avail_factors[cl].to_dict().items()}
            )

        demand = align_time_index(self.demand)
        parameters["Demand"] = {}
        parameters["Demand"].update(
            {("DA", h): val for h, val in demand["DA"].to_dict().items()}
        )
        parameters["Demand"].update(
            {("2U", h): val for h, val in demand["2U"].to_dict().items()}
        )
        parameters["Demand"].update(
            {("2D", h): val for h, val in demand["2D"].to_dict().items()}
        )

        parameters["FuelPrice"] = dict()
        fuel_price = align_time_index(self.fuel_price)
        for fuel in self._sets["f"]:
            parameters["FuelPrice"].update(
                {(fuel, h): val for h, val in fuel_price[fuel].to_dict().items()}
            )

        parameters["PermitPrice"] = align_time_index(self.permit_price).to_dict()

        parameters["Markup"] = dict()
        if self.markup is not None:
            markup = align_time_index(self.markup)
            for cl in self._sets["cl"]:
                parameters["Markup"].update(
                    {(cl, h): val for h, val in markup[cl].to_dict().items()}
                )
        else:
            for cl in self._sets["cl"]:
                parameters["Markup"].update({(cl, h): 0 for h in self._sets["h"]})

        if (self.action == "calibrate") or (self.action == "backtest"):
            parameters["NetImports"] = align_time_index(self.net_imports).to_dict()
            parameters["PowerRES"] = align_time_index(self.res_generation).to_dict()

        if self.action == "forward":
            parameters["AFSolar"] = align_time_index(self.avail_factor_pv).to_dict()
            parameters["AFWind"] = align_time_index(self.avail_factor_wind).to_dict()
            parameters["ExportsMax"] = align_time_index(self.max_exports).to_dict()
            parameters["ImportsMax"] = align_time_index(self.max_imports).to_dict()
            parameters["SolarCapacity"] = align_time_index(self.capacity_pv).to_dict()
            parameters["WindCapacity"] = align_time_index(self.capacity_wind).to_dict()

        return parameters

    def build_storage(self, parameters=None):
        if parameters is None:
            parameters = dict()

        parameters["StorageChargingEfficiency"] = self.storage_charge_ef or 1
        parameters["StorageDischargeEfficiency"] = self.storage_discharge_ef or 1
        parameters["StorageCapacity"] = self.storage_capacity or 0
        parameters["StorageChargingCapacity"] = self.storage_charge_cap or 0
        parameters["StorageMinimum"] = self.storage_min or 0
        parameters["StorageFinalMin"] = self.storage_final_min or 0
        return parameters

    def build_initial_state(self, parameters):
        parameters["CommittedInitial"] = (
            self.committed_initial
            if self.committed_initial is not None
            else {cl: val for cl, val in parameters["Nunits"].items()}
        )
        parameters["PowerInitial"] = (
            self.power_initial
            if self.power_initial is not None
            else {
                cl: parameters["PowerCapacity"][cl] * parameters["Nunits"][cl]
                for cl in self._sets["cl"]
            }
        )
        parameters["StorageInitial"] = self.storage_initial or 0
        return parameters

    def build(self):
        parameters = pipe(
            dict(),
            self.build_cluster_data,
            self.build_market_data,
            self.build_time_series,
            self.build_storage,
            self.build_initial_state,
        )
        return self.sets, parameters
