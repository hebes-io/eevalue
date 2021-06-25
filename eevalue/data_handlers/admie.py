# -*- coding: utf-8 -*-

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import glob
import xlrd

import numpy as np 
import pandas as pd

from tqdm.auto import tqdm
from urllib.request import urlopen
from urllib.error import HTTPError
from collections import defaultdict

from eevalue.definitions import ROOT_DIR


############## Data download ##############################################################
###########################################################################################

def fetch_data(period, filetype, destination, suffix):
    """ General data downloader from ADMIE (Greek TSO)
    
    Parameters
    ----------
    period: List of datetime dates or pandas DatetimeIndex with freq='D'
    filetype: String. See https://www.admie.gr/getFiletypeInfoEN for valid values
    destination: String. The path to the folder for storing the downloaded data
    suffix: String. The name to add at each stored file (after the date)

    Example:
    
    period = pd.date_range(start=datetime.date(2019, 12, 14), 
                           end=datetime.date(2020, 12, 31),
                           freq='D')
    
    fetch_admie_data(period, 'DayAheadSchedulingUnitAvailabilities', 
                             'availabilities', 'Availability.xls')

    Returns
    -------
    list of dates for which downloading failed
    """

    path = os.path.join(ROOT_DIR, "RawData/EL/{}".format(destination))
    if not os.path.exists(path):
        os.makedirs(path)
 
    missing_days = []
    pbar = tqdm(total=len(period))
    for day in period:   
        file_path = None 
        sday = day.strftime('%Y-%m-%d') 
        try:
            response = urlopen("https://www.admie.gr/getOperationMarketFile?"
                               f"dateStart={sday}&dateEnd={sday}&FileCategory={filetype}")
        except HTTPError:
            continue
        else:
            response = json.loads(response.read().decode('utf-8'))
            if len(response) > 0:
                file_path = response[0]['file_path']
        
        if file_path is not None:
            try:
                f = urlopen(file_path)
            except HTTPError:
                missing_days.append(day)
                continue
        else:
            missing_days.append(day)
            continue
        
        sday = day.strftime('%Y%m%d')
        with open(os.path.join(path, sday + suffix), "wb") as stream:
            stream.write(f.read())
        pbar.update(1)
        
    pbar.close()
    return missing_days



############## Data aggregation into datasets #############################################
###########################################################################################

def aggregate_load(period):
    path = os.path.join(ROOT_DIR, "RawData/EL/day_ahead_results")
    pbar = tqdm(total=len(period))

    data = {}
    for day in period:   
        sday = day.strftime('%Y%m%d')
        xlfile = glob.glob(os.path.join(path, sday) + '*')

        if len(xlfile) == 0:
            continue 
        else:
            try:
                book = xlrd.open_workbook(xlfile[0], formatting_info=True)
            except xlrd.XLRDError:
                continue

        sheet = book.sheet_by_name(f'{sday}_DAS')
        idx = sheet.col_values(0).index('LOAD DECLARATIONS + LOSSES')
        data[day] = sheet.row_values(idx)[1:25] 
        pbar.update(1)
            
    pbar.close()
    result = pd.DataFrame.from_dict(data, orient='index', columns=range(24))
    result = result.stack()
    result.index = result.index.map(lambda x: x[0].replace(hour=int(x[1])))
    result = result.to_frame('Total load [MW]')
    
    path = os.path.join(ROOT_DIR, "DataSets/Load/EL")
    result.to_csv(os.path.join(path, "load.csv"))


def aggregate_imports(period):
    path = os.path.join(ROOT_DIR, "RawData/EL/day_ahead_results")
    pbar = tqdm(total=len(period))

    data = {}
    for day in period:   
        sday = day.strftime('%Y%m%d')
        xlfile = glob.glob(os.path.join(path, sday) + '*')

        if len(xlfile) == 0:
            continue 
        else:
            try:
                book = xlrd.open_workbook(xlfile[0], formatting_info=True)
            except xlrd.XLRDError:
                continue

        sheet = book.sheet_by_name(f'{sday}_DAS')
        
        names = sheet.col_values(0)
        start = names.index('NET BORDER SCHEDULES') + 1
        end = names.index('BORDER IMPORTS')
        
        res = 0
        for i in range(start, end):
            try:
                res = res + np.array(sheet.row_values(i)[1:25])
            except np.core._exceptions.UFuncTypeError:
                res = res + np.pad(np.array(sheet.row_values(i)[1:24]), (0, 1), 'constant')
            
        data[day] = (-1)*res # ADMIE uses negative sign for imports and positive for exports    
        pbar.update(1)

    pbar.close()
    result = pd.DataFrame.from_dict(data, orient='index', columns=range(24))
    result = result.stack()
    result.index = result.index.map(lambda x: x[0].replace(hour=int(x[1])))
    result = result.to_frame('Net imports [MW]')
    path = os.path.join(ROOT_DIR, "DataSets/NetImports/EL")
    result.to_csv(os.path.join(path, "imports.csv"))


def aggregate_secondary_reserves(period):
    path = os.path.join(ROOT_DIR, "RawData/EL/day_ahead_results")
    pbar = tqdm(total=len(period))

    data_2u = {}
    data_2d = {}

    for day in period:   
        sday = day.strftime('%Y%m%d')
        xlfile = glob.glob(os.path.join(path, sday) + '*')

        if len(xlfile) == 0:
            continue 
        else:
            try:
                book = xlrd.open_workbook(xlfile[0], formatting_info=True)
            except xlrd.XLRDError:
                continue

        sheet = book.sheet_by_name(f'{sday}_SecondaryReserve')
        names = sheet.col_values(0)
        data_2u[day] = sheet.row_values(names.index('Up - Requirement'))[2:26] 
        data_2d[day] = sheet.row_values(names.index('Dn - Requirement'))[2:26]
        pbar.update(1)
        
    pbar.close()
    data_2u = (pd.DataFrame.from_dict(data_2u, orient='index', columns=range(24))
                           .stack().to_frame('2U'))
    data_2u.index = data_2u.index.map(lambda x: x[0].replace(hour=int(x[1])))

    data_2d = (pd.DataFrame.from_dict(data_2d, orient='index', columns=range(24))
                           .stack().to_frame('2D'))
    data_2d.index = data_2d.index.map(lambda x: x[0].replace(hour=int(x[1])))

    result = pd.concat([data_2u, data_2d], axis=1)
    path = os.path.join(ROOT_DIR, "DataSets/Load/EL")
    result.to_csv(os.path.join(path, "reserves.csv"))


def aggregate_committed_capacity(period):    
    plants = pd.read_csv(os.path.join(ROOT_DIR, "DataSets/PowerPlants/EL/plants.csv"))
    units = plants[['Technology', 'Fuel', 'Unit']].groupby(['Technology', 'Fuel'])['Unit']      \
                                                  .apply(list).to_dict()

    for key, unit_names in units.items():
        if 'THESAVROS' in unit_names:
            units[key].remove('THESAVROS')
            units[key].extend(['THESAVROS1', 'THESAVROS2', 'THESAVROS3'])
            break

    path = os.path.join(ROOT_DIR, "RawData/EL/day_ahead_results")
    pbar = tqdm(total=len(period))
    data = defaultdict(dict)

    for day in period:   
        sday = day.strftime('%Y%m%d')
        xlfile = glob.glob(os.path.join(path, sday) + '*')

        if len(xlfile) == 0:
            continue 
        else:
            try:
                book = xlrd.open_workbook(xlfile[0], formatting_info=True)
            except xlrd.XLRDError:
                continue

        sheet = book.sheet_by_name(f'{sday}_DAS')
        all_names = sheet.col_values(0)
        
        for key, unit_names in units.items():
            both = set(unit_names).intersection(all_names)
            idx = [all_names.index(x) for x in both]

            result = 0
            for i in idx:
                try:
                    result += np.array(sheet.row_values(i)[1:25])
                except np.core._exceptions.UFuncTypeError:
                    result += np.pad(np.array(sheet.row_values(i)[1:24]), (0, 1), 'constant')

            data[key][day] = result
        pbar.update(1)
            
    pbar.close()
    path = os.path.join(ROOT_DIR, "DataSets/Generation/EL")

    for item in units:
        tech, fuel = item
        result = pd.DataFrame.from_dict(data[item], 'index')
        result = result.stack()
        result.index = result.index.map(lambda x: x[0].replace(hour=int(x[1])))
        result = result.to_frame('Capacity committed [MW]')
        result.to_csv(os.path.join(path, f"{tech}_{fuel}.csv"))


def aggregate_available_capacity(period):
    plants = pd.read_csv(os.path.join(ROOT_DIR, "DataSets/PowerPlants/EL/plants.csv"))
    combinations = set(zip(plants['Technology'], plants['Fuel']))
    
    total_capacity = dict()
    for item in combinations:
        tech, fuel = item
        subset = plants[(plants['Technology']==tech) & (plants['Fuel']==fuel)]
        total_capacity[item] = subset['PowerCapacity'].sum()
    
    units = plants[['Technology', 'Fuel', 'Unit']].groupby(['Technology', 'Fuel'])['Unit']   \
                                                  .apply(list).to_dict()
    
    path = os.path.join(ROOT_DIR, "RawData/EL/availabilities")
    pbar = tqdm(total=len(period))

    data = defaultdict(dict)
    for day in period:   
        sday = day.strftime('%Y%m%d')
        xlfile = glob.glob(os.path.join(path, sday) + '*')
        
        if len(xlfile) == 0:
            continue 
        else:
            try:
                book = xlrd.open_workbook(xlfile[0], formatting_info=True)
            except xlrd.XLRDError:
                continue
                
        sheet = book.sheet_by_name('Unit_MaxAvail_Publish')    
        for item in combinations:
            subset = units[item] 
            subset_idx = [i for i, plant in enumerate(sheet.col_values(1)) if plant in subset]
            data[item][day] = {
                'Available capacity [MW]': sum([float(sheet.cell(i,3).value) for i in subset_idx])
            }    
        pbar.update(1)

    pbar.close()
    path = os.path.join(ROOT_DIR, "DataSets/Capacity/EL")
    
    for item in combinations:
        tech, fuel = item
        result = pd.DataFrame.from_dict(data[item], orient='index')
        result['Nominal capacity [MW]'] = total_capacity[item]
        result.to_csv(os.path.join(path, f"{tech}_{fuel}.csv"))
    

def aggregate_available_hydro(period):
    plants = pd.read_csv(os.path.join(ROOT_DIR, "DataSets/PowerPlants/EL/plants.csv"))
    hydro_plants = plants[plants['Technology']=='HDR']['Unit'].tolist()

    path = os.path.join(ROOT_DIR, "RawData/EL/availabilities")
    pbar = tqdm(total=len(period))

    data = defaultdict(dict)
    for day in period:   
        sday = day.strftime('%Y%m%d')
        xlfile = glob.glob(os.path.join(path, sday) + '*')

        if len(xlfile) == 0:
            continue 
        else:
            try:
                book = xlrd.open_workbook(xlfile[0], formatting_info=True)
            except xlrd.XLRDError:
                continue

        sheet = book.sheet_by_name('Unit_MaxAvail_Publish')
        subset_idx = [i for i, plant in enumerate(sheet.col_values(1)) if plant in hydro_plants]
        
        values = {}
        for i in subset_idx:
            unit = sheet.cell(i,1).value
            values[unit] = float(sheet.cell(i,3).value)

        data[day] = values
        pbar.update(1)

    pbar.close()
    result = pd.DataFrame.from_dict(data, orient='index')
    path = os.path.join(ROOT_DIR, "DataSets/Hydro/EL")    
    result.to_csv(os.path.join(path, "availability.csv"))


def aggregate_res_generation(period):
    path = os.path.join(ROOT_DIR, "RawData/EL/res_forecast")
    pbar = tqdm(total=len(period))

    data = {}
    for day in period:   
        sday = day.strftime('%Y%m%d')
        xlfile = glob.glob(os.path.join(path, sday) + '*')

        if len(xlfile) == 0:
            continue 
        else:
            try:
                book = xlrd.open_workbook(xlfile[0], formatting_info=True)
            except xlrd.XLRDError:
                continue

        sheet = book.sheet_by_name(f'{sday}_RES_Forecast')
        idx = sheet.col_values(0).index('Total System')        
        data[day] = sheet.row_values(idx)[1:25]     
        pbar.update(1)

    pbar.close()
    result = pd.DataFrame.from_dict(data, orient='index', columns=range(24))
    result = result.stack()
    result.index = result.index.map(lambda x: x[0].replace(hour=int(x[1])))
    result = result.to_frame('RES generation [MW]')
    path = os.path.join(ROOT_DIR, "DataSets/Generation/EL")
    result.to_csv(os.path.join(path, "RES.csv"))


def aggregate_reservoir_levels(period):
    plants = pd.read_csv(os.path.join(ROOT_DIR, "DataSets/PowerPlants/EL/plants.csv"))
    hydro_plants = plants[plants['Technology']=='HDR']['Unit'].tolist()
    hydro_plants.remove('THESAVROS')
    hydro_plants.extend(['THESAVROS1', 'THESAVROS2', 'THESAVROS3'])

    path = os.path.join(ROOT_DIR, "RawData/EL/reservoirs")
    pbar = tqdm(total=len(period))

    data = {}
    data_avg = {}
    for day in period:   
        sday = day.strftime('%Y%m%d')
        xlfile = glob.glob(os.path.join(path, sday) + '*')

        if len(xlfile) == 0:
            continue 
        else:
            try:
                book = xlrd.open_workbook(xlfile[0], formatting_info=True)
            except xlrd.XLRDError:
                continue

        sheet = book.sheet_by_name(f'{sday}ReservoirFillingRate') 
        idx = [i for i, plant in enumerate(sheet.col_values(1)) if plant in hydro_plants]
        
        values = {}
        for i in idx:
            unit = sheet.cell(i,1).value
            try:
                rate = float(sheet.cell(i,2).value)
            except ValueError:
                rate = np.nan
            finally:
                values[unit] = rate
            
        data[day] = values
        data_avg[day] = sheet.cell(2,10).value
        pbar.update(1)
        
    pbar.close()
    result = pd.DataFrame.from_dict(data, orient='index')
    result['THESAVROS'] = result[['THESAVROS1', 'THESAVROS2', 'THESAVROS3']].mean(axis=1)
    result = result.drop(['THESAVROS1', 'THESAVROS2', 'THESAVROS3'], axis=1)
    path = os.path.join(ROOT_DIR, "DataSets/Hydro/EL")
    result.to_csv(os.path.join(path, "reservoirs.csv"))
    result = pd.DataFrame.from_dict(data_avg, orient='index', columns=['Filling Rate [%]'])
    result.to_csv(os.path.join(path, "reservoirs_avg.csv"))



