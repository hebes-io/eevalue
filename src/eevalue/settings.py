import os
from pathlib import Path

SOURCE_PATH = Path(os.path.dirname(os.path.abspath(__file__))).resolve().parent
PROJECT_PATH = Path(SOURCE_PATH).resolve().parent
DATA_PATH = PROJECT_PATH.joinpath("data")
CONF_PATH = PROJECT_PATH.joinpath("conf")

# Technologies:
"""
Technology 	Description 	                   
COMC 	    Combined cycle 	                   
GTUR 	    Gas turbine 	                   
HDR 	    Hydropower 	           
STUR 	    Steam turbine 	                    	                    
"""
DEFAULT_TECHNOLOGIES = ["COMC", "GTUR", "HDR", "STUR"]
DEFAULT_FUELS = ["GAS", "COAL", "LIG", "NUC", "WAT"]
