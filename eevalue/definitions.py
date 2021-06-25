import os 
from pathlib import Path

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(SOURCE_DIR).parent

# Technologies:
"""
Technology 	Description 	                   
COMC 	    Combined cycle 	                   
GTUR 	    Gas turbine 	                   
HDR 	    Hydropower 	           
STUR 	    Steam turbine 	                    	                    
"""
DEFAULT_TECHNOLOGIES  = ['COMC', 'GTUR', 'HDR', 'STUR']
DEFAULT_FUELS         = ['GAS', 'COAL', 'LIG', 'NUC', 'WAT']
DEFAULT_MARKETS       = ['DA', '2U', '2D']