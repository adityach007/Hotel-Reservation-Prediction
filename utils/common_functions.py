import os
import pandas
from src.logger import getLogger
from src.custom_exception import CustomException
import yaml

import pandas as pd

logger = getLogger(__name__)
## The __name__ variable is a special, built-in variable in Python. Its value is the name of the current module (or file).
## Think of it like this: if you have a log message that says "Successfully read the YAML file," it's more helpful if the log also tells you which file that message came from. 
## Naming the logger with __name__ gives each log message a "source" or a "name tag," making it much easier to track down where things are happening in a big project.

def readYaml(filePath): 
    try:
        if not os.path.exists(filePath):
            raise FileNotFoundError(f"File is not in the given path.")
        
        with open(filePath, 'r') as yamlFile:
            config = yaml.safe_load(yamlFile)
            logger.info("Successfully read the YAML file.")
            return config
    
    except Exception as e:
        logger.error("Error while reading YAML file.")
        raise CustomException("Failed to read the YAML file", e)
    
## Data Processing function
def loadDataFromCSV(pathOfCSV):
    try:
        logger.info("Loading data from csv.")
        return pd.read_csv(pathOfCSV)
    
    except Exception as e:
        logger.error("Error loading the data.")
        raise CustomException("Failed to load data", e)