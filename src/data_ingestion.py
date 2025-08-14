import pandas as pd
import os
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import getLogger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import readYaml

logger = getLogger(__name__)

class DataIngestion:
    def __init__(self,config):
        self.config = config["data_ingestion"]
        self.bucketName = self.config["bucket_name"]
        self.bucketFileName = self.config["bucket_file_name"]
        self.trainTestSplit = self.config["train_ratio"]

        ## the line self.bucketName = self.config["bucket_name"] is taking the value for "bucket_name" from the config dictionary and saving it to the class as a variable called self.bucketName.
        ## This is a great practice because it means you can change the name of the bucket or the file in a separate configuration file without ever needing to touch the code in this class.
        
        os.makedirs(RAW_DIR, exist_ok = True)

        logger.info(f"Data Ingestion started with {self.bucketName} and file name is {self.bucketFileName}")

    def downloadCsvFromGCP(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucketName)
            blob = bucket.blob(self.bucketFileName)

            blob.download_to_filename(RAW_FILE_PATH)

            logger.info(f"Raw file is successfully downloaded to {RAW_FILE_PATH}.")

        except Exception as e:
            logger.error("Error while downloading the CSV file.")
            raise CustomException("Failed to download CSV file ", e)
        
    def trainTestSplitting(self):
        try:
            logger.info("Starting the data splitting process.")
            data = pd.read_csv(RAW_FILE_PATH)

            trainData, testData = train_test_split(data, test_size = 1 - self.trainTestSplit, random_state = 42)

            trainData.to_csv(TRAIN_FILE_PATH)
            testData.to_csv(TEST_FILE_PATH)

            logger.info(f"Train data saved to {TRAIN_FILE_PATH}.")
            logger.info(f"Test data saved to {TEST_FILE_PATH}.")

        except Exception as e:
            logger.error("Error while splitting the data.")
            raise CustomException("Failed to split the train and test data ",e)
        
    
    def run(self):
        try:
            logger.info("Starting data ingestion process.")
            self.downloadCsvFromGCP()
            self.trainTestSplitting()

            logger.info("Data ingestion completed successfully.")

        except CustomException as e:
            logger.error(f"Custom Exception : {str(e)}.")

        finally:
            logger.info("Data Ingestion completed.")

if __name__ == "__main__":
    dataIngestion = DataIngestion(readYaml(CONFIG_PATH))
    dataIngestion.run()

    ## Think of it like this: the if __name__ == "__main__": block is a special gatekeeper.
    ## When you run a Python file directly, the __name__ variable is automatically set to "__main__". The gate is open, and all the code inside the block runs.
    ## However, if another Python file imports your file as a module, the __name__ variable gets the name of your file. 
    ## The condition is now false, the gate is closed, and the code inside the block is skipped.
    ## This is a best practice that ensures your code only runs its main logic when you intend for it to.
