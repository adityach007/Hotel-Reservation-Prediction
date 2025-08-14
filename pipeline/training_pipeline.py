from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessing
from src.model_training import ModelTraining
from utils.common_functions import readYaml
from config.paths_config import *

if __name__ == "__main__":
    ## 1. Data ingestion

    dataIngestion = DataIngestion(readYaml(CONFIG_PATH))
    dataIngestion.run()

    ## 2. Data processing

    dataProcessor = DataProcessing(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    dataProcessor.process()

    ## 3. Model training

    trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_TRAINING_PATH)
    trainer.runningModel()