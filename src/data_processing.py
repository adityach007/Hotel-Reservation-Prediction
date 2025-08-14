import os
import pandas as pd
import numpy as np
from src.logger import getLogger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import readYaml, loadDataFromCSV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = getLogger(__name__)

class DataProcessing:
    def __init__(self, trainPath, testPath, processedDirectory, configPath):
        self.trainPath = trainPath
        self.testPath = testPath
        self.processedDirectory = processedDirectory

        self.config = readYaml(configPath)

        if not os.path.exists(self.processedDirectory):
            os.makedirs(self.processedDirectory)

        
    def processingData(self, df):
        try:
            logger.info("Starting our data processing step.")

            logger.info("Dropping the columns")
            df.drop(columns = ["Unnamed: 0", "Booking_ID"], inplace = True)
            df.drop_duplicates(inplace = True)

            categoricalColumns = self.config["data_processing"]["categorical_columns"]
            numericalColumns = self.config["data_processing"]["numerical_columns"]

            logger.info("Applying label encoding.")

            labelencoder = LabelEncoder()
            mappings = {}

            for col in categoricalColumns:
                df[col] = labelencoder.fit_transform(df[col])
                mappings[col] = {label : code for label, code in zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_))}

            logger.info("Label Mappings are : ")
            for col, mapping in mappings.items():
                logger.info(f"{col} : {mapping}")

            logger.info("Doing skewness handling.")
            skewnessThreshold = self.config["data_processing"]["skewness_threshold"]
            skewness = df[numericalColumns].apply(lambda x : x.skew())

            for col in skewness[skewness > skewnessThreshold].index:
                df[col] = np.log1p(df[col])

            return df

        except Exception as e:
            logger.error("Error during preprocess step {e}.")
            raise CustomException("Error while proprocessing the data ",e)
        
    def balancingData(self, df):
        try:
            logger.info("Handling imbalanced data.")
            
            X = df.drop(columns = "booking_status")
            y = df["booking_status"]

            smote = SMOTE(random_state = 42)
            XResampled, yResampled = smote.fit_resample(X, y)

            balancedDf = pd.DataFrame(XResampled, columns = X.columns)
            balancedDf["booking_status"] = yResampled

            logger.info("Data balanced successfully.")
            return balancedDf

        except Exception as e:
            logger.error(f"Error during balancing data step {e}.")
            raise CustomException("Error while balancing data ", e)

    def featureSelection(self, balancedDf):
        try:
            logger.info("Starting our feature selection.")
            
            X = balancedDf.drop(columns = "booking_status")
            y = balancedDf["booking_status"]

            model = RandomForestClassifier(random_state = 42)
            model.fit(X, y)

            featureImportance = model.feature_importances_
            featureImportanceDf = pd.DataFrame({
                "feature" : X.columns,
                "importance" : featureImportance
            })

            topFeaturesImportanceDf = featureImportanceDf.sort_values(by = "importance", ascending = False)
            numberOfSelectedFeatures = self.config["data_processing"]["no_of_features"]
            
            top10Features = topFeaturesImportanceDf["feature"].head(numberOfSelectedFeatures).values
            logger.info(f"Featrures selected : {top10Features}")
            top10FeaturesDf = balancedDf[top10Features.tolist() + ["booking_status"]]
            logger.info("Feature selection completed succesfully.")

            return top10FeaturesDf
        
        except Exception as e:
            logger.error(f"Error during feature selection {e}.")
            raise CustomException("Failed to perform feature selection ", e)
        
    def saveDataToCSV(self, df, filePath):
        try:
            logger.info("Saving our data in processed folder.")
            df.to_csv(filePath, index = False)

            logger.info(f"Data saved successfully at {filePath}")
        
        except Exception as e:
            logger.info(f"Error while saving data {e}.")
            raise CustomException("Failed to convert to csv ", e)
        
    def process(self):
        try:
            logger.info("Loading data from raw directory.")

            trainDf = loadDataFromCSV(self.trainPath)
            testDf = loadDataFromCSV(self.testPath)

            ## Preprocess step
            trainDf = self.processingData(trainDf)
            testDf = self.processingData(testDf)

            ## Balancing data
            trainDf = self.balancingData(trainDf)
            testDf = self.balancingData(testDf)

            ## Feature selection
            trainDf = self.featureSelection(trainDf)
            testDf = testDf[trainDf.columns]

            ## Saving data
            self.saveDataToCSV(trainDf, PROCESSED_TRAIN_DATA_PATH)
            self.saveDataToCSV(testDf, PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing completed successfully.")

        except Exception as e:
            logger.error(f"Error while proprocessing pipeline {e}")
            raise CustomException("Error suring preprocessing pipeline ", e)
        
if __name__ == "__main__":
    dataProcessor = DataProcessing(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    dataProcessor.process()