import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import getLogger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import readYaml, loadDataFromCSV
from scipy.stats import randint

import mlflow
import mlflow.sklearn

logger = getLogger(__name__)

class ModelTraining:
    def __init__(self, trainPath, testPath, modelOutputPath):
        self.trainPath = trainPath
        self.testPath = testPath
        self.modelOutputPath = modelOutputPath

        self.paramsDistribution = LIGHTGM_PARAMS
        self.randomSearchParams = RANDOM_SEARCH_PARAMS

    def loadAndSplitData(self):
        try:
            logger.info(f"Start loading training data from {self.trainPath}.")
            trainDf = loadDataFromCSV(self.trainPath)

            logger.info(f"Start loading testing data from {self.testPath}.")
            testDf = loadDataFromCSV(self.testPath)

            XTrain = trainDf.drop(columns = ["booking_status"])
            yTrain = trainDf["booking_status"]

            XTest = testDf.drop(columns = ["booking_status"])
            yTest = testDf["booking_status"]

            logger.info("Data splitted successfully for Model Training.")

            return XTrain, yTrain, XTest, yTest
        
        except Exception as e:
            logger.error(f"Error while loading data {e}.")
            raise CustomException("Failed to load data ", e)
        
    
    def trainingLGBMModel(self, XTrain, yTrain):
        try:
            logger.info("Initializing the model.")
            lgbmModel = lgb.LGBMClassifier(random_state = self.randomSearchParams["random_state"])

            logger.info("Starting our model hypertuning.")
            randomSearch = RandomizedSearchCV(
                estimator = lgbmModel,
                param_distributions = self.paramsDistribution,
                n_iter = self.randomSearchParams["n_iter"],
                cv = self.randomSearchParams["cv"],
                n_jobs = self.randomSearchParams["n_jobs"],
                verbose = self.randomSearchParams["verbose"],
                random_state = self.randomSearchParams["random_state"],
                scoring = self.randomSearchParams["scoring"]
            )

            logger.info("Starting our model hyperparameter tuning.")
            randomSearch.fit(XTrain, yTrain)

            logger.info("Hyperparameter tuning completed.")

            bestParameter = randomSearch.best_params_
            bestLGBMModel = randomSearch.best_estimator_

            logger.info(f"Best parameters are : {bestParameter}.")
            return bestLGBMModel
        
        except Exception as e:
            logger.info(f"Error while training model {e}.")
            raise CustomException("Failed to train model ", e)
        
    
    def evaluateModel(self, model, XTest, yTest):
        try:
            logger.info("Starting model evaluation.")
            yPrediction = model.predict(XTest)

            accuracyScore = accuracy_score(yTest, yPrediction)
            precisionScore = precision_score(yTest, yPrediction)
            recallScore = recall_score(yTest, yPrediction)
            f1Score = f1_score(yTest, yPrediction)

            logger.info(f"Accuracy Score : {accuracyScore}")
            logger.info(f"Precision Score : {precisionScore}")
            logger.info(f"Recall Score : {recallScore}")
            logger.info(f"f1 Score : {f1Score}")
        
            return {
                "accuracy" : accuracyScore,
                "precision" : precisionScore,
                "recall" : recallScore,
                "f1" : f1Score
            }

        except Exception as e:
            logger.info(f"Error while evaluating the model {e}.")
            raise CustomException("Failed to evaluating the model ", e)
        
    def saveModel(self, model):
        try:
            os.makedirs(os.path.dirname(self.modelOutputPath), exist_ok = True)

            logger.info("Saving the model.")
            joblib.dump(model, self.modelOutputPath)
            logger.info(f"Model saved to {self.modelOutputPath}")

        except Exception as e:
            logger.error(f"Error while saving model {e}.")
            raise CustomException("Failed to save the model ", e)
        
    def runningModel(self):
        try:
            with mlflow.start_run():
                logger.info("Starting our model training pipeline.")

                logger.info("Starting our MLFlow experimentation.")
                logger.info("Logging the training and testing dataset to MLFlow.")

                mlflow.log_artifact(self.trainPath, artifact_path = "datasets")
                mlflow.log_artifact(self.testPath, artifact_path = "datasets")

                XTrain, yTrain, XTest, yTest = self.loadAndSplitData()
                bestLGBMModel = self.trainingLGBMModel(XTrain, yTrain)
                metrics = self.evaluateModel(bestLGBMModel, XTest, yTest)
                self.saveModel(bestLGBMModel)

                logger.info("Logging the model into MLFlow.")
                mlflow.log_artifact(self.modelOutputPath)

                logger.info("Logging params and metrics in the MLFlow.")
                mlflow.log_params(bestLGBMModel.get_params())
                mlflow.log_metrics(metrics)

                logger.info("Model training successfully converted.")

        except Exception as e:
            logger.error(f"Error while running the model {e}.")
            raise CustomException("Failed to run the model ", e)
        

if __name__ == "__main__":
    trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_TRAINING_PATH)
    trainer.runningModel()   