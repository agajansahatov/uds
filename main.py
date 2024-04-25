import os
from src.auto_driver import AutoDriver
from src.preprocessor import Preprocessor
from src.model_builder import ModelBuilder
from src.model_trainer import ModelTrainer

if __name__ == "__main__":
    auto_driver = AutoDriver('models/agajan_lake_model.h5')
    auto_driver.run(15)
