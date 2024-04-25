import os
from src.auto_driver import AutoDriver
from src.preprocessor import Preprocessor
from src.model_builder import ModelBuilder
from src.model_trainer import ModelTrainer

if __name__ == "__main__":
    auto_driver = AutoDriver('xinglina_lake_model2.h5')
    auto_driver.run()
