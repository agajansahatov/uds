import os
from src.auto_driver import AutoDriver
from src.preprocessor import Preprocessor
from src.model_builder import ModelBuilder
from src.model_trainer import ModelTrainer

if __name__ == "__main__":
    # model_trainer = ModelTrainer('./data-mountain/')
    # model_trainer.train_model('./models/', 'agajan_mountain_model')
    auto_driver = AutoDriver('models/agajan_mountain_model.h5')
    auto_driver.run(15)
