import os
from src.auto_driver import AutoDriver
from manual_recorder import ManualRecorder
from src.preprocessor import Preprocessor
from src.model_trainer import ModelBuilder
from src.model_trainer import ModelTrainer

if __name__ == "__main__":
    model_path = 'models/model.h5'
    if os.path.exists(model_path):
        # If trained model exists, drive the car automatically
        autonomous_driver = AutoDriver(model_path)
        autonomous_driver.drive_autonomously()
    else:
        # If trained model doesn't exist, switch to manual mode and start recording
        manual_recorder = ManualRecorder()
        record_road = input(
            '''If you want to enable auto-driver, 
            you should record the road while driving by yourself 
            until the auto-driver gathers enough data. 
            Do you want to record while driving? (yes/no): ''')

        if record_road.lower() == 'yes':
            # Start recording
            manual_recorder.start_recording()

            # Gather enough data for training
            input("Press Enter to stop recording...")
            manual_recorder.stop_recording()

            # Preprocess recorded data
            preprocess_data('recorded_data/')

            # Train model with recorded data
            build_model()
            train_model()

            # Move trained model to models folder
            os.makedirs('models', exist_ok=True)
            os.rename('model.h5', model_path)

            # Drive the car automatically after training
            autonomous_driver = AutonomousDriver(model_path)
            autonomous_driver.drive_autonomously()
        else:
            # Allow manual driving without recording
            print("Manual driving mode activated.")
