# UDS (Udacity Driverless System)

"UDS, short for "Udacity Driverless System," is an autonomous driving project developed using deep learning techniques and socket communication with a simulator. 
It was developed as an internship project during my internship at Zhilin Information Technology Co., Ltd., organized by Taiyuan University of Technology and the company itself. The internship took place from April 10, 2024, to April 23, 2024.

## Overview
This project aims to demonstrate the application of deep learning in creating a system capable of autonomous driving within a simulated environment. By leveraging neural network models and real-time data communication with the simulator, UDS simulates the behavior of an autonomous vehicle navigating various tracks. Throughout the development process, emphasis was placed on implementing robust algorithms for image preprocessing, model training, and real-time control of the simulated vehicle. The project represents a practical exploration of AI-driven autonomous driving systems and their potential applications in real-world scenarios.

## Technologies Used

UDS leverages the following technologies:
1. **Deep Learning Technology:** Deep learning techniques are employed to develop and train the neural network models responsible for autonomous driving decision-making.
2. **Python Programming Technology:** Python serves as the primary programming language for implementing various components of the project, including data preprocessing, model training, and system control.
3. **Network Communication Technology:** Network communication technology facilitates communication between the autonomous driving system and the simulation environment, enabling real-time interaction and control.
4. **Udacity Simulator:** The project interfaces with the Udacity Simulator, a crucial component for collecting training data, testing the autonomous driving system, and evaluating its performance.

## Installation Guide

1. Download and install Python v3.7.4.
2. Open the console in the root directory of the project and run `pip install .`. It will use `setup.py` to install all the requirements. If it gives an error, then run `pip install --upgrade pip setuptools wheel` to clean up the environment. The `setup.py` will install all the requirements along with the Udacity Simulator.
3. In PyCharm, if prompted to create a new venv, ignore it and then select Python v3.7.4 as the project interpreter from `File > Settings > Project > Python Interpreter`.

## Usage Guide

1. Create folders "data-lake" and "data-mountain" in the root directory of the project.
2. Disconnect from the internet and run the Udacity simulator (`term1-simulator-windows > beta_simulator_windows > beta_simulator.exe`).
3. Collect data by driving the car in the Udacity simulator.
4. Select the track you want to record data from and then click "Training Mode" in the Udacity simulator.
5. Click the record button and select the folder you want to save the data in (e.g., "data-lake").
6. Start driving, stay in the center of the road while driving, do 5 circles, and then click the record button again to save the recorded data (collect at least 10,000 images).
7. After collecting data using Udacity simulator training mode, open `main.py` and run the provided code:
    ```python
    from src.model_trainer import ModelTrainer
    
    if __name__ == "__main__":
        model_trainer = ModelTrainer('./data-lake/')
        model_trainer.train_model('./models/', 'agajan_lake_model')
    ```
8. If you collected images for the lake track, don't need to change the code. But if you selected the mountain track, replace "lake" with "mountain" in the code.
9. The code will preprocess images, build a model, and then train the data, generating the trained data in the "models/" folder. It will take around 30 minutes to train a model.
10. After it finishes training the model, you can test it by running the provided code:
    ```python
    from src.auto_driver import AutoDriver
    
    if __name__ == "__main__":
        auto_driver = AutoDriver('models/agajan_lake_model.h5')
        auto_driver.run(15)
    ```
11. After running the code, the console should display this message: `(3976) wsgi starting up on http://0.0.0.0:4567`.
12. Open the Udacity simulator, select the track based on your trained data, and enter the "Autonomous mode". If the car drives automatically by itself and doesn't crash, then it means that UDS is working properly.

## Implementation

I used the OOP approach to design the system. The system consists of the following components:

- `main.py`: Entry point of the system. Initializes the autonomous driver and starts the system.
- `src`: Directory containing all the components of the system.
  - `auto_driver.py`: Defines the `AutoDriver` class responsible for controlling the car autonomously using a trained model.
  - `config.py`: Configuration file defining constants used throughout the system.
  - `pi_controller.py`: Implements the PI controller for controlling the throttle.
  - `preprocessor.py`: Defines the `Preprocessor` class responsible for preprocessing images before feeding them into the model.
  - `model_builder.py`: Builds different neural network models for training.
  - `model_trainer.py`: Trains the neural network model using data collected from the simulator.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

Special thanks to the Udacity Self-Driving Car Nanodegree program for inspiration and guidance.

## Contributing

UDS welcomes contributions from the open-source community, offering opportunities for collaboration, feedback, and further improvement. Whether you're interested in deep learning, autonomous systems, or simulation environments, there are ample opportunities to get involved and make a meaningful impact.

Join us in exploring the exciting intersection of artificial intelligence and autonomous vehicles, as we continue to advance the capabilities and reliability of the Udacity Driverless System."
Contact us though email - agajan.st@gmail.com.