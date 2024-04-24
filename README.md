# uds
UDS stands for "Udacity Driverless System". It is a Udacity Autonomous Driving System based on Deep Learning. 

## Installation

1. Download and install Python v3.7.4 
2. Then open the console in the root directory of the project and run "pip install .".
If it gives error, then run "pip install --upgrade pip setuptools wheel" to clean up the environment.
3. In Pycharm, when you open the project if it interrupts you to create a new venv just ignore it and then select the Python v3.7.4 as project interpreter from settings File>Settings>Project>Python Interpreter.


## Implementation
1. Create folders "data-lake" and "data-mountain" in the root directory of the project.
2. Disconnect from the internet and run the udacity simulator - term1-simulator-windows>beta_simulator_windows>beta_simulator.exe
3. Now we need to collect data by driving the car in udacity simulator.
4. In the udacity simulator select Track you want to record data from and then click "Training Mode".
5. In the training mode click the record button and select the folder you want to save the data, for example if you selected the lake, then chooce folder "data-lake".
6. Start driving, do 5 circles and then click the record button again to save the recorded data.