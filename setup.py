import os
from setuptools import setup, find_packages
import requests
import zipfile

# Define package metadata and dependencies
setup(
    name='uds',
    version='v1.0.0',
    packages=find_packages(),
    url='https://github.com/agajansahatov/uds',
    license='',
    author='Adam',
    author_email='agajansahatov@icloud.com',
    description='Udacity Driverless System based on Deep Learning',
    install_requires=[
        'keras==2.6.0',
        'tensorflow==2.6.0',
        'scikit-learn',
        'python-socketio==4.6.0',
        'Flask-SocketIO==4.3.1',
        'python-engineio==3.13.2',
        'opencv-python==4.4.0.42',
        'protobuf==3.19.0',
        'pillow',
        'pandas',
        'eventlet',
        'flask'
    ],
)


# Function to download and extract the simulator
def install_simulator():
    simulator_url = "https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-windows.zip"
    simulator_dir = "term1-simulator-windows"

    try:
        # Download the simulator zip file
        response = requests.get(simulator_url)
        response.raise_for_status()  # Check if the request was successful
        with open("simulator.zip", "wb") as f:
            f.write(response.content)

        # Extract the simulator files
        with zipfile.ZipFile("simulator.zip", "r") as zip_ref:
            zip_ref.extractall(simulator_dir)

    except Exception as e:
        print(f"Error occurred while downloading or extracting the simulator: {e}")
    finally:
        # Clean up the zip file
        if os.path.exists("simulator.zip"):
            os.remove("simulator.zip")


# Call the install_simulator function
install_simulator()
