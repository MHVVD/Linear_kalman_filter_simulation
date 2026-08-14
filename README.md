# Linear Kalman Filter Simulation: Noisy Mouse

This repository contains a pure NumPy and OpenCV implementation of a basic Linear Kalman Filter. It takes a noisy mouse position (simulating poor sensor data like a bad GPS) and smooths it in real-time, demonstrating how the mathematical equations from standard Kalman Filter theory work in practice without relying on opaque library functions.

## Prerequisites
- Python 3.x
- NumPy
- OpenCV-Python

## Installation
```bash
pip install numpy opencv-python
```

## Running the Simulation
Execute the script from your terminal:
```bash
python noisy_mouse.py
```

## How it Works
When the window opens, move your mouse around. 
- A faint **Green Dot** shows where your mouse actually is (the truth).
- Wildly scattering **Red Crosses** simulate a noisy sensor.
- A thick **Blue Line** shows the Kalman Filter's estimate, which successfully cuts through the noise to track your real movement.

Press **q** to quit the application.
