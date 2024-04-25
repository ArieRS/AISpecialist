# Sign Language Recognition Project

## About
- This repository is used for Jetson AI Specialist certification. This project is about sign language recognition which was implemented on Jetson Nano. We used 8 classes for this task: book, hello, help, I, I love you, need, standby, and thank.
- You can see the explanation about this project on [presentation](https://youtu.be/WSRoR2fM3FU?si=9gYy7lCFrnV0QCeH).

## Create environment
- On the desktop, you can create the Python environment by using ```conda env create -f desktopenv.yml ```
- And for the Jetson Nano, you can create the environment by using ```conda env create -f jetsonenv.yml ```


## How to Use this code
1. Train the model by using action_detection_training.py
2. The output model from the first step was converted to onnx by using converth5.py
3. Implement the model in the jetson nano by using main.py
