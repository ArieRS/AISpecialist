# Sign Language Recognition Project

## Project Demonstration

[![Project Demonstration](https://i9.ytimg.com/vi_webp/dBV_yIljy3c/mq1.webp?sqp=CKDyprEG-oaymwEmCMACELQB8quKqQMa8AEB-AH-CYAC0AWKAgwIABABGEEgEyh_MA8=&rs=AOn4CLBGRz0GOMbrGaflpYY5KkwQCCPaxA)](https://www.youtube.com/embed/dBV_yIljy3c?si=H98ek2FlSMDMeY12)


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
