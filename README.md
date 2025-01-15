## Overview
This personal project aim to identify the stage of disease of potatoes. It helps people classify the stage of disease potatoes got infected with
## Features
1. Detects the stages of diseased
2. Trained on dataset of infected potatoes
3. Simple User-Friendly Interface for users to upload the image and classify the disease
## Technologies used
1. Programming language: Python
2. Frameworks: Tensorflow/Pytorch
3. Dataset: Custom labeled images of real infected potato images
4. Deployment: fastapi
## Installtion
  1. Clone the repository
  ``` [https://github.com/kimvongola/Fake-medicines-detection-app.git](https://github.com/kimvongola/Plants-Disease-Classification.git) ```
  2. Download tensorflow,fastapi,pillow,python-serving-apy,,uvicorn,python-multipart,numpy,matplotlib via pip install
  3. Donwload Docker
  4. run main-tf-serving.py
  5. Type ``` docker run -t --rm -p "your port number":"your port number" -v "Path to api folder" tensorflow/serving --rest_api_port="your port number" --model_config_file=self_study_DL/models.config``` in the command prompt and run

  6. Open new command prompt and change your directory to path to frontend
  7. Run the following command ```npm install --from-lock-json > npm audit fix > npm run start

## Usage
1. Click the empty space in the website and selected infected potatoes image to upload

## Result
[Example of healthy potato potato result](Result/IMG_2636.png)

[Example of early blight potato potato result](Result/IMG_2637.png)

[Example of late blight potato potato result](Result/IMG_2638.png)


