# Dialect Classification Project
## Description
The primary purpose of this project was twofold:
1. To design and train a deep learning model that is capable of classifying a given speech sample into one of 6 American dialect categories
2. To build a user-facing web application that allows users to upload their own audio clips and have them classified in real time by the deep learning system

The diagram below illustrates the processing logic of the fully integrated web application:
![alt text](https://i.imgur.com/H8k4TOO.png)

The web application was built using the Flask micro-framework, which allowed us to combine powerful machine learning libraries with web functionality in a native Python environment.
## How to Use
The primary application files are located in the `flask_app` directory, in which the `app.py` file contains the main program driver. Within this directory, the back-end classifcation code can be found within the `src` folder, while the front-end web files can be found in the `static` and `template` folders.

To run the application, simply run the `app.py` file and navigate to the locally hosted URL shown in the console.
