# Paired-t-test-for-Object-detection
In this repository, we provide code to perform paired t-test to establish deep learning model's statistical significance. <br />
The model for which we provide the code is an object detection model. To the best of my knowledge, there is no other repo which provides this kind of code.<br /> 
We have added the results in our paper **"A semi-supervised teacher-student framework for surgical tool detection and localization"**. Please refer here for more details. <br />

## Results
![results](Mansoor_CAI22.png)

### Requirements 

Install following requirements. A requirements file is also provided. 
 ```sh
numpy~=1.23.2
pandas~=1.4.3
seaborn~=0.11.2
matplotlib~=3.5.3
statannot~=0.2.3
torchvision~=0.13.1
   ```
   
### Steps
 ```sh
Step 1: Run the file "prepare_data.py" to save numpy arrays for all the models. 
Step 2: Run "Paired_test.py" to perform t-test. This will compute p-values and save box-plot. 
   ```
## Citing semi-supervised tool detection


## FAQ
If anyone wants to reproduce the code and encounters a problem or wants to give a suggestion, feel free to contact me at my email [a01753093@tec.mx]
