# CoronaHack-Chest-X-Ray-Classify
 
Dataset URL:https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset

<div align="center">
<img src="https://github.com/Wade0125Studio/CoronaHack-Chest-X-Ray-Classify/blob/main/img/AI-Covid19-Logo.jpg">
</div>


<h2>Abstract:</h2>
CoronaHack - Chest X-Ray Dataset is a chest X-ray dataset collected during the COVID-19 outbreak. The purpose of this dataset is to assist medical professionals in the diagnosis and treatment of patients with COVID-19. In this study, a convolutional neural network (CNN) was used to classify this dataset to assist in automated COVID-19 diagnosis. The results show that this CNN model has a high classification accuracy when performing two-class classification (Pnemonia,Normal), especially the detection accuracy of COVID-19 is as high as 94.55%. This study shows that using CNN can effectively classify chest X-rays and assist medical professionals in the diagnosis and treatment of COVID-19.
<h2>Introduction:</h2>
Corona COVID19 virus affects the respiratory system of healthy individual & Chest XRay is one of the important imaging methods to identify the corona virus.With the Chest XRay dataset, Develop a Machine Learning Model to classify the X Rays of Healthy vs Pneumonia (Corona) affected patients & this model powers the AI application to test the Corona Virus in Faster Phase.
The training set has a total of 5286 images, 3944 images of the diseased and 1342 images of the non-diseased. The test set has a total of 624 images, 390 images with disease and 234 images without disease.

<h2>Method:</h2>
In this study, we employ various convolutional neural network models to classify the CoronaHack - Chest X-Ray Dataset. These models include Xception, VGG16, VGG19, ResNet101V2, InceptionV3, InceptionResNetV2, EfficientNetV2S, and EfficientNetV2M and self define.
These models are all trained on ImageNet and have good image recognition capabilities. We use these pre trained models to classify pneumonia and normal chest radiographs. By fine tuning the weights of these pre trained models, we can apply these models to our classification tasks and can effectively improve the classification accuracy of the models.
Because the database is limited, we use image enhancement technology (random scaling and random brightness adjustment and random movement of height and width), the optimizer uses Adamax, the learning rate is 0.0001, and the training is 100 times.
<h2>Experiment and Results:</h2>
<h3>Xception</h3>
<div align="center">
<img src="https://github.com/Wade0125Studio/CoronaHack-Chest-X-Ray-Classify/blob/main/img/Training_history%20Xception.png">
</div>
<div align="center">
<img src="https://github.com/Wade0125Studio/CoronaHack-Chest-X-Ray-Classify/blob/main/img/Confusion%20Matrix%20Xception.png">
</div>
<div align="center">
<img src="https://github.com/Wade0125Studio/CoronaHack-Chest-X-Ray-Classify/blob/main/img/Xception-Results.png">
</div>
<div align="center">
<img src="https://github.com/Wade0125Studio/CoronaHack-Chest-X-Ray-Classify/blob/main/img/Xception%20results%20EX.png">
</div>

<h3>VGG16</h3>
<div align="center">
<img src="https://github.com/Wade0125Studio/CoronaHack-Chest-X-Ray-Classify/blob/main/img/Training_history%20VGG16.png">
</div>
<div align="center">
<img src="https://github.com/Wade0125Studio/CoronaHack-Chest-X-Ray-Classify/blob/main/img/Confusion%20Matrix%20VGG16.png">
</div>
<div align="center">
<img src="https://github.com/Wade0125Studio/CoronaHack-Chest-X-Ray-Classify/blob/main/img/VGG16-Results.png">
</div>
<div align="center">
<img src="https://github.com/Wade0125Studio/CoronaHack-Chest-X-Ray-Classify/blob/main/img/VGG16%20results%20EX.png">
</div>
<h3>VGG19</h3>
<h3>ResNet101V2</h3>
<h3>InceptionV3</h3>
<h3>InceptionResNetV2</h3>
<h3>EfficientNetV2S</h3>
<h3>EfficientNetV2M</h3>
<h3>Self-Define</h3>


<h2>Conclusion:</h2>
In this study, we used multiple convolutional neural network (CNN) models to classify the CoronaHack - Chest X-Ray Dataset collected during the COVID-19 pandemic, with the aim of assisting in the automation of COVID-19 diagnosis. Through experimental comparisons of various models, we found that the VGG19 model had the highest classification accuracy, followed by the Self-Define, InceptionResNetV2, and EfficientNetV2M models. The COVID-19 detection accuracy of these models reached an extremely high level.
These results indicate that CNN models have high accuracy in classifying chest
XRays, which can effectively assist medical professionals in diagnosing and treating COVID 19. In addition,this study provides an open and reliable chest XRays dataset, which can promote the development and progress of related research.
In the future, we can further study how to optimize the performance of these models and develop more effective automated COVID 19 diagnosis systems to address the challenges of the COVID 19 pandemic.




