import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
  except RuntimeError as e:
    print(e)
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix,accuracy_score
from plot_keras_history import show_history, plot_history
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import random
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from sklearn.metrics import precision_score,recall_score
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, MaxPool2D, Add
from tensorflow.keras.callbacks import LearningRateScheduler,ModelCheckpoint
#----------------------------------Load data------------------------------------------

df_meta=pd.read_csv("./Coronahack-Chest-XRay-Dataset/Chest_xray_Corona_Metadata.csv",index_col=0)

df_meta.head(10)





x = ["Normal","Pnemonia"]
h = [df_meta[ (df_meta["Label"]=="Normal")].shape[0],
     df_meta[ (df_meta["Label"]=="Pnemonia")].shape[0]] 
color = ['b','r']
plt.bar(x,h,color=color)
plt.title("All Dataset Label Distributed")
plt.xlabel("Label")
plt.ylabel("Count")
for a,b in zip(x,h):

    plt.text(a, b, '%.0f' % b, ha='center')
plt.show()


x = ["Normal","Pnemonia"]
h = [df_meta[(df_meta["Dataset_type"]=="TRAIN") & (df_meta["Label"]=="Normal")].shape[0],
     df_meta[(df_meta["Dataset_type"]=="TRAIN") & (df_meta["Label"]=="Pnemonia")].shape[0]] 
color = ['b','r']
plt.bar(x,h,color=color)
plt.title("Train Dataset Label Distributed")
plt.xlabel("Label")
plt.ylabel("Count")
for a,b in zip(x,h):

    plt.text(a, b, '%.0f' % b, ha='center')
plt.show()


x = ["Normal","Pnemonia"]
h = [df_meta[(df_meta["Dataset_type"]=="TEST") & (df_meta["Label"]=="Normal")].shape[0],
     df_meta[(df_meta["Dataset_type"]=="TEST") & (df_meta["Label"]=="Pnemonia")].shape[0]]
color = ['b','r']
plt.bar(x,h,color=color)
plt.title("Test Dataset Label Distributed")
plt.xlabel("Label")
plt.ylabel("Count")
for a,b in zip(x,h):

    plt.text(a, b, '%.0f' % b, ha='center')
plt.show()


train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input,zoom_range=0.1,brightness_range=[0.5,1.3],
                                   width_shift_range=0.1,height_shift_range=0.1,validation_split=0.1)
test_datagen=ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input)


BATCH_SIZE=64
path="./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/"
train_images=train_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TRAIN"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=True,seed=1234,subset='training', directory=path+"/train")

val_images=train_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TRAIN"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=True,seed=1234,subset='validation', directory=path+"/train")

test_images = test_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TEST"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=False, directory=path+"/test")

#----------------------------------Xception------------------------------------------
inputs = tf.keras.layers.Input((150,150,3))
base_model=tf.keras.applications.xception.Xception(include_top=False, weights="imagenet",input_shape=(150,150,3), pooling='avg') 
x=base_model(inputs)
output=layers.Dense(2, activation='sigmoid')(x)
model=tf.keras.models.Model(inputs=inputs, outputs=output)
model.compile(Adamax(learning_rate=1e-4), loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(train_images, validation_data=val_images, epochs=100)
model.save("./model/Xception.h5")

#model = keras.models.load_model("./model/Xception.h5")

show_history(history)
plot_history(history, path="./img/Training_history Xception.png",title="Xception Training history")
plt.close()
preds=model.predict(test_images)
preds = np.argmax(preds,axis=1)
preds=preds>0.5
gt=[0 if x=="Normal" else 1 for x in df_meta[df_meta["Dataset_type"]=="TEST"]["Label"]]
print("Results on test set:")
print('---------------------------------------------------------')
print('Accuracy:'+str(accuracy_score(gt,preds)))
print('---------------------------------------------------------')
print('F1-Score:'+str(f1_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print('Recall:'+str(recall_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print('Precision:'+str(precision_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print("ROC AUC score:",roc_auc_score(gt,preds))
print('---------------------------------------------------------')
print(classification_report(gt,preds,target_names=["Normal","Covid"]))

conf_matrix = confusion_matrix(gt, preds)
sns.heatmap(conf_matrix,xticklabels = ["Normal","Covid"], yticklabels =["Normal","Covid"],annot=True,fmt='g')
plt.title('Xception Confusion Matrix')
plt.savefig("./img/Confusion Matrix Xception.png")
plt.show()


test_images.labels
preds=model.predict(test_images)
preds = np.argmax(preds,axis=1)
preds=preds>0.5
labels = (test_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in preds]
real=[labels[k] for k in test_images.labels]
filenames=test_images.filenames
path_test="./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/"
r=random.randint(0,len(os.listdir(path_test)))
img=cv2.imread(path_test+filenames[r])
plt.imshow(img)
plt.title("Xception Results:Predicton:{0} | Real:{1}".format(predictions[r],real[r]))
plt.savefig("./img/Xception Results EX.png")
results=pd.DataFrame({"Filename":filenames,"real":real,"Predictions":predictions})
results.to_csv("./result/Xception Results.csv",index=False)


#----------------------------------VGG16------------------------------------------


train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input,zoom_range=0.1,brightness_range=[0.5,1.3],
                                   width_shift_range=0.1,height_shift_range=0.1,validation_split=0.1)
test_datagen=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)


BATCH_SIZE=64
path="./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/"
train_images=train_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TRAIN"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=True,seed=1234,subset='training', directory=path+"/train")

val_images=train_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TRAIN"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=True,seed=1234,subset='validation', directory=path+"/train")

test_images = test_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TEST"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=False, directory=path+"/test")


inputs = tf.keras.layers.Input((150,150,3))
base_model=tf.keras.applications.VGG16(include_top=False, weights="imagenet",input_shape=(150,150,3), pooling='avg') 
x=base_model(inputs)
output=layers.Dense(2, activation='sigmoid')(x)
model=tf.keras.models.Model(inputs=inputs, outputs=output)
model.compile(Adamax(learning_rate=1e-4), loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(train_images, validation_data=val_images, epochs=100)
model.save("./model/VGG16.h5")

#model = keras.models.load_model("./model/VGG16.h5")

show_history(history)
plot_history(history, path="./img/Training_history VGG16.png",
             title="Training history VGG16")
plt.close()
preds=model.predict(test_images)
preds = np.argmax(preds,axis=1)
preds=preds>0.5
gt=[0 if x=="Normal" else 1 for x in df_meta[df_meta["Dataset_type"]=="TEST"]["Label"]]
print("Results on test set:")
print('---------------------------------------------------------')
print('Accuracy:'+str(accuracy_score(gt,preds)))
print('---------------------------------------------------------')
print('F1-Score:'+str(f1_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print('Recall:'+str(recall_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print('Precision:'+str(precision_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print("ROC AUC score:",roc_auc_score(gt,preds))
print('---------------------------------------------------------')
print(classification_report(gt,preds,target_names=["Normal","Covid"]))
conf_matrix = confusion_matrix(gt, preds)
sns.heatmap(conf_matrix,xticklabels = ["Normal","Covid"], yticklabels =["Normal","Covid"],annot=True,fmt='g')
plt.title('VGG16 Confusion Matrix')
plt.savefig("./img/Confusion Matrix VGG16.png")
plt.show()



test_images.labels
preds=model.predict(test_images)
preds = np.argmax(preds,axis=1)
preds=preds>0.5
labels = (test_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in preds]
real=[labels[k] for k in test_images.labels]
filenames=test_images.filenames
path_test="./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/"
r=random.randint(0,len(os.listdir(path_test)))
img=cv2.imread(path_test+filenames[r])
plt.imshow(img)
plt.title("VGG16 Results:Predicton:{0} | Real:{1}".format(predictions[r],real[r]))
plt.savefig("./img/VGG16 Results EX.png")
results=pd.DataFrame({"Filename":filenames,"real":real,"Predictions":predictions})
results.to_csv("./result/VGG16 Results.csv",index=False)



#----------------------------------VGG19------------------------------------------

train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg19.preprocess_input,zoom_range=0.1,brightness_range=[0.5,1.3],
                                   width_shift_range=0.1,height_shift_range=0.1,validation_split=0.1)
test_datagen=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg19.preprocess_input)


BATCH_SIZE=64
path="./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/"
train_images=train_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TRAIN"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=True,seed=1234,subset='training', directory=path+"/train")

val_images=train_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TRAIN"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=True,seed=1234,subset='validation', directory=path+"/train")

test_images = test_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TEST"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=False, directory=path+"/test")


inputs = tf.keras.layers.Input((150,150,3))
base_model=tf.keras.applications.VGG19(include_top=False, weights="imagenet",input_shape=(150,150,3), pooling='avg') 
x=base_model(inputs)
output=layers.Dense(2, activation='sigmoid')(x)
model=tf.keras.models.Model(inputs=inputs, outputs=output)
model.compile(Adamax(learning_rate=1e-4), loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(train_images, validation_data=val_images, epochs=100)
model.save("./model/VGG19.h5")

model = keras.models.load_model("./model/VGG19.h5")
show_history(history)
plot_history(history, path="./img/Training_history VGG19.png",
             title="Training history VGG19")
plt.close()
preds=model.predict(test_images)
preds = np.argmax(preds,axis=1)
preds=preds>0.5
gt=[0 if x=="Normal" else 1 for x in df_meta[df_meta["Dataset_type"]=="TEST"]["Label"]]
print("Results on test set:")
print('---------------------------------------------------------')
print('Accuracy:'+str(accuracy_score(gt,preds)))
print('---------------------------------------------------------')
print('F1-Score:'+str(f1_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print('Recall:'+str(recall_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print('Precision:'+str(precision_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print("ROC AUC score:",roc_auc_score(gt,preds))
print('---------------------------------------------------------')
print(classification_report(gt,preds,target_names=["Normal","Covid"]))
conf_matrix = confusion_matrix(gt, preds)
sns.heatmap(conf_matrix,xticklabels = ["Normal","Covid"], yticklabels =["Normal","Covid"],annot=True,fmt='g')
plt.title('VGG19 Confusion Matrix')
plt.savefig("./img/Confusion Matrix VGG19.png")
plt.show()


test_images.labels
preds=model.predict(test_images)
preds = np.argmax(preds,axis=1)
preds=preds>0.5
labels = (test_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in preds]
real=[labels[k] for k in test_images.labels]
filenames=test_images.filenames
path_test="./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/"
r=random.randint(0,len(os.listdir(path_test)))
img=cv2.imread(path_test+filenames[r])
plt.imshow(img)
plt.title("VGG19 Results:Predicton:{0} | Real:{1}".format(predictions[r],real[r]))
plt.savefig("./img/VGG19 Results EX.png")
results=pd.DataFrame({"Filename":filenames,"real":real,"Predictions":predictions})
results.to_csv("./result/VGG19 Results.csv",index=False)



#----------------------------------ResNet101V2------------------------------------------


train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input,zoom_range=0.1,brightness_range=[0.5,1.3],
                                   width_shift_range=0.1,height_shift_range=0.1,validation_split=0.1)
test_datagen=ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input)


BATCH_SIZE=64
path="./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/"
train_images=train_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TRAIN"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=True,seed=1234,subset='training', directory=path+"/train")

val_images=train_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TRAIN"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=True,seed=1234,subset='validation', directory=path+"/train")

test_images = test_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TEST"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=False, directory=path+"/test")


inputs = tf.keras.layers.Input((150,150,3))
base_model=tf.keras.applications.ResNet101V2(include_top=False, weights="imagenet",input_shape=(150,150,3), pooling='avg') 
x=base_model(inputs)
output=layers.Dense(2, activation='sigmoid')(x)
model=tf.keras.models.Model(inputs=inputs, outputs=output)
model.compile(Adamax(learning_rate=1e-4), loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(train_images, validation_data=val_images, epochs=100)
model.save("./model/ResNet101V2.h5")

model = keras.models.load_model("./model/ResNet101V2.h5")
show_history(history)
plot_history(history, path="./img/Training_history ResNet101V2.png",
             title="Training history ResNet101V2")
plt.close()
preds=model.predict(test_images)
preds = np.argmax(preds,axis=1)
preds=preds>0.5
gt=[0 if x=="Normal" else 1 for x in df_meta[df_meta["Dataset_type"]=="TEST"]["Label"]]
print("Results on test set:")
print('---------------------------------------------------------')
print('Accuracy:'+str(accuracy_score(gt,preds)))
print('---------------------------------------------------------')
print('F1-Score:'+str(f1_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print('Recall:'+str(recall_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print('Precision:'+str(precision_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print("ROC AUC score:",roc_auc_score(gt,preds))
print('---------------------------------------------------------')
print(classification_report(gt,preds,target_names=["Normal","Covid"]))
conf_matrix = confusion_matrix(gt, preds)
sns.heatmap(conf_matrix,xticklabels = ["Normal","Covid"], yticklabels =["Normal","Covid"],annot=True,fmt='g')
plt.title('ResNet101V2 Confusion Matrix')
plt.savefig("./img/Confusion Matrix ResNet101V2.png")
plt.show()


test_images.labels
preds=model.predict(test_images)
preds = np.argmax(preds,axis=1)
preds=preds>0.5
labels = (test_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in preds]
real=[labels[k] for k in test_images.labels]
filenames=test_images.filenames
path_test="./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/"
r=random.randint(0,len(os.listdir(path_test)))
img=cv2.imread(path_test+filenames[r])
plt.imshow(img)
plt.title("ResNet101V2 Results:Predicton:{0} | Real:{1}".format(predictions[r],real[r]))
plt.savefig("./img/ResNet101V2 Results EX.png")
results=pd.DataFrame({"Filename":filenames,"real":real,"Predictions":predictions})
results.to_csv("./result/ResNet101V2 Results.csv",index=False)


#----------------------------------InceptionV3------------------------------------------


train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_v3.preprocess_input,zoom_range=0.1,brightness_range=[0.5,1.3],
                                   width_shift_range=0.1,height_shift_range=0.1,validation_split=0.1)
test_datagen=ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_v3.preprocess_input)


BATCH_SIZE=64
path="./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/"
train_images=train_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TRAIN"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=True,seed=1234,subset='training', directory=path+"/train")

val_images=train_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TRAIN"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=True,seed=1234,subset='validation', directory=path+"/train")

test_images = test_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TEST"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=False, directory=path+"/test")


inputs = tf.keras.layers.Input((150,150,3))
base_model=tf.keras.applications.InceptionV3(include_top=False, weights="imagenet",input_shape=(150,150,3), pooling='avg') 
x=base_model(inputs)
output=layers.Dense(2, activation='sigmoid')(x)
model=tf.keras.models.Model(inputs=inputs, outputs=output)
model.compile(Adamax(learning_rate=1e-4), loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(train_images, validation_data=val_images, epochs=100)
model.save("./model/InceptionV3.h5")

model = keras.models.load_model("./model/InceptionV3.h5")
show_history(history)
plot_history(history, path="./img/Training_history InceptionV3.png",
             title="Training history InceptionV3")
plt.close()
preds=model.predict(test_images)
preds = np.argmax(preds,axis=1)
preds=preds>0.5
gt=[0 if x=="Normal" else 1 for x in df_meta[df_meta["Dataset_type"]=="TEST"]["Label"]]
print("Results on test set:")
print('---------------------------------------------------------')
print('Accuracy:'+str(accuracy_score(gt,preds)))
print('---------------------------------------------------------')
print('F1-Score:'+str(f1_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print('Recall:'+str(recall_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print('Precision:'+str(precision_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print("ROC AUC score:",roc_auc_score(gt,preds))
print('---------------------------------------------------------')
print(classification_report(gt,preds,target_names=["Normal","Covid"]))
conf_matrix = confusion_matrix(gt, preds)
sns.heatmap(conf_matrix,xticklabels = ["Normal","Covid"], yticklabels =["Normal","Covid"],annot=True,fmt='g')
plt.title('InceptionV3 Confusion Matrix')
plt.savefig("./img/Confusion Matrix InceptionV3.png")
plt.show()


test_images.labels
preds=model.predict(test_images)
preds = np.argmax(preds,axis=1)
preds=preds>0.5
labels = (test_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in preds]
real=[labels[k] for k in test_images.labels]
filenames=test_images.filenames
path_test="./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/"
r=random.randint(0,len(os.listdir(path_test)))
img=cv2.imread(path_test+filenames[r])
plt.imshow(img)
plt.title("InceptionV3 Results:Predicton:{0} | Real:{1}".format(predictions[r],real[r]))
plt.savefig("./img/InceptionV3 Results EX.png")
results=pd.DataFrame({"Filename":filenames,"real":real,"Predictions":predictions})
results.to_csv("./result/InceptionV3 Results.csv",index=False)


#----------------------------------InceptionResNetV2------------------------------------------


train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input,zoom_range=0.1,brightness_range=[0.5,1.3],
                                   width_shift_range=0.1,height_shift_range=0.1,validation_split=0.1)
test_datagen=ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input)


BATCH_SIZE=64
path="./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/"
train_images=train_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TRAIN"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=True,seed=1234,subset='training', directory=path+"/train")

val_images=train_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TRAIN"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=True,seed=1234,subset='validation', directory=path+"/train")

test_images = test_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TEST"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=False, directory=path+"/test")


inputs = tf.keras.layers.Input((150,150,3))
base_model=tf.keras.applications.InceptionResNetV2(include_top=False, weights="imagenet",input_shape=(150,150,3), pooling='avg') 
x=base_model(inputs)
output=layers.Dense(2, activation='sigmoid')(x)
model=tf.keras.models.Model(inputs=inputs, outputs=output)
model.compile(Adamax(learning_rate=1e-4), loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(train_images, validation_data=val_images, epochs=100)
model.save("./model/InceptionResNetV2.h5")

model = keras.models.load_model("./model/InceptionResNetV2.h5")
show_history(history)
plot_history(history, path="./img/Training_history InceptionResNetV2.png",
             title="Training history InceptionResNetV2")
plt.close()
preds=model.predict(test_images)
preds = np.argmax(preds,axis=1)
preds=preds>0.5
gt=[0 if x=="Normal" else 1 for x in df_meta[df_meta["Dataset_type"]=="TEST"]["Label"]]
print("Results on test set:")
print('---------------------------------------------------------')
print('Accuracy:'+str(accuracy_score(gt,preds)))
print('---------------------------------------------------------')
print('F1-Score:'+str(f1_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print('Recall:'+str(recall_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print('Precision:'+str(precision_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print("ROC AUC score:",roc_auc_score(gt,preds))
print('---------------------------------------------------------')
print(classification_report(gt,preds,target_names=["Normal","Covid"]))
conf_matrix = confusion_matrix(gt, preds)
sns.heatmap(conf_matrix,xticklabels = ["Normal","Covid"], yticklabels =["Normal","Covid"],annot=True,fmt='g')
plt.title('InceptionResNetV2 Confusion Matrix')
plt.savefig("./img/Confusion Matrix InceptionResNetV2.png")
plt.show()

test_images.labels
preds=model.predict(test_images)
preds = np.argmax(preds,axis=1)
preds=preds>0.5
labels = (test_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in preds]
real=[labels[k] for k in test_images.labels]
filenames=test_images.filenames
path_test="./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/"
r=random.randint(0,len(os.listdir(path_test)))
img=cv2.imread(path_test+filenames[r])
plt.imshow(img)
plt.title("InceptionResNetV2 Results:Predicton:{0} | Real:{1}".format(predictions[r],real[r]))
plt.savefig("./img/InceptionResNetV2 Results EX.png")
results=pd.DataFrame({"Filename":filenames,"real":real,"Predictions":predictions})
results.to_csv("./result/InceptionResNetV2 Results.csv",index=False)


#----------------------------------EfficientNetV2S------------------------------------------
BATCH_SIZE=16
path="./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/"


train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input,zoom_range=0.1,brightness_range=[0.5,1.3],
                                   width_shift_range=0.1,height_shift_range=0.1,validation_split=0.1)
test_datagen=ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input)


train_images=train_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TRAIN"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=True,seed=20230527,subset='training', directory=path+"/train")

val_images=train_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TRAIN"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=True,seed=20230527,subset='validation', directory=path+"/train")

test_images = test_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TEST"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=False, directory=path+"/test")



inputs = tf.keras.layers.Input((150,150,3))
base_model=tf.keras.applications.EfficientNetV2S(include_top=False, weights="imagenet",input_shape=(150,150,3), pooling='avg') 
x=base_model(inputs)
output=layers.Dense(2, activation='sigmoid')(x)
model=tf.keras.models.Model(inputs=inputs, outputs=output)
model.compile(Adamax(learning_rate=1e-4), loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(train_images, validation_data=val_images, epochs=100)
model.save("./model/EfficientNetV2S.h5")

model = keras.models.load_model("./model/EfficientNetV2S.h5")
show_history(history)
plot_history(history, path="./img/Training_history EfficientNetV2S.png",
             title="Training history EfficientNetV2S")
plt.close()


test_images.reset()
preds=model.predict(test_images)
preds = np.argmax(preds,axis=1)
preds=preds>0.5
gt=[0 if x=="Normal" else 1 for x in df_meta[df_meta["Dataset_type"]=="TEST"]["Label"]]
print("Results on test set:")
print('---------------------------------------------------------')
print('Accuracy:'+str(accuracy_score(gt,preds)))
print('---------------------------------------------------------')
print('F1-Score:'+str(f1_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print('Recall:'+str(recall_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print('Precision:'+str(precision_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print("ROC AUC score:",roc_auc_score(gt,preds))
print('---------------------------------------------------------')
print(classification_report(gt,preds,target_names=["Normal","Covid"]))
conf_matrix = confusion_matrix(gt, preds)
sns.heatmap(conf_matrix,xticklabels = ["Normal","Covid"], yticklabels =["Normal","Covid"],annot=True,fmt='g')
plt.title('EfficientNetV2S Confusion Matrix')
plt.savefig("./img/Confusion Matrix EfficientNetV2S.png")
plt.show()

test_images.reset()
test_images.labels
preds=model.predict(test_images)
preds = np.argmax(preds,axis=1)
preds=preds>0.5
labels = (test_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in preds]
real=[labels[k] for k in test_images.labels]
filenames=test_images.filenames
path_test="./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/"
r=random.randint(0,len(os.listdir(path_test)))
img=cv2.imread(path_test+filenames[r])
plt.imshow(img)
plt.title("EfficientNetV2S Results:Predicton:{0} | Real:{1}".format(predictions[r],real[r]))
plt.savefig("./img/EfficientNetV2S Results EX.png")
results=pd.DataFrame({"Filename":filenames,"real":real,"Predictions":predictions})
results.to_csv("./result/EfficientNetV2S Results.csv",index=False)



#----------------------------------EfficientNetV2M------------------------------------------
BATCH_SIZE=16
path="./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/"


train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input,zoom_range=0.1,brightness_range=[0.5,1.3],
                                   width_shift_range=0.1,height_shift_range=0.1,validation_split=0.1)
test_datagen=ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input)


train_images=train_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TRAIN"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=True,seed=1234,subset='training', directory=path+"/train")

val_images=train_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TRAIN"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=True,seed=1234,subset='validation', directory=path+"/train")

test_images = test_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TEST"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=False, directory=path+"/test")



inputs = tf.keras.layers.Input((150,150,3))
base_model=tf.keras.applications.EfficientNetV2M(include_top=False, weights="imagenet",input_shape=(150,150,3), pooling='avg') 
x=base_model(inputs)
output=layers.Dense(2, activation='sigmoid')(x)
model=tf.keras.models.Model(inputs=inputs, outputs=output)
model.compile(Adamax(learning_rate=1e-4), loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(train_images, validation_data=val_images, epochs=100)
model.save("./model/EfficientNetV2M.h5")

model = keras.models.load_model("./model/EfficientNetV2M.h5")
show_history(history)
plot_history(history, path="./img/Training_history EfficientNetV2M.png",
             title="Training history EfficientNetV2M")
plt.close()

test_images.reset()
preds=model.predict(test_images)
preds = np.argmax(preds,axis=1)
preds=preds>0.5
gt=[0 if x=="Normal" else 1 for x in df_meta[df_meta["Dataset_type"]=="TEST"]["Label"]]
print("Results on test set:")
print('---------------------------------------------------------')
print('Accuracy:'+str(accuracy_score(gt,preds)))
print('---------------------------------------------------------')
print('F1-Score:'+str(f1_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print('Recall:'+str(recall_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print('Precision:'+str(precision_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print("ROC AUC score:",roc_auc_score(gt,preds))
print('---------------------------------------------------------')
print(classification_report(gt,preds,target_names=["Normal","Covid"]))
conf_matrix = confusion_matrix(gt, preds)
sns.heatmap(conf_matrix,xticklabels = ["Normal","Covid"], yticklabels =["Normal","Covid"],annot=True,fmt='g')
plt.title('EfficientNetV2M Confusion Matrix')
plt.savefig("./img/Confusion Matrix EfficientNetV2M.png")
plt.show()

test_images.reset()
test_images.labels
preds=model.predict(test_images)
preds = np.argmax(preds,axis=1)
preds=preds>0.5
labels = (test_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in preds]
real=[labels[k] for k in test_images.labels]
filenames=test_images.filenames
path_test="./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/"
r=random.randint(0,len(os.listdir(path_test)))
img=cv2.imread(path_test+filenames[r])
plt.imshow(img)
plt.title("EfficientNetV2M Results:Predicton:{0} | Real:{1}".format(predictions[r],real[r]))
plt.savefig("./img/EfficientNetV2M Results EX.png")
results=pd.DataFrame({"Filename":filenames,"real":real,"Predictions":predictions})
results.to_csv("./result/EfficientNetV2M Results.csv",index=False)

#----------------self-define---------------------



train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input,zoom_range=0.1,brightness_range=[0.5,1.3],
                                   width_shift_range=0.1,height_shift_range=0.1,validation_split=0.1)
test_datagen=ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)


BATCH_SIZE=64
path="./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/"
train_images=train_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TRAIN"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=True,seed=1234,subset='training', directory=path+"/train")

val_images=train_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TRAIN"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=True,seed=1234,subset='validation', directory=path+"/train")

test_images = test_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TEST"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=False, directory=path+"/test")
input_shape = (150,150, 3)



# 定義ResNet的基本塊
class BasicBlock(tf.keras.layers.Layer):
    expansion = 4

    def __init__(self, filters, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2D(filters, kernel_size=3, strides=stride, padding='same')
        self.bn1 = BatchNormalization()
        self.relu = ReLU()
        self.conv2 = Conv2D(filters, kernel_size=3, strides=1, padding='same')
        self.bn2 = BatchNormalization()

        self.shortcut = tf.keras.Sequential()
        if stride != 1:
            self.shortcut = tf.keras.Sequential([
                Conv2D(filters, kernel_size=1, strides=stride),
                BatchNormalization()
            ])

    def call(self, inputs):
        residual = inputs

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        shortcut = self.shortcut(inputs)

        x = Add()([x, shortcut])
        x = self.relu(x)

        return x

# 定義ResNet模型
class ResNet(tf.keras.Model):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()

        self.in_channels = 64

        self.conv1 = Conv2D(64, kernel_size=5, strides=2, padding='same')
        self.bn1 = BatchNormalization()
        self.relu = ReLU()
        self.maxpool = MaxPool2D(pool_size=3, strides=2, padding='same')

        self.layer1 = self.build_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.build_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.build_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.build_layer(block, 512, layers[3], stride=2)

        self.avgpool = GlobalAveragePooling2D()
        self.fc = Dense(num_classes, activation='sigmoid')

    def build_layer(self, block, filters, blocks, stride=1):
        layers = [block(filters, stride)]
        for _ in range(1, blocks):
            layers.append(block(filters, stride=1))
        return tf.keras.Sequential(layers)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x
    # def get_config(self):
    #     config = super(ResNet, self).get_config()
    #     # Add 'block' and 'layers' to the config dictionary
    #     config['block'] = self.block
    #     config['layers'] = self.layers
    #     return config
    # @classmethod
    # def from_config(cls, config):
    #     return cls(block=config['block'], layers=config['layers'])

# 建立ResNet-18模型
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# 建立ResNet-34模型
def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])
# 建立ResNet-101模型
def ResNet101():
    return ResNet(BasicBlock, [6, 4, 26, 3])

# 使用ResNet-18進行測試
base_model = ResNet34()
inputs = tf.keras.Input(shape=input_shape)
outputs =base_model(inputs)


model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()


model.compile(Adam(learning_rate=1e-4), loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

filepath="./logs/Weights-Epoch-{epoch:02d}-VA-{val_accuracy:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,mode='auto',save_best_only=True)
history = model.fit(train_images, validation_data=val_images, epochs=100,callbacks=[checkpoint])
model.save("./model/self-define.h5")

#model = keras.models.load_model("./model/VGG16.h5")

show_history(history)
plot_history(history, path="./img/Training_history self-define.png",
             title="Training history self-define")
plt.close()
model = keras.models.load_model("./model/Weights-Epoch-31-VA-0.9962.h5", custom_objects={'ResNet': lambda: ResNet(block=BasicBlock, layers=[3, 4, 6, 3])})

preds=model.predict(test_images)
preds = np.argmax(preds,axis=1)
preds=preds>0.5
gt=[0 if x=="Normal" else 1 for x in df_meta[df_meta["Dataset_type"]=="TEST"]["Label"]]
print("Results on test set:")
print('---------------------------------------------------------')
print('Accuracy:'+str(accuracy_score(gt,preds)))
print('---------------------------------------------------------')
print('F1-Score:'+str(f1_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print('Recall:'+str(recall_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print('Precision:'+str(precision_score(gt,preds,average='weighted')))
print('---------------------------------------------------------')
print("ROC AUC score:",roc_auc_score(gt,preds))
print('---------------------------------------------------------')
print(classification_report(gt,preds,target_names=["Normal","Covid"]))


conf_matrix = confusion_matrix(gt, preds)
sns.heatmap(conf_matrix,xticklabels = ["Normal","Covid"], yticklabels =["Normal","Covid"],annot=True,fmt='g')
plt.title('Self-Define Confusion Matrix')
plt.savefig("./img/Confusion Matrix Self-Define.png")
plt.show()


test_images.labels
preds=model.predict(test_images)
preds = np.argmax(preds,axis=1)
preds=preds>0.5
labels = (test_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in preds]
real=[labels[k] for k in test_images.labels]
filenames=test_images.filenames
path_test="./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/"
r=random.randint(0,len(os.listdir(path_test)))
img=cv2.imread(path_test+filenames[r])
plt.imshow(img)
plt.title("Self-Define Results:Predicton:{0} | Real:{1}".format(predictions[r],real[r]))
plt.savefig("./img/Self-Define Results EX.png")
results=pd.DataFrame({"Filename":filenames,"real":real,"Predictions":predictions})
results.to_csv("./result/Self-Define-Results.csv",index=False)




