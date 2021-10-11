from __future__ import print_function
from itertools import cycle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import cv2
import os
import time
import sys
import json

# colors for terminal
CRED = '\033[91m'
CREDEND = '\033[0m'

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", type=str, help="path and name to output model")
ap.add_argument("-c", "--configFile", type=str, help="path and name to the configuration file")
args = vars(ap.parse_args())

params = args["configFile"]
params = json.loads(open(params).read())

# initialize the initial learning rate, number of epochs to train for, and batch size
INIT_LR = 1e-3
EPOCHS = params["epochs"]
BS = params["batch_size"]

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
start = time.time()
print(CRED + "[INFO] loading images..." + CREDEND)
image_paths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# loop over the image paths
for image_path in image_paths:
    # extract the class label from the filename
    label = image_path.split(os.path.sep)[-2].split(" ")
    # load the image, swap color channels, and resize it to be a fixed 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 255]
data = np.array(data) / 255.0
labels = np.array(labels)

if params["number_of_classes"] == 2:
    # perform one-hot encoding on the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
else:
    lb = MultiLabelBinarizer()
    labels = lb.fit_transform(labels)

# partition the data into training and testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=params["test_size"], random_state=42)

# initialize the training data augmentation object
trainAug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                              height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                              horizontal_flip=True, fill_mode="nearest")


# load the DL network, ensuring the head fully connected layer sets are left
baseModel = eval(params["dl_network"])(weights="imagenet", include_top=False,
                             input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation=params["activation_function"])(headModel)
headModel = Dropout(params["dropout_keep_prob"])(headModel)
headModel = Dense(params["number_of_classes"], activation=params["activation_function_output"])(headModel)

# place the head FC model on top of the base model (this will become the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will not be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

# compile the model
print(CRED +"[INFO] compiling model..."+CREDEND)
opt = eval(params["optimizer"])(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss=params["loss_function"], optimizer=opt,
              metrics=["accuracy"])

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# train the head of the network
print(CRED +"[INFO] training head..."+CREDEND)
dLOOKmodeler = model.fit_generator(
    trainAug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS,
    callbacks=[early_stopping_callback])

# make predictions on the testing set
print(CRED +"\n [INFO] evaluating network..."+CREDEND)
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print("\n\n" + classification_report(testY.argmax(axis=1), predIdxs) + "\n\n")

# compute the confusion matrix and and use it to derive the raw accuracy, sensitivity, and specificity
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
true_positive = np.diag(cm)
false_positive = np.sum(cm, axis=0) - true_positive
false_negative = np.sum(cm, axis=1) - true_positive
true_negative = cm.sum() - (false_positive + false_negative + true_positive)

print(CRED +"[INFO] More results...\n"+CREDEND)
print(CRED +"The following metrics represents the results per class, i.e.: "+CREDEND,list(lb.classes_), "\n")

print(CRED +"Confusion Matrix\n"+CREDEND, cm, "\n")
print(CRED +"True positive for each class: "+CREDEND, true_positive)
print(CRED +"False positive for each class: "+CREDEND, false_positive)
print(CRED +"False negative for each class: "+CREDEND, false_negative)
print(CRED +"True negative for each class: "+CREDEND, true_negative)
TPR = true_positive.astype(float) / (true_positive.astype(float) + false_negative.astype(float))
print(CRED +"\nSensitivity for each class: "+CREDEND, TPR)
TNR = true_negative.astype(float) / (true_negative.astype(float) + false_positive.astype(float))
print(CRED +"Specificity for each class: "+CREDEND, TNR)


acc_NN = accuracy_score(testY.argmax(axis=1), predIdxs)
print(CRED +'\nOverall accuracy of Neural Network model: '+CREDEND, acc_NN,"\n")


# ----------------- PLOTS - Confusion Matrix, ROC curves, Training/Validation Loss Accuracy -----------------

# Confusion Matrix
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
labels = list(lb.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True, fmt='d', cmap="Blues", vmin=0.2);
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.savefig("Confusion-Matrix " + params["dl_network"] + ".pdf")
print(CRED +"[INFO] Confusion Matrix plot saved..."+CREDEND)

# ROC curves
y_score_new = model.predict(testX, batch_size=BS)
n_classes = params["number_of_classes"]
lw = 2
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(testY[:, i], y_score_new[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(testY.ravel(), y_score_new.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='red', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['grey', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig("ROC-curves " + params["dl_network"] + ".pdf")
print(CRED +"[INFO] ROC-curves plot saved..."+CREDEND)

# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='red', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['grey', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.savefig("ROC-curves-zoomed" + params["dl_network"] + ".pdf")
print(CRED +"[INFO] ROC-curves zoomed plot saved..."+CREDEND)

# Training/Validation Loss Accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), dLOOKmodeler.history["loss"], label="Training loss")
plt.plot(np.arange(0, N), dLOOKmodeler.history["val_loss"], label="Validation loss")
plt.plot(np.arange(0, N), dLOOKmodeler.history["accuracy"], label="Training accuracy")
plt.plot(np.arange(0, N), dLOOKmodeler.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("ACC-LOSS " + params["dl_network"] + ".pdf")
print(CRED +"[INFO] ACC-LOSS plot saved..."+CREDEND)


# serialize the model to disk
print(CRED +"[INFO] saving model..."+CREDEND)
model.save(args["model"], save_format="h5")
end = time.time()
print(CRED +"Total execution Time (sec):"+CREDEND, end - start)
