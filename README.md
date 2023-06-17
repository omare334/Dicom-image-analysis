# Dicom-image-analysis
Here I get 20,000 Dicom images that are 3d pre process them and run them throguh the CNN that deals with dicom videos in addition to comparing CNN with other methods. There is a presentation
in this repo that can show you some of the results including some of the code used is in the repo. 
# The Preprocessing
Multiple methods have been used here. The preprocessing where truncation , normalisation and some other methods to deal with the contrast of the image . As this is health
data the preprocessing has to make sense . So the idead behind the normalisation and using the apache mask is to exenuate the striation within th breast. This helps me
highlight the tumour and the striation more you can see more details in the presentation in the repo I may add the photos here to later.
# The machine learning methods
The task here was to use transfer learning for the multiclassification problem , xgboost and also a completley unsupervised model by using feature selection from 
the cnn model and clustering ussing kmeans and hdbscan. 
# CNN 
While it is a tight race , the best model is the cnn model using pretrained layers from the VGG-16 which is trained on imagenet. The code for the the model can be seen 
here : 
```python
#importing packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16
#checking if gpu is avalaible
if tf.test.is_gpu_available():
    print('GPU available:')
    for gpu in tf.config.list_physical_devices('GPU'):
        print(gpu)
    tf.config.set_visible_devices(gpu, 'GPU')
else:
    print('No GPU available.')
#loading in dicom and labels
dicom_array = df
labels = one_hot_encoded_list
dicom_array = np.repeat(dicom_array, 3, axis=-1)

labels = np.asarray(labels)
labels = labels.astype(np.int32)
num_classes = np.max(labels) + 1
labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    dicom_array, labels_one_hot, test_size=0.2, random_state=42, stratify=labels
)

input_shape = (5, 280, 280, 3)

# Load the VGG-16 model
base_model = VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=input_shape[1:]
)

for layer in base_model.layers:
    layer.trainable = True
#creating model
input_layer = layers.Input(shape=input_shape, name="input")
x = layers.TimeDistributed(base_model)(input_layer)
x = layers.Conv3D(64, kernel_size=(1, 3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
x = layers.BatchNormalization()(x)
x = layers.GlobalAveragePooling3D()(x)
x = layers.Dropout(0.3)(x)
output_layer = layers.Dense(num_classes, activation='softmax', name="output")(x)

model = keras.Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=10, batch_size=8, validation_split=0.2)
```
in this exmaple VGG16 has been used with a time distibuted layer as these are dicom videos that add depth they are not neccessarily 3d dicom images the other method we garnered
more success is just treating each seperate slide as a 2d image separatley to inflate the size of the data as I did not have a lot of cancer data , however the improvement may have been 
exactrebated as this can happen with a larger dataset. Ideally with unilimited resources I would run the time distibuted model on the whole dataset in which with more slices taken rather 
than just 5 to improve the statistical power and therfore the results.
# XGboost and unsupervised model
Both rely on feature selection from the CNN Model and do not really add anything too useful. The silhoutte score is still quite low from the unsupervised clustering
so idenitfying suitable phenogroups is hard . Also the xgboost was not as good as the CNN.
#limitations 
Due to the nature of the 2d and 3d feature extraction the number of features extracted for both models are different ideally they would be the same 
so we get a better idea of what seprates the data more effeciently. I would also like to attempt to use the whole dataset to create a more fair comparison between the 3d video based 
images and 2d images.

