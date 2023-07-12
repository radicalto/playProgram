import os
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.layers.core import Flatten
from keras.layers.core import Dense

files_train = 0
files_validation = 0

cwd = os.getcwd() # os.getcwd() 方法用于返回当前工作目录。
print(cwd)
folder = '../../data/train_data/train'
train_folder_all = os.listdir(folder)
print('folder: ',train_folder_all)

for sub_folder in os.listdir(folder):
    path, dirs, files = next(os.walk(os.path.join(folder,sub_folder)))
    # os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下。
    ''' path 表示当前正在访问的文件夹路径
        dirs 表示该文件夹下的子目录名list
        files 表示该文件夹下的文件list'''
    print('path: ',path)
    print('dirs: ',dirs)
    print('files: ',files)
    files_train += len(files)
    print('files_train',files_train)

img_width, img_height = 48, 48
train_data_dir = "../../data/train_data/train"
validation_data_dir = "../../data/train_data/test"

nb_train_samples = files_train
nb_validation_samples = files_validation
batch_size = 32
epochs = 15
num_classes = 2
model = applications.VGG16(weights='imagenet', include_top=False, input_shape = (img_width, img_height, 3))
# keras application 中已经训练的模型

for layer in model.layers[:10]:
    layer.trainable = False

x = model.output
x = Flatten()(x)
predictions = Dense(num_classes, activation="softmax")(x)

model_final = Model(input = model.input, output = predictions)

# 编译
model_final.compile(loss="categorical_crossentropy",
                    optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                    metrics=["accuracy"])
# ImageDataGenerator通过实时数据增强生成张量图像数据的epoch
train_datagen = ImageDataGenerator(
rescale=1./255,
horizontal_flip=True,
fill_mode="nearest",
zoom_range=0.1,
width_shift_range=0.1,
height_shift_range=0.1,
rotation_range=5)

test_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.1,
width_shift_range = 0.1,
height_shift_range=0.1,
rotation_range=5)
train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size,
class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width),
class_mode = "categorical")

checkpoint = ModelCheckpoint("../../data/car1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

history_object = model_final.fit_generator(
train_generator,
samples_per_epoch = nb_train_samples,
epochs = epochs,
validation_data = validation_generator,
nb_val_samples = nb_validation_samples,
callbacks = [checkpoint, early])

