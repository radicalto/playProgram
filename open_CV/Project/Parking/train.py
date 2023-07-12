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

cwd = os.getcwd()
folder = '../../data/train_data/train'
for sub_folder in os.listdir(folder):
    path, dirs, files = next(os.walk(os.path.join(folder,sub_folder)))
    files_train += len(files)


folder = '../../data/train_data/test'
for sub_folder in os.listdir(folder):
    path, dirs, files = next(os.walk(os.path.join(folder,sub_folder)))
    files_validation += len(files)
# 访问系统中训练数据和测试数据，返回样本个数
print(files_train,files_validation)
img_width, img_height = 48, 48
train_data_dir = "../../data/train_data/train"
validation_data_dir = "../../data/train_data/test"

nb_train_samples = files_train
nb_validation_samples = files_validation
batch_size = 32
epochs = 15
num_classes = 2
model = applications.VGG16(weights='imagenet', include_top=False, input_shape = (img_width, img_height, 3))

for layer in model.layers[:10]:
    layer.trainable = False


x = model.output
x = Flatten()(x)
predictions = Dense(num_classes, activation="softmax")(x)

# model_final = Sequential([Dense(model.input, activation="selu")(x),Dense(num_classes, activation="softmax")(x)])

model_final = Model(inputs = model.input, outputs = predictions)

# 编译 ：定义损失函数、优化器和性能指标
model_final.compile(loss="categorical_crossentropy",
                    optimizer=optimizers.SGD(learning_rate=0.0001, momentum=0.9),
                    metrics=["accuracy"])


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
# 使用的模型
checkpoint = ModelCheckpoint("../../data/car.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# 训练数据
history_object = model_final.fit(
train_generator,
steps_per_epoch = nb_train_samples,
epochs = epochs,
validation_data = validation_generator,
validation_steps = nb_validation_samples,
callbacks = [checkpoint, early])
