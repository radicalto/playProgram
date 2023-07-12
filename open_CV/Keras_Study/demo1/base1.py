import keras

import numpy as np
import matplotlib.pyplot as plt
# 产生100个随机点
from keras.layers import Dense

x_data = np.random.rand(100)
noise = np.random.normal(0,0.01,x_data.shape)# 噪声
y_data = x_data*0.1+0.2+noise
plt.scatter(x_data,y_data)
plt.show()

# 创建一个顺序模型
model = keras.models.Sequential()
# 在模型中添加一个全连接层
model.add(Dense(units=1,input_dim=1))
model.compile(loss='mse',
            optimizer='sgd')

# 训练3001个批次
for step in range(300):
    # 每次训练一个批次
    cost = model.train_on_batch(x_data,y_data)
    # 每500个batch 输出一次cost
    if step % 500 == 0:
        print('cost: ',cost)
# 打印权值和偏置值
w,b = model.layers[0].get_weights()
print('w: ',w,'b: ',b)
# x_data 输入到网络中，得到预测值y_pred
y_pred = model.predict(x_data)
# 显示随机点
plt.scatter(x_data,y_data)
# 显示预测结果
plt.plot(x_data,y_pred,'r-',lw=3)
plt.show()