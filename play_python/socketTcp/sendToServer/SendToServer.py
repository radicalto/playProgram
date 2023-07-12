import socket
import json
import time
from socketData import json_data
'''
nc localhost 44444  客户端
'''

# 创建一个套接字对象
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 连接服务器
s.connect(('127.0.0.1', 44444))

while True:
    # 构造要发送的JSON数据
    # data = {'name': 'John', 'age': 30, 'city': 'New York'}


    # json_data = json.dumps(data_str)

    # 发送JSON数据到服务端
    # s.sendall(json_data.encode())
    s.sendto(json_data.encode(), ('127.0.0.1', 44444))
    time.sleep(10)

# 关闭套接字连接
s.close()