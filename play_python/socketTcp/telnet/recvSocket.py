import socket
import json

'''
nc localhost 44444  客户端
'''

# 创建一个套接字对象
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 连接服务器
s.connect(('127.0.0.1', 44444))

while True:
    # 接收数据
    data = s.recv(1024)
    s.sendall()
    if not data:
        break

    # 解码JSON数据
    json_data = data.decode()
    obj = json.loads(json_data)

    # 输出JSON数据
    print(obj)

# 关闭套接字连接
s.close()