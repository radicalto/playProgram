import socket

# 连接Flume服务器
import time

TCP_IP = '192.168.1.128'  # Flume 服务器地址
TCP_PORT = 44444  # Flume 服务器端口号
BUFFER_SIZE = 1024
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((TCP_IP, TCP_PORT))
while True:
    # 构造需要发送到Flume服务器的数据
    data = "{'name': 'John', 'age': 30, 'city': 'New York'}"
    data_bytes = data.encode('utf-8')

    # 将数据发送到Flume服务器
    client_socket.send(data_bytes)
    time.sleep(5)

# 关闭连接
client_socket.close()