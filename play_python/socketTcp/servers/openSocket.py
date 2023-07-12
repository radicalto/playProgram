import socket
import json
import time

# 创建一个套接字对象
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定本地地址和端口
s.bind(('127.0.0.1', 44444))

# 开始监听
s.listen(1)

# 等待客户端连接
conn, addr = s.accept()

while True:
    # 构造要发送的JSON数据
    data = {'name': 'John', 'age': 30, 'city': 'New York'}
    json_data = json.dumps(data)

    # 发送JSON数据到客户端
    conn.sendall(json_data.encode())

    # 休眠5秒钟
    time.sleep(5)

# 关闭套接字连接
conn.close()
s.close()
