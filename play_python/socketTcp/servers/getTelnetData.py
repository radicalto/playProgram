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

    recvData,addr = conn.recvfrom(1024)
    # strData = recvData.decode()
    # jsonData = json.loads(strData)
    # print(jsonData,type(jsonData))
    print(recvData)
# 关闭套接字连接
conn.close()
s.close()
