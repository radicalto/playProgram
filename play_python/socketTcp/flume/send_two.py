import kafka
import time
import socket
from socketData import data_str
# 连接 Kafka Producer
producer = kafka.KafkaProducer(bootstrap_servers=['localhost:9092'])

# 连接 Flume 服务器
TCP_IP = 'localhost'
TCP_PORT = 44444
BUFFER_SIZE = 1024
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((TCP_IP, TCP_PORT))

# 发送数据到 Flume 服务器并写入 Kafka
for i in range(0,2):
    data = data_str+"第"+i+"遍"
    data_bytes = data.encode('utf-8')

    # 发送数据到 Flume 服务器
    client_socket.send(data_bytes)

    # 将数据写入 Kafka Topic
    producer.send('getSocket', value=data_bytes)

    # 暂停 5 秒
    time.sleep(2)

# 关闭连接
client_socket.close()
producer.close()