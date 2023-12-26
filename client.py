import socket
import io
import time

import cv2
import numpy as np
from PIL import Image

# 创建Socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 服务端IP和端口
server_ip = '127.0.0.1'
server_port = 11111

# 绑定IP和端口
server_socket.bind((server_ip, server_port))

# 监听
server_socket.listen(1)

print(f"Server listening on {server_ip}:{server_port}")

# 接受连接
client_socket, client_address = server_socket.accept()

print(f"Connection from {client_address} has been established.")

# 循环接收多个图像
while True:
    t1 = time.time()

    # 接收图像大小
    size_data = client_socket.recv(4)  # 假设图像大小以4字节整数形式发送
    if not size_data:
        break
    image_size = int.from_bytes(size_data, byteorder='little')

    print("size", image_size)
    # 接收图像数据
    image_data = b''
    while len(image_data) < image_size:
        UAV1_packet = client_socket.recv(image_size - len(image_data))
        if not UAV1_packet:
            break
        image_data += UAV1_packet

    # 接收图像大小
    size_data2 = client_socket.recv(4)  # 假设图像大小以4字节整数形式发送
    if not size_data2:
        break
    image_size2 = int.from_bytes(size_data2, byteorder='little')

    # 接收图像数据
    image_data2 = b''
    while len(image_data2) < image_size2:
        UAV2_packet = client_socket.recv(image_size2 - len(image_data2))
        if not UAV2_packet:
            break
        image_data2 += UAV2_packet

    t2 = time.time()

    print("FPS-1  ", 1/(t2 - t1))

    # 将接收到的数据转换为numpy数组
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    # 从numpy数组中恢复图像
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # 显示图像
    cv2.imshow("Received Image", image)

    print("image-shape", image.shape)
    # 将接收到的数据转换为numpy数组
    image_array2 = np.frombuffer(image_data2, dtype=np.uint8)
    # 从numpy数组中恢复图像
    image2 = cv2.imdecode(image_array2, cv2.IMREAD_COLOR)
    # 显示图像
    cv2.imshow("Received Image--2", image2)

    t3 = time.time()

    print("FPS-2  ", 1/(t3 - t1))

    cv2.waitKey(1)





# # 将接收到的图像数据保存为图片
# image = Image.open(io.BytesIO(image_data))
# image.save("received_image.jpg")  # 保存接收到的图像


# 关闭Socket
client_socket.close()
server_socket.close()