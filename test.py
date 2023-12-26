# import socket
# import io
#
# import cv2
# import numpy as np
# from PIL import Image
#
# server_ip = '127.0.0.1'  # Listen on all interfaces
# server_port = 11111
#
# # 创建TCP服务器的socket
# server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_socket.bind((server_ip, server_port))
# server_socket.listen()
#
# print(f"Server listening on {server_ip}:{server_port}")
#
# client_socket, address = server_socket.accept()
# print(f"Connection from {address} has been established.")
#
# # 读取数据
# image_data = bytearray()
# try:
#     # 在实际应用中，需要有逻辑确定何时停止接收数据
#     # 例如，可以先发送数据长度，或者在发送完毕后关闭连接
#
#     while True:
#         # 首先接收图像数据的大小
#         size_data = client_socket.recv(4)  # 假设图像大小以4字节整数形式发送
#         image_size = int.from_bytes(size_data, byteorder='little')
#
#         print("image_data_size", image_size)
#
#         while len(image_data) < image_size:
#             packet = client_socket.recv(8)
#             if not packet:
#                 break
#             image_data.extend(packet)
#         #把接收到的字节数据转换为图像
#         image = Image.open(io.BytesIO(image_data))
#         image.save('received_image.png')  # 保存图像
#
#         # 将接收到的数据转换为numpy数组
#         image_array = np.frombuffer(image_data, dtype=np.uint8)
#
#         # 从numpy数组中恢复图像
#         image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#         # 显示图像
#         cv2.imshow("Received Image", image)
#         cv2.waitKey(1)
#
# finally:
#     client_socket.close()
#     server_socket.close()
#
# # # 使用Pillow从字节数据创建图像对象
# # image = Image.open(io.BytesIO(image_data))
# # # 图像现在在内存中，可以进行进一步的处理
# # # 这里只是演示在屏幕上显示它
# # image.show()


import torch
import torch.nn as nn

# 创建一个输入张量，维度为（batch_size, channels, depth, height, width）
input_tensor = torch.randn(2, 3, 32, 32, 32)
input_images = torch.randn(2, 3, 6, 480, 640)

# 创建一个3D卷积层，输入通道数为3，输出通道数为6，卷积核大小为3x3x3
conv3d_layer = nn.Conv3d(in_channels=3, out_channels=6, kernel_size=3)

# 将输入张量传递给卷积层进行前向计算
output = conv3d_layer(input_images)

# 打印输出张量的大小
print(output.size())

# 创建一个大小为[1, 32, 413, 416, 4]的5维输入张量
x = torch.randn(1, 32, 413, 416, 4)

# 将输入张量转换为四维张量，大小为[1, 32*413, 416, 4]
x_reshaped = torch.reshape(x, (1, 32*413, 416, 4))

# 打印转换后的张量大小
print(x_reshaped.size())


import numpy as np

# 假设img1和img2分别是两个图像的灰度向量表示
img1 = np.array([0.1, 0.2, 0.3, 0.4])  # 示例向量，实际情况下需要根据图像进行转换
img2 = np.array([0.1, 0.1, 0.9, 0.9])  # 示例向量，实际情况下需要根据图像进行转换

# 计算余弦相似度
similarity = np.dot(img1, img2) / (np.linalg.norm(img1) * np.linalg.norm(img2))
print('余弦相似度：', similarity)
