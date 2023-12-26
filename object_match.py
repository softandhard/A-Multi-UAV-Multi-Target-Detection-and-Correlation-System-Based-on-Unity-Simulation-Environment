import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams, letterbox
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, \
    check_imshow, xyxy2xywh, save_one_box
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.plots import colors, plot_one_box
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

import socket
import time
import numpy as np
from deep_sort_pytorch.deep_sort.sort import feature_matching

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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

print(f"IP地址 {client_address} 的连接已经建立.")

#源码：https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        print("id==", id)
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def object_detect(input_image, params, model, device, half):
    # Padded resize
    img = letterbox(input_image, new_shape=640, stride=32)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    start = time_synchronized()
    pred = model(img, augment=params.augment)[0]
    end = time_synchronized()
    print("detection_FPS", 1 / (end - start))

    # Apply NMS
    pred = non_max_suppression(
        pred, params.conf_thres, params.iou_thres, classes=params.classes, agnostic=params.agnostic_nms)
    t2 = time_synchronized()

    return pred, img


def post_process(pred_result, image, img, names, view_img, save_txt, txt_path, opt):
    # Process detections
    for i, det in enumerate(pred_result):  # detections per image

        info, im0 = '', image

        info += '%gx%g ' % img.shape[2:]  # print string

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], im0.shape).round()

            # 打印结果
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                info += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # 显示检测框
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                        -1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if opt.hide_labels else (
                        names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                    # 画出目标框，显示类别
                    plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                                 line_thickness=opt.line_thickness)


def object_compare(drone1_pred, drone2_pred, drone1_image, drone2_image, img, img2, names, deepsort, distance_metric, start_time):
    drone1_all_features = np.array([])
    drone2_all_features = np.array([])
    for i, det in enumerate(drone1_pred):
        info, im0 = '', drone1_image
        info += '%gx%g ' % img.shape[2:]  # print string
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                # 检测到的每个类别的数量
                n = (det[:, -1] == c).sum()
                # 加入到显示信息中
                info += '%g %ss, ' % (n, names[int(c)])

            xywh_bboxs = []
            confs = []
            drone1_class = []
            drone1_boxes = []
            drone1_conf = []
            # 转换为deepsort的处理格式
            for *xyxy, conf, cls in det:
                # to deep sort format
                print("class==", cls)
                x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                print("xywh==", xywh_obj)
                xywh_bboxs.append(xywh_obj)
                confs.append([conf.item()])
                drone1_class.append(cls)
                drone1_boxes.append(xywh_obj)
                drone1_conf.append([conf.item()])
            print("box-num", len(xywh_bboxs))
            xywhs = torch.Tensor(xywh_bboxs)
            confss = torch.Tensor(confs)

            # 提取每一个相机中的所有目标框特征
            drone1_all_features = deepsort._get_features(xywhs, im0)
            print("features-len", len(drone1_all_features))
            # 过滤置信度低的检测框
            detections = [drone1_all_features[i] for i, conf in enumerate(
                confss) if conf > 0.3]
            print("detect-len==", len(detections))
            print('%sDone' % info)

    for i, det in enumerate(drone2_pred):
        info2, im0 = '', drone2_image
        info2 += '%gx%g ' % img2.shape[2:]  # print string
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                info2 += '%g %ss, ' % (n, names[int(c)])  # add to string

            xywh_bboxs = []
            confs = []

            # Adapt detections to deep sort input format
            for *xyxy, conf, cls in det:
                # to deep sort format
                x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                xywh_bboxs.append(xywh_obj)
                confs.append([conf.item()])
            print("box-num-2", len(xywh_bboxs))
            xywhs = torch.Tensor(xywh_bboxs)
            confss = torch.Tensor(confs)

            # 提取每一个相机中的所有目标框特征
            drone2_all_features = deepsort._get_features(xywhs, im0)
            print("features-2-len", len(drone2_all_features))
            # 过滤置信度低的检测框
            detections = [drone2_all_features[i] for i, conf in enumerate(
                confss) if conf > 0.3]
            print("detect-2-len==", len(detections))
            print('%sDone' % info2)

    # 当检测到目标时
    if len(drone1_all_features) != 0 and len(drone2_all_features) != 0:
        # 计算不同相机所检测到的目标相似度
        cost_matrix = distance_metric.multi_objects_distance \
            (features=drone1_all_features, targets=drone2_all_features)
        print("相似度 ", (1-cost_matrix) * 100)
        print("相似度-len ", cost_matrix.shape)
    t3 = time_synchronized()
    # Print time
    print('每帧处理时间：(%.3fs)' % (t3 - start_time))
    fps = 1.0 / (t3 - start_time)
    print('处理帧率：(%.3fs)' % fps)
    print(" ")

def result_save(drone1_img, drone2_img, show_vid, save_vid, vid_path, save_path, out, vid_writer):
    # Stream results
    if show_vid:
        cv2.imshow("drone1", drone1_img)
        cv2.imshow("drone2", drone2_img)
        if cv2.waitKey(1) == ord(' '):  # space to quit
            raise StopIteration

    # Save results (image with detections)
    if save_vid:
        if vid_path != save_path:  # new video
            vid_path = save_path
            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.release()  # release previous video writer

            fps, w, h = 30, drone1_img.shape[1], drone1_img.shape[0]
            save_path = str(Path(out) / "drone1")
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        vid_writer.write(drone1_img)

        if vid_path != save_path:  # new video
            vid_path = save_path
            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.release()  # release previous video writer

            fps, w, h = 30, drone2_img.shape[1], drone2_img.shape[0]
            save_path = str(Path(out) / "drone2")
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        vid_writer.write(drone2_img)

def detect(opt):
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.img_size, opt.evaluate
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    # 如果权重不存在，尝试下载文件
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    # cfg.DEEPSORT.REID_CKPT == "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
    # cfg.DEEPSORT.REID_CKPT == opt.deep_sort_weights
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
    distance_metric = feature_matching.NearestNeighborDistanceMetric("cosine", matching_threshold=0.2, budget=100)
    # Initialize
    device = select_device(opt.device)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # load FP32 model
    model = attempt_load(yolo_weights, map_location=device)
    # model stride
    stride = int(model.stride.max())
    # check img_size
    imgsz = check_img_size(imgsz, s=stride)
    # get class names
    names = model.module.names if hasattr(model, 'module') else model.names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # 检测环境是否支持显示画面
    if show_vid:
        show_vid = check_imshow()
    view_img = check_imshow()

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'

    while True:
        start_time = time.time()

        img_t1 = time.time()

        # 接收图像大小
        size_data = client_socket.recv(4)  # 假设图像大小以4字节整数形式发送
        if not size_data:
            break
        image_size = int.from_bytes(size_data, byteorder='little')

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

        img_t3 = time.time()

        print("unity-FPS==", 1 / (img_t3 - img_t1))

        # 当接收到的图像非空时进行检测
        if np.all(image) != None and np.all(image2) != None:
            # 对图像进行目标检测和后处理
            drone1_pred, img = object_detect(image, opt, model, device, half)
            post_process(drone1_pred, image, img, names, view_img, save_txt, txt_path, opt)

            drone2_pred, img2 = object_detect(image2, opt, model, device, half)
            post_process(drone2_pred, image2, img2, names, view_img, save_txt, txt_path, opt)
            # 目标特征比较
            object_compare(drone1_pred, drone2_pred, image, image2, img, img2, names, deepsort, distance_metric, start_time)
            # 保存图像结果
            result_save(image, image2, show_vid, save_vid, vid_path, save_path, out, vid_writer)



# cpu处理的速度远低于GPU
# cpu：6-7 fps  gpu：100 fps上下 40-50
#  [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'bounding_box_train', 'truck', 'boat', 'traffic light',
#          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
#          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
#          'hair drier', 'toothbrush' ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='C:/Users/cjh/Desktop/Yolov5_DeepSort_Pytorch/unity_dataset', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--show-vid', default=True, action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', default=[0, 1, 2], nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)
