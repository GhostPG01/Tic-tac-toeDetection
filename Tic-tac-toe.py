import os
import ujson
import aicube
from libs.PipeLine import ScopedTiming
from libs.Utils import *
from media.sensor import *
from media.display import *
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np
import image
import gc
from machine import UART, FPIOA, TOUCH
import time

# -----------------------
# UART & Display Setup
# -----------------------
fpioa = FPIOA()
fpioa.set_function(3, FPIOA.UART1_TXD)
fpioa.set_function(4, FPIOA.UART1_RXD)
uart = UART(UART.UART1, 115200)

# 使用 hdmi 模式
display_mode = "hdmi"
if display_mode == "lcd":
    DISPLAY_WIDTH = ALIGN_UP(800, 16)
    DISPLAY_HEIGHT = 480
else:
    DISPLAY_WIDTH = ALIGN_UP(1920, 16)
    DISPLAY_HEIGHT = 1080

OUT_RGB888P_WIDTH = ALIGN_UP(1280, 16)
OUT_RGB888P_HEIGHT = 720

# -----------------------
# File Paths
# -----------------------
root_path = "/sdcard/mp_deployment_source/"
config_path = root_path + "deploy_config.json"

# -----------------------
# 全局变量：保存上一帧检测到的棋盘区域
# -----------------------
prev_chessboard = None  # 格式：(x, y, w, h)

# -----------------------
# 辅助函数：计算图像填充参数
# -----------------------
def two_side_pad_param(input_size, output_size):
    ratio_w = output_size[0] / input_size[0]
    ratio_h = output_size[1] / input_size[1]
    ratio = min(ratio_w, ratio_h)
    new_w = int(ratio * input_size[0])
    new_h = int(ratio * input_size[1])
    dw = (output_size[0] - new_w) / 2
    dh = (output_size[1] - new_h) / 2
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    return top, bottom, left, right, ratio

# -----------------------
# 辅助函数：读取配置文件
# -----------------------
def read_deploy_config(path):
    try:
        with open(path, 'r') as f:
            return ujson.load(f)
    except Exception as e:
        print("JSON解析错误:", e)
        return None

# -----------------------
# 新数据包构造函数：只发送10个数据
# -----------------------
def make_packet(chessboard_box, pieces):
    """
    chessboard_box: (x, y, w, h) (显示坐标)；如果当前未检测到，则用上一帧值
    pieces: list of (cx, cy, color) ;其中color: 黑棋→1，白棋→2
    Data Packet (10 numbers):
       Data[0] = 0xE8 (校验位)
       Data[1:10]: 对应棋盘九宫格中每个格子的状态，
           如果该格子中心附近（距离<=50像素）检测到棋子，则发送该棋子的颜色（1或2），否则发送0。
    """
    packet = []
    packet.append(0xE8)
    if chessboard_box is None:
        # 没有棋盘则后面9个全部为0
        packet.extend([0]*9)
        return bytearray(packet)
    x, y, w, h = chessboard_box
    # 计算棋盘九宫格中心（行优先）
    cell_centers = []
    cell_w = w / 3.0
    cell_h = h / 3.0
    for row in range(3):
        for col in range(3):
            cx = int(x + col * cell_w + cell_w/2)
            cy = int(y + row * cell_h + cell_h/2)
            cell_centers.append((cx, cy))
    # 对每个 cell 查找最近的棋子
    cell_state = [0] * 9
    for i, cell in enumerate(cell_centers):
        best_dist = 1e9
        best_color = 0
        for p in pieces:
            dist = ((p[0]-cell[0])**2 + (p[1]-cell[1])**2)**0.5
            if dist < 50 and dist < best_dist:
                best_dist = dist
                best_color = p[2]
        if best_dist < 50:
            cell_state[i] = best_color
        else:
            cell_state[i] = 0
    packet.extend(cell_state)
    return bytearray(packet)

# -----------------------
# 主检测函数
# -----------------------
def detection():
    global prev_chessboard
    print("det_infer start")

    config = read_deploy_config(config_path)
    if config is None:
        print("配置加载失败")
        return -1

    # 配置要求类别列表为 ["chessboard", "black", "white", "green"]
    # 这里只关心["chessboard", "black", "white"]
    kmodel_name = config["kmodel_path"]
    labels = config["categories"]
    confidence_threshold = config["confidence_threshold"]
    nms_threshold = config["nms_threshold"]
    img_size = config["img_size"]
    num_classes = config["num_classes"]
    nms_option = config["nms_option"]
    model_type = config["model_type"]
    if model_type == "AnchorBaseDet":
        anchors = config["anchors"][0] + config["anchors"][1] + config["anchors"][2]
    else:
        anchors = None

    kmodel_frame_size = img_size
    frame_size = [OUT_RGB888P_WIDTH, OUT_RGB888P_HEIGHT]
    strides = [8, 16, 32]
    top, bottom, left, right, ratio = two_side_pad_param(frame_size, kmodel_frame_size)

    kpu = nn.kpu()
    # 修改模型加载路径
    kpu.load_kmodel(root_path + kmodel_name)
    ai2d = nn.ai2d()
    ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
    ai2d.set_pad_param(True, [0,0,0,0, top, bottom, left, right], 0, [114,114,114])
    ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
    ai2d_builder = ai2d.build([1,3,OUT_RGB888P_HEIGHT,OUT_RGB888P_WIDTH],
                               [1,3,kmodel_frame_size[1], kmodel_frame_size[0]])

    sensor = Sensor()
    sensor.reset()
    sensor.set_hmirror(True)
    sensor.set_vflip(True)
    sensor.set_framesize(width = DISPLAY_WIDTH, height = DISPLAY_HEIGHT)
    sensor.set_pixformat(PIXEL_FORMAT_YUV_SEMIPLANAR_420)
    sensor.set_framesize(width = OUT_RGB888P_WIDTH , height = OUT_RGB888P_HEIGHT, chn=CAM_CHN_ID_2)
    sensor.set_pixformat(PIXEL_FORMAT_RGB_888_PLANAR, chn=CAM_CHN_ID_2)
    sensor_bind_info = sensor.bind_info(x = 0, y = 0, chn = CAM_CHN_ID_0)

    sensor_bind_info = sensor.bind_info(x=0, y=0, chn=CAM_CHN_ID_0)
    Display.bind_layer(**sensor_bind_info, layer=Display.LAYER_VIDEO1)

    if display_mode == "hdmi":
        Display.init(Display.LT9611, to_ide=True)
    else:
        Display.init(Display.ST7701, to_ide=True)

    osd_img = image.Image(DISPLAY_WIDTH, DISPLAY_HEIGHT, image.ARGB8888)
    MediaManager.init()
    sensor.run()

    initial_data = np.ones((1,3,kmodel_frame_size[1], kmodel_frame_size[0]), dtype=np.uint8)
    ai2d_output_tensor = nn.from_numpy(initial_data)

    while True:
        chessboard_box = None
        pieces = []   # 格式: (cx, cy, color) 其中 color: black -> 1, white -> 2
        rgb_img = sensor.snapshot(chn=CAM_CHN_ID_2)
        try:
            if rgb_img.format() != image.RGBP888:
                continue
        except Exception as e:
            print("获取图像异常:", e)
            continue

        # 使用原始图像进行预处理
        ai2d_input = rgb_img.to_numpy_ref()
        ai2d_input_tensor = nn.from_numpy(ai2d_input)
        ai2d_builder.run(ai2d_input_tensor, ai2d_output_tensor)

        kpu.set_input_tensor(0, ai2d_output_tensor)
        kpu.run()
        results = []
        for i in range(kpu.outputs_size()):
            out_data = kpu.get_output_tensor(i)
            res_arr = out_data.to_numpy().reshape(-1)
            del out_data
            results.append(res_arr)

        det_boxes = aicube.anchorbasedet_post_process(
            results[0], results[1], results[2],
            kmodel_frame_size, frame_size, strides,
            num_classes, confidence_threshold, nms_threshold,
            anchors, nms_option
        )

        # 如果返回为平铺数组，则按每6个数字分组
        if det_boxes:
            if not isinstance(det_boxes[0], (list, tuple)):
                new_boxes = []
                for i in range(0, len(det_boxes), 6):
                    box = det_boxes[i:i+6]
                    if len(box) == 6:
                        new_boxes.append(box)
                det_boxes = new_boxes

        osd_img.clear()
        if det_boxes:
            for box in det_boxes:
                label = labels[box[0]].lower()
                x1 = box[2]
                y1 = box[3]
                x2 = box[4]
                y2 = box[5]
                x_disp = int(x1 * DISPLAY_WIDTH // OUT_RGB888P_WIDTH)
                y_disp = int(y1 * DISPLAY_HEIGHT // OUT_RGB888P_HEIGHT)
                w_disp = int((x2 - x1) * DISPLAY_WIDTH // OUT_RGB888P_WIDTH)
                h_disp = int((y2 - y1) * DISPLAY_HEIGHT // OUT_RGB888P_HEIGHT)
                osd_img.draw_rectangle(x_disp, y_disp, w_disp, h_disp, color=(255,0,0))

                if label == "chessboard":
                    if chessboard_box is None or (w_disp * h_disp) > (chessboard_box[2] * chessboard_box[3]):
                        chessboard_box = (x_disp, y_disp, w_disp, h_disp)
                elif label in ["black", "white"]:
                    center = (x_disp + w_disp//2, y_disp + h_disp//2)
                    color_val = 1 if label == "black" else 2
                    pieces.append((center[0], center[1], color_val))
                osd_img.draw_string_advanced(x_disp, y_disp-40, 32, labels[box[0]], color=(255,255,255))

        # 如果本帧未检测到棋盘，则采用上一帧棋盘坐标
        if chessboard_box is None and prev_chessboard is not None:
            chessboard_box = prev_chessboard
        elif chessboard_box is not None:
            prev_chessboard = chessboard_box

        # 根据棋盘计算九宫格中心，并判断每个格子是否有棋子
        cell_values = [0]*9
        if chessboard_box is not None:
            x, y, w, h = chessboard_box
            cell_centers = []
            cell_w = w / 3.0
            cell_h = h / 3.0
            for row in range(3):
                for col in range(3):
                    cx = int(x + col * cell_w + cell_w/2)
                    cy = int(y + row * cell_h + cell_h/2)
                    cell_centers.append((cx, cy))
            for i, cell in enumerate(cell_centers):
                best_dist = 1e9
                best_color = 0
                for p in pieces:
                    dist = ((p[0]-cell[0])**2 + (p[1]-cell[1])**2)**0.5
                    if dist < 50 and dist < best_dist:
                        best_dist = dist
                        best_color = p[2]
                if best_dist < 50:
                    cell_values[i] = best_color
                else:
                    cell_values[i] = 0
        # 构造数据包（10个数：第一个为0xE8，其余9个为格子值）
        packet = make_packet(chessboard_box, pieces)  # 新的make_packet根据上述逻辑构造
        uart.write(packet)
        print(packet)

        # 显示结果图像
        Display.show_image(osd_img, 0, 0, Display.LAYER_OSD3)
        gc.collect()
        time.sleep(0.1)
        rgb_img = None

    sensor.stop()
    Display.deinit()
    MediaManager.deinit()
    gc.collect()
    time.sleep(0.1)
    nn.shrink_memory_pool()
    print("det_infer end")
    return 0

if __name__ == "__main__":
    detection()
