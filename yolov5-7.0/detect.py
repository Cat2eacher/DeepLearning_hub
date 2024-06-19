# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.
detect.py主要有run(),parse_opt(),main()三个函数构成
Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
"""
/ =========================================== /
    0 导入库与路径
/ =========================================== /
"""

'''====================  0.1 导入安装好的python库  ==================='''
import argparse  # 解析命令行参数
import os  # 与操作系统、平台相关的库
import platform
import sys
from pathlib import Path

import torch

'''====================  0.2 获取当前文件的绝对路径 ===================='''

FILE = Path(__file__).resolve()  # __file__指的是当前文件(即detect.py),FILE最终保存着当前文件的绝对路径,比如D://yolov5/detect.py
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

'''====================  0.3 加载自定义模块 =========================='''
# 从当前路径下导入自定义的库
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams  # 数据加载
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

"""
/ =========================================== /
    1 run
/ =========================================== /
"""
# 用于自动切换模型的推理模式
# torch.no_grad()
# tensor的requires_grad自动设置为False，不进行梯度的计算
@smart_inference_mode()
# 传入参数，参数可通过命令行传入，也可通过代码传入，parser.add_argument()函数用于添加参数
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos  不保存检测结果的图像或视频，默认为False, 保存
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)  #将source转换为字符串，source为输入的图片、视频、摄像头等
    """====================  1.1 判断输入来源  ==================="""
    # ----------------------------- #
    #   a 处理输入源, 确定输入数据类型
    #   布尔值区分输入是图片、视频、网络流还是摄像头
    # ----------------------------- #
    # 判断是否保存图片，如果nosave为False，且source不是txt文件，则保存图片
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # 判断source是否是文件，Path(source)使用source创建一个Path对象，用于获取输入源信息，判断后缀是否在IMG_FORMATS和VID_FORMATS中，如果是，则is_file为True
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)  # IMG_FORMATS 和 VID_FORMATS 保存图像和视频的扩展名
    #判断source是否是url，如果是，则is_url为True
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # 判断是source是否为摄像头
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    # 判断source是否是截图，如果是，则screenshot为True
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download 确保输入源为本地文件，如果是url，则下载到本地，check_file()函数用于下载url文件

    """====================  1.2 创建输出目录  ==================="""
    # Directories 保存结果的文件夹
    # save_dir是保存运行结果的文件夹名，通过递增的方式来命名
    # 第一次运行时路径是“runs\detect\exp”，第二次运行时路径是“runs\detect\exp1”
    # project 对应的是 runs/detect 的目录
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    """====================  1.3 加载网络模型  ==================="""
    # Load model
    device = select_device(device)  # 选择设备 CPU/CUDA
    # #加载模型，DetectMultiBackend()定义在models.common模块中，函数用于加载模型，
    # weights为模型路径(比如yolov5s.pt)，device为设备，dnn为是否使用opencv dnn，data为数据集，fp16为是否使用fp16推理
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt  #获取模型的stride，names，pt
    # 确保输入图片的尺寸imgsz能整除stride=32 如果不能则调整为能被整除并返回
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    """====================  1.4 加载输入数据  ==================="""
    # Dataloader
    bs = 1  # batch_size,初始化batch_size为1
    # 不同的输入源设置不同的数据加载方式
    if webcam:  #如果source是摄像头，则创建LoadStreams()对象
        view_img = check_imshow(warn=True)  # 检测cv2.imshow()方法是否可以执行，不能执行则抛出异常
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)  # 加载输入数据流
        bs = len(dataset)
    elif screenshot:  #如果source是截图，则创建LoadScreenshots()对象
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:  # image/video dataloader
        # return path, im, im0, self.cap, s
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs   #初始化vid_path和vid_writer，vid_path为视频路径，vid_writer为视频写入对象

    """====================  1.5 网络模型推理  ==================="""
    # Run inference
    # warmup，预热，用于提前加载模型，加快推理速度
    # 使用空白图片（零矩阵）预先用GPU跑一遍预测流程，*imgsz为图像大小，即(1,3,640,640)
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    # seen为已检测的图片数量，windows为空列表，dt为时间统计对象
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:  #遍历数据集，path为图片路径，im为图片，im0s为原始图片，vid_cap为视频读取对象，s为字符信息
        # dataset 通过迭代器来实现，循环一次执行一次__next__
        # return path, im, im0, self.cap, s
        # ----------------------------- #
        #   a 输入图像预处理
        # ----------------------------- #
        with dt[0]:
            # 将图片转换为tensor，并放到模型的设备上，pytorch模型的输入必须是tensor
            im = torch.from_numpy(im).to(model.device)  # torch.Size([3,640,640])
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32  #如果模型使用fp16推理，则将图片转换为fp16，否则转换为fp32
            im /= 255  # 0 - 255 to 0.0 - 1.0  图片归一化，将图片像素值从0-255转换为0-1
            if len(im.shape) == 3:  # 如果图片的维度为3，则添加batch维度
                # 在前面添加batch维度，即将图片的维度从3维转换为4维，即(3,640,640)转换为(1,3,640,640)，pytorch模型的输入必须是4维的
                im = im[None]  # expand for batch dim, torch.Size([1,3,640,640])

        # ----------------------------- #
        #   b 推理前向传播
        # ----------------------------- #
        # Inference
        with dt[1]:  # 开始计时,推理时间
            # 可视化文件路径。如果为True则保留推理过程中的特征图，保存在runs文件夹中
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # 推理，model()函数用于推理，im为输入图片，augment为是否使用数据增强，visualize为是否可视化,输出pred为一个列表
            # pred保存所有的bound_box的信息
            # 模型预测出来的所有检测框，torch.size=[1,25200,85]=[1,80*80*3+40*40*3+20*20*3,85]
            pred = model(im, augment=augment, visualize=visualize)

        # ----------------------------- #
        #   c 预测结果后处理
        # ----------------------------- #
        # NMS 非极大值抑制，用于去除重复的预测框
        with dt[2]:
            # pred: 网络输出结果
            # conf_thres：置信度阈值
            # iou_thres：iou阈值
            # classes: 是否只保留特定的类别 默认为None
            # agnostic_nms：进行nms是否也去除不同类别之间的框
            # max_det: 检测框结果的最大数量 默认1000
            # pred 是一个列表，列表中一个Tensor, tensor.shape = [5,6], 6 =  [box, conf, cla_id]
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            # [1,5,6],  batch_size=1，5个检测框，6=4+1+1 box+confidence+class

        # ----------------------------- #
        #   d 结果保存和输出
        # ----------------------------- #
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        # Process predictions  处理预测结果
        for i, det in enumerate(pred):  # per image
            # i：batch_size = 1
            # det:表示检测框的信息 [5,6]
            seen += 1  #检测的图片数量加1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count  #path[i]为路径列表，ims[i].copy()为将输入图像的副本存储在im0变量中，dataset.count为当前输入图像的帧数
                s += f'{i}: '  #在打印输出中添加当前处理的图像索引号i
            else:
                # 一般都是从LoadImages读取文件中的照片或视频 batch_size=1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path 将路径转换为Path对象
            save_path = str(save_dir / p.name)  # im.jpg 图片/视频的保存路径save_path，p.name为图片名称
            # im.txt，保存预测框坐标的txt文件路径，save_dir为保存图片的文件夹，p.stem为图片名称，dataset.mode为数据集的模式，如果是image，则为图片，否则为视频
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string 打印输出，im.shape[2:]为图片的宽和高
            # 得到原图的宽和高
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # 如果save_crop的值为true，则将检测到的bounding_box单独保存成一张图片
            imc = im0.copy() if save_crop else im0  # for save_crop
            # 绘图工具
            # 创建Annotator对象，用于在图片上绘制预测框和标签,im0为输入图片，line_width为线宽，example为标签
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):  #如果预测框的数量大于0
                # Rescale boxes from img_size to im0 size
                # 将预测信息映射到原图
                # 将标注的bounding_box大小调整为和原图一致（因为推理时原图经过了放缩）此时坐标格式为xyxy
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():  #遍历每个类别,unique()对数组元素去重
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # 遍历每个预测框,xyxy为预测框的坐标，conf为置信度，cls为类别,reversed()函数用于将列表反转，
                    # *是一个扩展语法，*xyxy表示将xyxy中的元素分别赋值给x1,y1,x2,y2
                    if save_txt:  # Write to file,如果save_txt为True，则将预测框的坐标和类别写入txt文件中
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format ,如果save_conf为True，则将置信度也写入txt文件中
                        with open(f'{txt_path}.txt', 'a') as f:  #打 开txt文件,'a'表示追加
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image 如果save_img为True，则将预测框和标签绘制在图片上
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')   #如果hide_labels为True，则不显示标签，否则显示标签，如果hide_conf为True，则不显示置信度，否则显示置信
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:  # 如果save_crop为True，则保存裁剪的图片
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:  # 如果view_img为True，则实时展示图片
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    """====================  1.6 打印输出结果  ==================="""
    # Print results, 打印结果
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image, 每张图片的速度
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")  # 打印保存的路径
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


"""
/ =========================================== /
    2 opt参数
/ =========================================== /
"""
# https://yolov5.blog.csdn.net/article/details/124378167

def parse_opt():  # 为模型进行推理传入参数
    """
    weights: 用于检测的模型权重所在路径
    source: 检测的路径，可以是图片，视频，文件夹，也可以是摄像头（‘0’）
    data: 数据集的配置文件，用于获取类别名称，和训练时的一样
    imgsz: 网络输入的图片大小，默认为640
    conf-thres: 置信度阈值，大于该阈值的框才会被保留
    iou-thres: NMS的阈值，大于该阈值的框会被合并，小于该阈值的框会被保留，一般设置为0.45
    max-det: 每张图片最多检测的目标数，默认为1000
    device: 检测的设备，可以是cpu，也可以是gpu，可以不用设置，会自动选择
    view-img: 是否显示检测结果，默认为False
    save-txt: 是否将检测结果保存为txt文件，包括类别，框的坐标，默认为False
    save-conf: 是否将检测结果保存为txt文件，包括类别，框的坐标，置信度，默认为False
    save-crop: 是否保存裁剪预测框的图片，默认为False
    nosave: 不保存检测结果，默认为False
    classes: 检测的类别，默认为None，即检测所有类别，如果设置了该参数，则只检测该参数指定的类别
    agnostic-nms: 进行NMS去除不同类别之间的框，默认为False
    augment: 推理时是否进行TTA数据增强，默认为False
    update: 是否更新模型，默认为False,如果设置为True，则会更新模型,对模型进行剪枝，去除不必要的参数
    project: 检测结果保存的文件夹，默认为runs/detect
    name: 检测结果保存的文件夹，默认为exp
    exist-ok: 如果检测结果保存的文件夹已经存在，是否覆盖，默认为False
    line-thickness: 框的线宽，默认为3
    hide-labels: 是否隐藏类别，默认为False
    hide-conf: 是否隐藏置信度，默认为False
    half: 是否使用半精度推理，默认为False
    dnn: 是否使用OpenCV的DNN模块进行推理，默认为False
    vid-stride: 视频帧采样间隔，默认为1，即每一帧都进行检测
    """
    parser = argparse.ArgumentParser()
    # ----------------------------- #
    #   a 主要修改
    # ----------------------------- #
    # 模型：训练的权重路径
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    # 输入：测试数据 包括图片/视频路径,'0'(电脑自带摄像头),rtsp等视频流
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    # 数据：配置数据文件路径，包括image/labels/classes等信息
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    # ----------------------------- #
    #   b 次要修改
    # ----------------------------- #
    # 推理时网络输入图片的尺寸
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    # 置信度阈值，default=0.25
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    # 非极大抑制时的 IoU 阈值，默认为 0.45
    # Yolo在每个尺度，每一个gridcell 对应的3个Anchor都会输出检测目标信息，必然存在对同一目标的多次检测
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    # ----------------------------- #
    #   c 基本不改
    # ----------------------------- #
    # 保留的最大检测框数量，每张图片中检测目标的个数最多为1000
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    # 使用的设备，可以是 cuda 设备的 ID（例如 0、0,1,2,3）或者是 'cpu'，默认为 '0'
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 检测的时候是否实时的把检测结果显示出来，默认False
    parser.add_argument('--view-img', action='store_true', help='show results')
    # 是否将预测的框坐标以txt文件形式保存，默认False
    # 使用--save-txt 在路径runs/detect/exp*/labels/*.txt下生成每张图片预测的txt文件，保存了一些类别信息和边框的位置信息。
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # 是否保存检测结果的置信度到 txt文件，默认为 False
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # 是否把模型检测的物体裁剪下来，默认为False
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    # 不保存图片
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # 仅检测指定类别，默认为 None
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # 是否使用数据增强进行推理，默认为 False
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # 是否可视化特征图，默认为 False
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    # 如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
    # 用于在模型训练的最后阶段去除优化器信息，以减小模型文件的大小，并将模型准备好用于推断或其他目的
    parser.add_argument('--update', action='store_true', help='update all models')
    # 结果保存的项目目录路径，默认为 'ROOT/runs/detect'
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    # 结果保存的子目录名称，默认为 'exp'
    parser.add_argument('--name', default='exp', help='save results to project/name')
    # 每次预测模型的结果是否保存在原来的文件夹
    # 如果指定了这个参数，本次预测的结果还是保存在上一次保存的文件夹里；如果不指定就是每次预测结果保存一个新的文件夹下。
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    #  画 bounding box 时的线条宽度，默认为 3
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    #  是否隐藏标签信息，默认为 False
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    # 是否隐藏置信度信息，默认为 False
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    # 是否使用 FP16 半精度进行推理，默认为 False
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # 是否使用 OpenCV DNN 进行 ONNX 推理，默认为 False
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


"""
/ =========================================== /
    3 main
/ =========================================== /
"""


def main(opt):
    # 检查环境/打印参数
    check_requirements(exclude=('tensorboard', 'thop'))  # 检查程序所需的依赖项是否已安装，如果没有安装依赖，则会自动安装
    # 执行run()函数
    # opt 是一个包含参数信息的实例，这个类定义了一些实例变量  (eg) opt.weights = 'yolov5s.pt'
    # vars(opt) 将返回一个字典，其中包含了 opt 的所有实例变量及其对应的值  (eg) 'weights' : 'yolov5s.pt'
    # **vars(opt)将字典解包为关键字参数列表。这在需要将字典作为函数参数传递的情况下非常有用  (eg) weights='yolov5s.pt'
    run(**vars(opt))  # 将 opt 变量的属性和属性值作为关键字参数传递给 run() 函数


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
