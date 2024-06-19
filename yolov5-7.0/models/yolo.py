# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules
文件主要由三个部分：Detect类 DetectionModel类 和 parse_model函数 组成。
Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""
"""
/ =========================================== /
    0 导入库与路径
/ =========================================== /
"""

'''====================  0.1 导入安装好的python库  ==================='''
import argparse  # 解析命令行参数模块
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

'''====================  0.2 获取当前文件的绝对路径 ===================='''
FILE = Path(__file__).resolve()   # __file__指的是当前文件(即yolo.py),FILE最终保存着当前文件的绝对路径,比如D://yolov5/models/yolo.py
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

'''====================  0.3 加载自定义模块 =========================='''
from models.common import *  # yolov5网络结构中的通用模块
from models.experimental import *  #  实验性质的代码
from utils.autoanchor import check_anchor_order  # 导入检查anchors合法性的函数
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization  # 定义了Annotator类，可以在图像上绘制矩形框和标注信息
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)  # 定义了一些与PyTorch有关的工具函数

# 导入thop包 用于计算FLOPs
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

"""
/ =========================================== /
    1 Detect 检测头
/ =========================================== /
"""


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction ONNX动态量化
    export = False  # export mode
    '''====================  1.1 获取预测得到的参数  ==================='''

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        # ----------------------------- #
        #   a 成员变量初始化
        # ----------------------------- #
        # nc: 数据集类别数量
        self.nc = nc  # number of classes
        # no: 表示每个anchor的输出数，前nc个字符对应类别，后5个对应：是否有目标conf，目标框的中心xy，目标框的宽高wh
        self.no = nc + 5  # number of outputs per anchor
        # nl=3: 表示检测器的个数，yolov5是3层预测
        self.nl = len(anchors)  # number of detection layers
        # na=3: 表示每个检测器anchors的数量，除以2是因为[10,13, 16,30, 33,23]这个长度是6，对应3个anchor
        self.na = len(anchors[0]) // 2  # number of anchors
        # grid: 表示初始化grid列表大小，后续计算grid，grid就是每个格子的x，y坐标（整数，比如0-19）
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        # anchor_grid即每个grid对应的anchor宽高
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        # 注册常量anchor，并将预选框（尺寸）以数对形式存入，并命名为anchors
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        # ch=(128, 256, 512) 每一张输入图像进行三次预测，每一个预测结果包含nc+5个值
        # (bs, 255, 80, 80),(bs, 255, 40, 40),(bs, 255, 20, 20)
        # 255 -> (nc+5)*3 ===> 为了提取出预测框的位置信息以及预测框尺寸信息
        # ModuleList(
        # (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
        # (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
        # (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
        # )
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        # inplace: 一般都是True，默认不使用AWS，Inferentia加速
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    '''====================  1.2 向前传播  =========================='''
    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4,
                                                                   2).contiguous()  # contiguous 将数据保证内存中位置连续

            if not self.training:  # inference
                # ----------------------------- #
                #   a 重构grid
                # ----------------------------- #
                # 计算得到的网格尺寸与当前特征图的高宽不匹配
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)

                # ----------------------------- #
                #   b Detect
                # ----------------------------- #
                else:  # Detect (boxes only)
                    # 将输入张量切分，但这次只包括xy、wh和置信度
                    # x(bs,3,20,20,85) 按照通道维度（维度索引为4）拆分成三个部分 sigmoid()张量中的数值转换到 (0, 1) 之间
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    # 将模型输出的相对坐标转换为图像坐标空间中的绝对坐标
                    # xy:中心点的 x、y 坐标偏移量（相对于其所在网格点）
                    # xy * 2 + self.grid[i]:将相对偏移量与网格中心点坐标相加，能够得到预测框中心点在特征图上的绝对坐标
                    # self.stride[i]代表当前特征层相对于输入图像的下采样步长。将特征图上的坐标乘以相应的步长，可以将坐标从特征图的尺度转换回原始图像的尺度。这样一来，xy现在就表示了预测框中心在原图像上的真实坐标。
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    # 预测框的宽度和高度信息，将其从模型输出的相对值转换为实际的像素尺寸
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                #  y(bs,na*nx,ny,no) = (bs,3*20*20,85)
                # 调整为批次大小(bs)、锚框数(self.na)乘以特征图的宽高(nx * ny)、以及每个预测包含的输出维度(self.no)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    '''====================  1.3 grid相对坐标转换到绝对坐标系中  ==========='''

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        """
        _make_grid的方法，用于生成用于目标检测或实例分割的网格点坐标和锚框的网格
        Args:
            nx,ny:特征图的宽度和高度
            i:索引特定层级的锚框
        Returns:
        """
        # 获取锚框（anchors）的设备（GPU或CPU）和数据类型，确保生成的网格具有相同的设备和数据类型
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        # 定义网格的形状，其中self.na是该层级的锚框数量，ny和nx是网格的维度，2代表每个网格点有(x, y)两个坐标
        shape = 1, self.na, ny, nx, 2  # grid shape
        # 在指定的设备和数据类型上，创建从0到ny-1和从0到nx-1的一维张量，分别代表y轴和x轴的坐标
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        # 使用torch.meshgrid生成坐标网格
        # grid --> (20, 20, 2), 复制成3倍，因为是三个框 -> (3, 20, 20, 2)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        # torch.Size([20, 20, 2])->torch.Size([1, 3, 20, 20, 2])  复制成3倍，因为是三个框
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        # anchor_grid即每个grid对应的anchor宽高，stride是下采样率，三层分别是8，16，32
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        # 返回生成的网格坐标张量grid和锚框调整后的尺寸张量anchor_grid
        return grid, anchor_grid


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


"""
/ =========================================== /
    2 BaseModel
/ =========================================== /
"""


class BaseModel(nn.Module):
    # YOLOv5 base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        """
        执行一次模型的前向传播过程。可以选择性地开启性能分析和特征可视化
        参数:
        - x: 输入数据
        - profile: 布尔值，是否进行性能分析
        - visualize: 布尔值，是否进行特征的可视化
        """
        # 初始化输出列表y用于存储各层输出，以及时间差列表dt用于性能分析（如果启用）
        y, dt = [], []  # outputs
        # 遍历模型中的所有层及其索引
        for m in self.model:
            if m.f != -1:  # if not from previous layer 如果当前层的输入不是来自上一层（即有特定的输入来源指示）
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            # 开启性能分析时，记录当前层的运行时间到dt
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run 执行当前层的前向传播计算
            # 保存当前层的输出到y列表中，如果该层索引在self.save中则保存，否则不保存以节省内存
            y.append(x if m.i in self.save else None)  # save output
            if visualize:  # 如果开启了特征可视化
                feature_visualization(x, m.type, m.i, save_dir=visualize)  # 对当前层的输出进行特征可视化并保存
        return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


"""
/ =========================================== /
    3 DetectionModel
/ =========================================== /
"""


class DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        '''====================  3.1 load configs  ==================='''
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name  # 获取文件名
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict, 以字典类型存储

        '''====================  3.2 Define model  ==================='''
        # Define model
        # 初始化获取 ch 的值
        # 首先尝试从模型配置 self.yaml 中获取键 'ch' 对应的值。如果存在，则直接使用这个值。
        # 如果不存在，则使用传入的默认值 ch。并确保 self.yaml 配置中包含输入通道数的信息。
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # ----------------------------- #
        #   a 搭建网络
        # ----------------------------- #
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # ----------------------------- #
        #   b Build strides, anchors
        # ----------------------------- #
        # Build strides, anchors
        m = self.model[-1]  # Detect() 获取模型中的最后一个模块m
        if isinstance(m, (Detect, Segment)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # 定义一个临时的前向传播函数forward，用于计算特征图的大小
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward:[8,16,32]
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)  # anchor从原始图像尺度转换到特征层的尺度
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # ----------------------------- #
        #   c Init weights, biases
        # ----------------------------- #
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    '''====================  3.3 forward model  ==================='''
    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg='yolov5s-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLOv5 classification model from a YOLOv5 detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        # Create a YOLOv5 classification model from a *.yaml file
        self.model = None


"""
/ =========================================== /
    4 parse_model 网络配置
    用yaml文件搭建网络结构
    parse_model函数用在DetectionModel模块中，主要作用是解析模型yaml的模块
    通过读取yaml文件中的配置，并且到common.py中找到对应的模块，组成一个完整的模型解析模型文件(字典形式)
    简单来说，就是把yaml文件中的网络结构实例化成对应的模型。后续如果需要动模型框架的话，需要对这个函数做相应的改动。
/ =========================================== /
"""


def parse_model(d, ch):  # model_dict, input_channels(3)
    """
    Parse a YOLOv5 model.yaml dictionary
    Args:
        d: yaml 配置文件（字典形式） yolov5s.yaml
        ch: 记录模型每一层的输出channel，初始ch=[3]，后面会删除
    """
    '''====================  4.1 对应参数初始化  ==================='''
    # 使用 logging 模块输出打印列标签
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # 获取anchors，nc，depth_multiple，width_multiple参数
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    # na: 每个尺度上包含的anchor数
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors, na=3
    # no: na * (类别数 + 5)
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5), no=3*(80+5)

    '''====================  4.2 搭建网络层  ====================='''
    # ----------------------------- #
    #   a 网络层列表, 网络输出引用列表, 输出通道数
    # ----------------------------- #
    # layers:保存每一层的层结构
    # save:记录下所有层结构中from不是-1的层结构序号
    # ch:保存当前层的输出channel
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    # ----------------------------- #
    #   b 使用当前层的参数搭建当前层
    # ----------------------------- #
    # 读取 backbone, head 中的网络单元，每一次循环读取一层
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # f: from，当前层输入来自哪些层
        # n: number，当前层次数 初定
        # m: module，当前层名称
        # args: 当前层类参数 初定
        # layer0:[-1, 1, Conv, [64, 6, 2, 2]], from=-1, number=1, module="Conv", args=[64, 6, 2, 2]

        # 利用 eval 函数, 读取 model 参数对应的类名 如‘Focus’,'Conv'等
        # 利用 eval 函数将字符串转换为变量 如‘None’,‘nc’，‘anchors’等
        m = eval(m) if isinstance(m, str) else m  # eval strings, m:<class 'models.common.Conv'>
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings, [64, 6, 2, 2]

        # depth gain: 控制深度，如yolov5s: n*0.33
        # n: 当前模块的次数(间接控制深度)
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
            Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
            BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
            # ----------------------------- #
            #   c 计算输入输出通道数 c1,c2
            # ----------------------------- #
            # c1: 当前层的输入channel数; c2: 当前层的输出channel数(初定); ch: 记录着所有层的输出channel数
            c1, c2 = ch[f], args[0]
            # no=85，只有最后一层c2=no，最后一层不用控制宽度，输出channel必须是no
            if c2 != no:  # if not output
                # width gain: 控制宽度，如yolov5s: c2*0.5; c2: 当前层的最终输出channel数(间接控制宽度)
                # 求取c2的实际值，c2=c2*width_multiple=64**0.5=32, 判断是否是8的倍数，强制变成8的倍数
                c2 = make_divisible(c2 * gw, 8)

            # ----------------------------- #
            #   d 用当前层的参数搭建当前层
            # ----------------------------- #
            args = [c1, c2, *args[1:]]  # args=[64, 6, 2, 2]-> args=[3, 32, 6, 2, 2]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
        # 判断是否是归一化模块
        elif m is nn.BatchNorm2d:
            args = [ch[f]]  # BN层只需要返回上一层的输出channel
        # 判断是否是Concat连接模块
        elif m is Concat:
            c2 = sum(ch[x] for x in f)  # Concat层则将f中所有的输出累加得到这层的输出channel
        # TODO: channel, gw, gd
        # 判断是否是detect模块
        elif m in {Detect, Segment}:
            # 在args中加入三个Detect层的输出channel
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        '''====================  4.3 打印和保存layers信息  ==============='''
        # m_: 得到当前层的module，将n个模块组合存放到m_里面
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        # 打印当前层结构的一些基本信息
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        # 计算这一层的参数量
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        # 把所有层结构中的from不是-1的值记下 [6,4,14,10,17,20,23]
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        # 将当前层结构module加入layers中
        layers.append(m_)
        if i == 0:
            ch = []  # 去除输入channel[3]
        # 把当前层的输出channel数加入ch
        ch.append(c2)  # 更新ch，下一层取上一层的ch_out
    return nn.Sequential(*layers), sorted(save)  # [4,6,10,14,17,20,23], 需要保存特征层的索引


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
