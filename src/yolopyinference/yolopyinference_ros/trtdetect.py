#ref: https://github.com/ultralytics/yolov5/blob/master/detect.py

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from std_msgs.msg import String
from vision_msgs.msg import Detection2D
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import ObjectHypothesisWithPose
from pathlib import Path
from yolopyinference_ros.YoloUtils import yaml_load, check_img_size, letterbox, increment_path, non_max_suppression, scale_boxes, xyxy2xywh
#, tensor_to_torch_array, select_device, non_max_suppression, scale_boxes, xyxy2xywh


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    #print_args(vars(opt))
    return opt


def smart_inference_mode(torch_1_9=True):
    # Applies torch.inference_mode() decorator if torch>=1.9.0 else torch.no_grad() decorator
    def decorate(fn):
        return (torch.inference_mode if torch_1_9 else torch.no_grad)()(fn)

    return decorate





class TRTBackend(nn.Module):
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        fp16 = True
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA

        import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
        from collections import OrderedDict, namedtuple
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        context = model.create_execution_context()
        bindings = OrderedDict()
        output_names = []
        fp16 = False  # default updated below
        dynamic = False
        for i in range(model.num_bindings):
            name = model.get_binding_name(i)
            dtype = trt.nptype(model.get_binding_dtype(i))
            if model.binding_is_input(i):
                if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                    dynamic = True
                    context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                if dtype == np.float16:
                    fp16 = True
            else:  # output
                output_names.append(name)
            shape = tuple(context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size
        #return dynamic, bindings

        # class names
        if 'names' not in locals():
            names = yaml_load(data)['names'] if data else {i: f'class{i}' for i in range(999)}
        if names[0] == 'n01440764' and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / 'data/ImageNet.yaml')['names']  # human-readable names

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        
        if self.dynamic and im.shape != self.bindings['images'].shape:
            i = self.model.get_binding_index('images')
            self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
            self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
            for name in self.output_names:
                i = self.model.get_binding_index(name)
                self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
        s = self.bindings['images'].shape
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = [self.bindings[x].data for x in sorted(self.output_names)]

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        #warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if (self.device.type != 'cpu'):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            #for _ in range(2 if self.jit else 1):  #
            self.forward(im)  # warmup
    
    # @staticmethod
    # def _model_type(p='path/to/model.pt'):
    #     # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
    #     # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
    #     from export import export_formats
    #     from utils.downloads import is_url
    #     sf = list(export_formats().Suffix)  # export suffixes
    #     if not is_url(p, check=False):
    #         check_suffix(p, sf)  # checks
    #     url = urlparse(p)  # if url may be Triton inference server
    #     types = [s in Path(p).name for s in sf]
    #     types[8] &= not types[9]  # tflite &= not edgetpu
    #     triton = not any(types) and all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])
    #     return types + [triton]

    @staticmethod
    def _load_metadata(f=Path('path/to/meta.yaml')):
        # Load metadata from meta.yaml if it exists
        if f.exists():
            d = yaml_load(f)
            return d['stride'], d['names']  # assign stride, names
        return None, None
    

# def main(opt):
#     #check_requirements(exclude=('tensorboard', 'thop'))
#     run(**vars(opt))

# if __name__ == "__main__":
#     opt = parse_opt()
#     main(opt)
# weights=ROOT / 'yolov5s.pt',  # model path or triton URL
#         source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
#         data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
#         imgsz=(640, 640),  # inference size (height, width)
#         conf_thres=0.25,  # confidence threshold
#         iou_thres=0.45,  # NMS IOU threshold
#         max_det=1000,  # maximum detections per image
#         device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
#         view_img=False,  # show results
#         save_txt=False,  # save results to *.txt
#         # save_conf=False,  # save confidences in --save-txt labels
#         # save_crop=False,  # save cropped prediction boxes
#         nosave=False,  # do not save images/videos
#         classes=None,  # filter by class: --class 0, or --class 0 2 3
#         agnostic_nms=False,  # class-agnostic NMS
#         augment=False,  # augmented inference
#         visualize=False,  # visualize features
#         #update=False,  # update all models
#         project=ROOT / 'runs/detect',  # save results to project/name
#         name='exp',  # save results to project/name
#         # exist_ok=False,  # existing project/name ok, do not increment
#         # line_thickness=3,  # bounding box thickness (pixels)
#         # hide_labels=False,  # hide labels
#         # hide_conf=False,  # hide confidences
#         half=False,  # use FP16 half-precision inference
#         #dnn=False,  # use OpenCV DNN for ONNX inference
#         vid_stride=1,  # video frame-rate stride
#@smart_inference_mode()
def detectfun(paramdict):
    weights = paramdict['weights']
    source = paramdict['source'] #str(paramdict.source)
    data = paramdict['data']
    imgsz = paramdict['imgsz']
    conf_thres = paramdict['conf_thres']
    iou_thres = paramdict['iou_thres']
    max_det = paramdict['max_det']
    device = paramdict['device']
    nosave = paramdict['nosave']
    classes = paramdict['classes']
    agnostic_nms = paramdict['agnostic_nms']
    visualize = paramdict['visualize']
    project = paramdict['project']
    name = paramdict['name']
    save_txt = paramdict['save_txt']
    
    half = paramdict['half']

    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    # if is_url and is_file:
    #     source = check_file(source)  # download

    # Directories
    save_dir = Path(project) / name #increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    if torch.cuda.is_available():
        print(torch.cuda.device_count())
        device=torch.device('cuda:0')
    else:
        device=torch.device('cpu')
    
    model = TRTBackend(weights, device=device, dnn=False, data=data, fp16=half)
    stride, names= model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup

    #dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # # Dataloader
    bs = 1  # batch_size
    im0 = cv2.imread(source)  # BGR
    img_size=640
    stride=32
    auto=False #True
    im = letterbox(im0, img_size, stride=stride, auto=auto)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous

    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Run inference
    visualize = increment_path(save_dir / Path(source).stem, mkdir=True) if visualize else False
    pred = model(im, augment=False, visualize=visualize)

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # pred = [cx, cy, w, h, conf, pred_cls(80)]
    detections_arr = Detection2DArray()

    # Process predictions
    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in det:
                xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1)
                #xywh = xyxy
                print(xywh)
    
                obj = Detection2D()
                obj.bbox.size_x = float(xywh[2])
                obj.bbox.size_y = float(xywh[3])
                obj.bbox.center.position.x = float(xywh[0])
                obj.bbox.center.position.y = float(xywh[1])
                obj.id = str(int(cls))
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(int(cls))
                hyp.hypothesis.score = float(conf)
                obj.results.append(hyp)
                detections_arr.detections.append(obj)

            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s = f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

    return detections_arr


class TRTDetectNode(Node):
    def __init__(self, name='trtdetect_node'):
        super().__init__(name)
        param_names = ['weights', 'source', 'data', 'imgsz', 'conf_thres', 'iou_thres', 'max_det', \
            'device', 'view_img', 'save_txt', 'nosave', 'classes', 'agnostic_nms', 'augment', \
                'visualize', 'project', 'name', 'half', 'vid_stride']
        self.params_config = {}
        my_parameter_descriptor = ParameterDescriptor(dynamic_typing=True)#(description='This parameter is mine!')
        for param_name in param_names:
            self.declare_parameter(param_name, None, my_parameter_descriptor)
            try:
                self.params_config[param_name] = self.get_parameter(
                    param_name).value
            except rclpy.exceptions.ParameterUninitializedException:
                self.params_config[param_name] = None
        imgsz=self.params_config['imgsz']
        newimgsz=(imgsz[0], imgsz[1])
        self.params_config['imgsz']=newimgsz
        # self.declare_parameters(
        #     namespace='',
        #     parameters=[
        #         ('conf_thres', rclpy.Parameter.Type.DOUBLE),
        #         ('iou_thres', rclpy.Parameter.Type.DOUBLE),
        #         ('max_det', rclpy.Parameter.Type.INTEGER)
        #     ])
    
        
        # Create the publisher. This publisher will publish a Detection2DArray message
        # to topic object_detections. The queue size is 10 messages.
        self.publisher_ = self.create_publisher(
            Detection2DArray, 'object_detections', 10)
        
        #Detection function
        print(self.params_config)
        detections = detectfun(self.params_config)

        # Publish the message to the topic
        self.publisher_.publish(detections)
        

def main(args=None):
    try:
        rclpy.init(args=args)
        node = TRTDetectNode('trtdetect_node')
        rclpy.spin(node)
    except Exception as e: 
        print(e)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()