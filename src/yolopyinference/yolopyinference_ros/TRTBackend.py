
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import cv2
import numpy as np
from pathlib import Path
from yolopyinference_ros.utils.YoloUtils import yaml_load


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
        # if names[0] == 'n01440764' and len(names) == 1000:  # ImageNet
        #     names = yaml_load(ROOT / 'data/ImageNet.yaml')['names']  # human-readable names

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
    
