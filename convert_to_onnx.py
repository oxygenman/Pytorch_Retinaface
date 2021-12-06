from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--long_side', default=640, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')

args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # ------------------------ export -----------------------------
    output_onnx = 'FaceDetector_960_560.onnx'
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["data"]
    output_names = ["bbox","score","landmark"]
    inputs = torch.randn(1,3, 960, 560 ).to(device)
    #image_path = "./curve/test.jpg"
    #img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #img_raw = cv2.resize(img_raw,(640,640))
    #img = np.float32(img_raw)
    #im_height, im_width, _ = img.shape
    #scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    #img -= (104, 117, 123) 
    #img = img.transpose(2, 0, 1)
    #img = torch.from_numpy(img).unsqueeze(0)
    #inputs = img.to(device)
    
    loc, conf, landms = net(inputs)
    print(loc.shape, conf.shape, landms.shape)
    #dynamic_axes  = {'data':[2,3]}
    torch.onnx.export(net, inputs, output_onnx, export_params=True, verbose=True,
                                   input_names=input_names, output_names=output_names,opset_version=10)
    import onnx

    onnx_model = onnx.load('FaceDetector_960_560.onnx')
    onnx.checker.check_model(onnx_model)

    import onnxruntime

    ort_session = onnxruntime.InferenceSession('FaceDetector_960_560.onnx')

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
    bbox,score,landmark = ort_session.run(None, ort_inputs)
    conf_nump=to_numpy(conf)

    #print(loc[0,:,:][250:])

    print("conf.size:",conf_nump.shape)
    
    print("bbox.size:",bbox.shape)
    print("landmark.size:",landmark.shape)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(loc), bbox, rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    #print(inputs[0][0])


