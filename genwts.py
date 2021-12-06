import torch
from torch import nn
import torchvision
import os
import struct
#from torchsummary import summary
from data import cfg_mnet, cfg_re50
from models.retinaface import RetinaFace


def main():
    print('cuda device count: ', torch.cuda.device_count())
    device = 'cuda:0'
    #net = RetinaFace(cfg=cfg_mnet, phase = 'test')
    net = torch.load('retinaface.pth')
    #net = net.to(device)
    #net.eval()
    print('model: ', net)
    #print('state dict: ', net.state_dict().keys())
    #tmp = torch.ones(1, 3, 352, 640).to(device)
    #print('input: ', tmp)
    #out = net(tmp)
    #print('output:', out)

    if os.path.exists('retinaface.wts'):
        return
    f = open("retinaface.wts", 'w')
    f.write("{}\n".format(len(net.state_dict().keys())))
    for k,v in net.state_dict().items():
        print('key: ', k)
        print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")

if __name__ == '__main__':
    main()