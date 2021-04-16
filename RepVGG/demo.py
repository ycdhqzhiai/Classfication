import argparse
import os
import time
import torch

import cv2
import numpy as np

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import accuracy, ProgressMeter, AverageMeter

from repvgg import get_RepVGG_func_by_name

parser = argparse.ArgumentParser(description='PyTorch ImageNet Test')
parser.add_argument('--path', type=str, default='3.png', help='path to dataset')
parser.add_argument('--mode', type=str, default='deploy', help='train or deploy')
parser.add_argument('--weights', type=str, default='RepVGG-A0-deploy.pth', help='path to the weights file')
parser.add_argument('--arch', type=str, default='RepVGG-A0')


def onnx_export(net, img, weights):
    try:
        import onnx
        from onnxsim import simplify

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = weights.replace('.pth', '.onnx')  # filename
        torch.onnx.export(net, img, f, verbose=False, opset_version=11, input_names=['images'],
                        output_names=['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, f)
        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
        exit()
    except Exception as e:
        print('ONNX export failure: %s' % e)


def preprocess(img):
    img = cv2.resize(img, (48,48))
    cv2.imwrite('eye48x48.jpg', img)
    img = img / 255
    img = img - [0.485, 0.456, 0.406]
    img = img / np.array([0.229, 0.224, 0.225])
    img = img.transpose(2, 0, 1)
    img_in = np.expand_dims(img, axis=0).astype(np.float32)
    return img_in

def main():
    args = parser.parse_args()
    onnx = True
    repvgg_build_func = get_RepVGG_func_by_name(args.arch)

    model = repvgg_build_func(deploy=args.mode=='deploy')
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
        use_gpu = False
    else:
        model = model.cuda()
        use_gpu = True
    if os.path.isfile(args.weights):
        print("=> loading checkpoint '{}'".format(args.weights))
        checkpoint = torch.load(args.weights)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        ckpt = {k.replace('module.', ''):v for k,v in checkpoint.items()}   # strip the names
        model.load_state_dict(ckpt)
    else:
        print("=> no checkpoint found at '{}'".format(args.weights))


    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], #123.675 116.28
    #                                  std=[0.229, 0.224, 0.225])
    model.eval()
    with torch.no_grad():
        img = cv2.imread(args.path)
        image = preprocess(img)
        image = torch.from_numpy(image).cuda()

        # ONNX export
        if onnx:
            onnx_export(model, image, args.weights)
            exit()
        output = model(image)
        print(output)
    #     for i, (images, target) in enumerate(val_loader):
    #         if use_gpu:
    #             images = images.cuda(non_blocking=True)
    #             target = target.cuda(non_blocking=True)

    #         # compute output
    #         output = model(images)
    #         loss = criterion(output, target)

    #         # measure accuracy and record loss
    #         acc1, acc5 = accuracy(output, target, topk=(1, 5))
    #         losses.update(loss.item(), images.size(0))
    #         top1.update(acc1[0], images.size(0))
    #         top5.update(acc5[0], images.size(0))

    #         # measure elapsed time
    #         batch_time.update(time.time() - end)
    #         end = time.time()

    #         if i % 10 == 0:
    #             progress.display(i)

    #     print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
    #           .format(top1=top1, top5=top5))

    # return top1.avg




if __name__ == '__main__':
    main()