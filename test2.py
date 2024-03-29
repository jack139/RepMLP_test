import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import accuracy, ProgressMeter, AverageMeter, load_checkpoint
import PIL
from repmlp.repmlp_resnet import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet Test')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('mode', metavar='MODE', default='train', choices=['train', 'deploy'], help='train or deploy')
parser.add_argument('weights', metavar='WEIGHTS', help='path to the weights file')
parser.add_argument('-a', '--arch', metavar='ARCH', default='RepMLP-Res50-light-224')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N',
                    help='mini-batch size (default: 100) for test')
parser.add_argument('-r', '--resolution', default=224, type=int,
                    metavar='R',
                    help='resolution (default: 224) for test')

def test():
    args = parser.parse_args()
    if args.arch == 'RepMLP-Res50-light-224':
        model = create_RepMLPRes50_Light_224(deploy=args.mode=='deploy')
    elif args.arch == 'RepMLP-Res50-bottleneck-224':
        model = create_RepMLPRes50_Bottleneck_224(deploy=args.mode=='deploy')
    elif args.arch == 'RepMLP-Res50-bottleneck-320':
        raise ValueError('TODO')
    elif args.arch == 'face-light-96':
        model = RepMLPResNet(num_blocks=[3,4,6,3], num_classes=18646, 
            block_type='light', img_H=96, img_W=96,
            h=6, w=6, reparam_conv_k=(1,3,5), fc1_fc2_reduction=1, fc3_groups=4,
            deploy=True)
    else:
        raise ValueError('not supported')

    num_params = 0
    for k, v in model.state_dict().items():
        print(k, v.shape)
        num_params += v.nelement()
    print('total params: ', num_params)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
        use_gpu = False
    else:
        model = model.cuda()
        use_gpu = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if os.path.isfile(args.weights):
        print("=> loading checkpoint '{}'".format(args.weights))
        load_checkpoint(model, args.weights)
    else:
        print("=> no checkpoint found at '{}'".format(args.weights))


    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.data, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.resolution == 224:
        trans = transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            #normalize,
        ])
    else:
        trans = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=PIL.Image.BILINEAR),
            #transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            #normalize,
        ]
        )

    val_dataset = datasets.ImageFolder(valdir, trans)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    validate(val_loader, model, criterion, use_gpu)


def validate(val_loader, model, criterion, use_gpu):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if use_gpu:
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            print(target, output)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg




if __name__ == '__main__':
    test()