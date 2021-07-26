import argparse
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from repmlp.repmlp_resnet import *
from repmlp.repmlp import repmlp_model_convert

parser = argparse.ArgumentParser(description='RepVGG Conversion')
parser.add_argument('load', metavar='LOAD', help='path to the weights file')
parser.add_argument('save', metavar='SAVE', help='path to the weights file')
parser.add_argument('-a', '--arch', metavar='ARCH', default='RepMLP-Res50-light-224')

def convert():
    args = parser.parse_args()

    if os.path.isfile(args.load):
        print("=> loading checkpoint '{}'".format(args.load))
        checkpoint = torch.load(args.load, map_location=torch.device('cpu'))

        # 初始model
        if args.arch == 'RepMLP-Res50-light-224':
            train_model = create_RepMLPRes50_Light_224(deploy=False)

            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            elif 'model' in checkpoint:
                checkpoint = checkpoint['model']
            ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
            print(ckpt.keys())
            train_model.load_state_dict(ckpt)

        elif args.arch == 'face-light-96':
            last_label = checkpoint['label_dict']

            train_model = RepMLPResNet(num_blocks=[3,4,6,3], num_classes=len(last_label), 
                block_type='light', img_H=96, img_W=96,
                h=6, w=6, reparam_conv_k=(1,3,5), fc1_fc2_reduction=1, fc3_groups=4,
                deploy=False)

            train_model.load_state_dict(checkpoint['model_state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            total_epochs = checkpoint['epoch']
            last_loss = checkpoint['loss']            
            print(f"Loaded {args.load}: epochs= {total_epochs}, loss= {last_loss:.6f}, num_classes= {len(last_label)}")

        else:
            raise ValueError('TODO')

        
    else:
        print("=> no checkpoint found at '{}'".format(args.load))

    repmlp_model_convert(train_model, save_path=args.save)


if __name__ == '__main__':
    convert()