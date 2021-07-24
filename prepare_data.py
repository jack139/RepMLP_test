import os, shutil

# 准备数据集
#SOURCE = '../datasets/CASIA-maxpy-clean'
SOURCE = '../datasets/glint_asia'
TRAIN = 'data/train'
VALID = 'data/val'

if not os.path.exists(TRAIN):
    os.mkdir(TRAIN)
if not os.path.exists(VALID):
    os.mkdir(VALID)

dirs = os.listdir(SOURCE)
print('total num_classes= ', len(dirs))

for d in dirs:
    src_path = os.path.join(SOURCE, d)
    img_list = os.listdir(src_path)
    if len(img_list)<20:
        continue

    train_path = os.path.join(TRAIN, d)
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    #os.makedirs(train_path)

    val_path = os.path.join(VALID, d)
    if os.path.exists(val_path):
        shutil.rmtree(val_path)
    os.makedirs(val_path)

    shutil.copytree(src_path, train_path)
    
    for i in range(int(len(img_list)*0.2)):
        shutil.move(os.path.join(train_path, img_list[i]), val_path)

dirs = os.listdir(TRAIN)
print('train num_classe= ', len(dirs))
