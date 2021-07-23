import os, shutil

# 准备数据集

dirs = os.listdir('CASIA-maxpy-clean')
print(len(dirs))

for d in dirs:
  path = os.path.join('CASIA-maxpy-clean', d)
  img_list = os.listdir(path)
  if len(img_list)<5:
    shutil.rmtree(path)
    continue

  val_path = os.path.join('val', d)
  if os.path.exists(val_path):
    shutil.rmtree(val_path)
  os.makedirs(val_path)
  
  for i in range(int(len(img_list)*0.2)):
    shutil.move(os.path.join(path, img_list[i]), val_path)

dirs = os.listdir('CASIA-maxpy-clean')
print(len(dirs))
