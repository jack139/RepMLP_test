## RepMLP TEST

### 使用pre-trained模型

使用训练模型
```
python3 test.py data/ train ../face_model/RepMLP/RepMLP-Res50-light-224_train.pth -a RepMLP-Res50-light-224
```

转换
```
python3 convert.py ../face_model/RepMLP/RepMLP-Res50-light-224_train.pth RepMLP-Res50-light-224_deploy.pth -a RepMLP-Res50-light-224
```

使用deploy模型
```
python3 test.py data/ deploy data/RepMLP-Res50-light-224_deploy.pth -a RepMLP-Res50-light-224
```



### 重新训练

训练
```
python3 train.py
```

转换
```
python3 convert.py data/checkpoint.ckpt.b160_e14_0.9395 data/checkpoint.ckpt.b160_e14_0.9395.deploy -a face-light-96
```

测试
```
python3 test2.py data/ deploy data/checkpoint.ckpt.b160_e14_0.9395.deploy -a face-light-96 -r 96
```
