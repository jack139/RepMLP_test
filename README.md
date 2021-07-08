## TEST

使用pre-trained模型
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