## 1、安装：
   pip install numpy
   pip install opencv-python
   
## 2.、執行方式
將資料按照以下目錄整理，即可直接執行。

-data

-train.txt

-test.txt

-val.txt

-Lenet-with-numpy

## 2.、訓練模型

可供訓練的模型共有五種，Lenet5、Lenet5_m、1-Layer-perceptron、2-Layers-perceptron以及參考來源使用的網路架構Lenet5_2

依序執行nn2L.ipynb所有的block可順利訓練1-Layer-perceptron、2-Layers-perceptron兩個模型

依序執行Lenet5_3.ipynb所有的block可訓練Lenet5模型

依序執行Lenet5_m.ipynb所有的block可訓練Lenet5_m模型

依序執行Lenet5_2.ipynb所有的block可訓練Lenet5_2模型

所有程式檔的前半段皆為前處理的步驟，其中由於train data資料較大，因此執行完成後會另外輸出npy檔以供未來使用。

##参考资料：  

本程式碼參考自https://github.com/mengxinZHANG/numpy-realizes-CNN
