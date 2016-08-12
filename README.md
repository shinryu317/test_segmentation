# test segmentation
---
Caffeにおけるセマンティックセグメンテーションの評価ツールです。  
Global accuracy, Class accuracy, Mean IoU の評価ができます。

## Requirements
---
本ソースコードは以下のライブラリが必要です。  
- NumPy
- OpenCV
- Caffe

## プログラム引数
- dataset  
テストサンプルと教師サンプルのペアが書かれたテキストファイル。
- basepath  
テストサンプルと教師サンプルまでのパス。
ファイル名は dataset に書かれているためパスだけでよい。
- palette  
テスト画像を評価して得られるクラスマップに塗る色を定義したモジュール。
- save_folder  
テスト画像を評価した後に描き出される保存するフォルダ（ディレクトリ）。  
ファイル名は連番であるためフォルダだけでよい。
- prototxt  
評価に用いる *.prototxt。
- caffemodel  
評価に用いる *.caffemodel。
- gpu_id  
動作させるGPUのID。CPUで動作させるなら-1を指定する。

## 実行例
```  
python  test_segmentation.py \  
    --dataset      /path/to/samples/test.txt  \  
    --basepath     /path/to/samples/test  /path/to/samples/testannot  \  
    --palette      palette.py \  
    --save_folder  result \  
    --prototxt     /path/to/models/test.prototxt  \  
    --caffemodel   /path/to/weights/_iter_50000.caffemodel  \  
    --gpu_id       0  
```