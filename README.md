# test segmentation
---
Evaluation tool for semantic segmentation task on caffe.

## Requirements
---
- NumPy
- OpenCV
- Caffe

## Arguments
- testset  
Pairs of image and label sample definition text file.
- prototxt  
Model definition file.
- caffemodel  
Trained model parameters.
- palette  
Color to paint in a class map.
- basepath  
Path to a image and label samples.
- save_dir  
Output directory of a painted class map.
- gpu_id  
Using gpu number.
- visualize  
Visualize flag of class maps.

## Usage
```  
python  test_segmentation.py \  
    --testset      /path/to/samples/test.txt  \  
    --prototxt     /path/to/models/test.prototxt  \  
    --caffemodel   /path/to/weights/_iter_50000.caffemodel  \  
    --basepath     /path/to/samples/test  /path/to/samples/testannot  \  
    --palette      palette.py \  
    --save_dir     result \  
    --gpu_id       0  \
    --visualize
```