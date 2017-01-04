DATA_PATH=/path/to/data
MODEL_PATH=/path/to/model

python  test_segmentation.py \
    --testset      $DATA_PATH/test.txt  \
    --basepath     $DATA_PATH/test  $DATA_PATH/testannot  \
    --palette      palette/palette.py \
    --prototxt     $MODEL_PATH/test.prototxt  \
    --caffemodel   $MODEL_PATH/_iter_50000.caffemodel  \
    --gpu_id       0 \
    --save_dir     result
