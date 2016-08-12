data_path=/path/to/samples
model_path=/path/to/models

python  test_segmentation.py \
    --dataset      ${data_path}/test.txt  \
    --basepath     ${data_path}/test  ${data_path}/testannot  \
    --palette      palette.py \
    --save_folder  result \
    --prototxt     ${model_path}/test.prototxt  \
    --caffemodel   ${model_path}/_iter_50000.caffemodel  \
    --gpu_id       0