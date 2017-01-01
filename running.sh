data_path=~/caffe/segnet/CamVid
model_path=~/caffe/segnet/models/retrain

python  test_segmentation.py \
    --testset      ${data_path}/test_.txt  \
    --basepath     ${data_path}/test  ${data_path}/testannot  \
    --palette      palette/palette.py \
    --prototxt     ${model_path}/prototxt/test.prototxt  \
    --caffemodel   ${model_path}/caffemodel_bn/test_weights.caffemodel  \
    --gpu_id       0 \
    --save_dir     result
