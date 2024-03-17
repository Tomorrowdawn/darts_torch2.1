RUN_NAME="imbalanced_cifar10"

LOG_PATH=./output/${RUN_NAME}.log
METRIC_PATH=./output/${RUN_NAME}_metrics.log
PRINT_PATH=./output/${RUN_NAME}_print.log
mkdir -p ./output
find . -name 'search-*' -type d -exec rm -rf {} +
set -e
python ./cnn/train_search.py \
    --data $CV_DATASETS_ROOT\
    --batch_size 64 \
    --learning_rate 0.025 \
    --learning_rate_min 0.001 \
    --momentum 0.9 \
    --weight_decay 3e-4 \
    --report_freq 50 \
    --epochs 50 \
    --device cuda:0 \
    --init_channels 16 \
    --layers 8 \
    --save ${RUN_NAME} \
    --seed 2 \
    --unrolled \
    > ${PRINT_PATH}
    #--metric_path ${METRIC_PATH} > ${PRINT_PATH}
    