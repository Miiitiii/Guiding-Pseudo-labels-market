DATA_ROOT=${1:-'data'}
python train_source.py --data_dir ${DATA_ROOT} --dataset market/source --run_name market_source --num_class 751 --lr 0.001 --batch_size 128