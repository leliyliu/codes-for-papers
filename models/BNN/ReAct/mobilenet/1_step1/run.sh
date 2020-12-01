# clear
export CUDA_VISIBLE_DEVICES=$1
python3 train.py --data=/home/liulian/datasets/imagenet --batch_size=256 --learning_rate=5e-4 --epochs=256 --weight_decay=1e-5 --teacher mobilenet_v2 | tee -a log/training.txt
