export CUDA_VISIBLE_DEVICES=$1
mkdir log
python3 train.py --data=~/datasets/imagenet --batch_size=256 --learning_rate=5e-4 --epochs=256 --weight_decay=0 -e | tee -a log/training.txt
