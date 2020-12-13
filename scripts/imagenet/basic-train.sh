export PYTHONPATH=$PYTHONPATH:'pwd'
python exps/imagenet/main.py ~/datasets/imagenet -g $1 -a $2 -e --pretrained --batch-size 128 --lr 0.1 -e 