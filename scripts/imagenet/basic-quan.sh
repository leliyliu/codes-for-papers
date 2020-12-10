export PYTHONPATH=$PYTHONPATH:'pwd'
export CUDA_VISIBLE_DEVICES=$1
python exps/imagenet/quan_main.py ~/datasets/imagenet -a $2 --pretrained --batch-size 512 --lr 4e-3 --quan-mode Conv2dHSQ