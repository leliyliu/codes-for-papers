export PYTHONPATH=$PYTHONPATH:'pwd'
export CUDA_VISIBLE_DEVICES=$1
python exps/cifar100/ad_quan.py -lr 0.01 -n $2 -g --pretrained