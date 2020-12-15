export PYTHONPATH=$PYTHONPATH:'pwd'
export CUDA_VISIBLE_DEVICES=$1
python exps/cifar100/kd_quan.py -lr 1e-2 -n $2 -g --pretrained --quan-mode $3