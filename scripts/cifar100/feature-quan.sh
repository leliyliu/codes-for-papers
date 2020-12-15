export PYTHONPATH=$PYTHONPATH:'pwd'
export CUDA_VISIBLE_DEVICES=$1
python exps/cifar100/feature_quan.py -lr 5e-3 -n $2 -g --pretrained --quan-mode $3