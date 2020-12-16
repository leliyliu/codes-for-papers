export PYTHONPATH=$PYTHONPATH:'pwd'
export CUDA_VISIBLE_DEVICES=$1
python exps/cifar100/main_quan.py -lr 0.02 -n $2 -g --pretrained --quan-mode $3 --q-mode kernel_wise