export PYTHONPATH=$PYTHONPATH:'pwd'
export CUDA_VISIBLE_DEVICES=$1
python exps/cifar10/main.py --lr 5e-3 -a $2 