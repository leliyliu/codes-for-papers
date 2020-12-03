export PYTHONPATH=$PYTHONPATH:'pwd'
export CUDA_VISIBLE_DEVICES=$1
python exps/cifar10/quan_main.py --lr 1e-3 --quan-mode Conv2dDPQ