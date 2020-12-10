export PYTHONPATH=$PYTHONPATH:'pwd'
export CUDA_VISIBLE_DEVICES=$1
python exps/cifar10/feature_quan.py --lr 1e-3 --quan-mode Conv2dHSQ -a $2 --pretrained --teacher-arch MobileNetV2