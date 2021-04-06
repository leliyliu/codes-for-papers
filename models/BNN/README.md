# BNN 
## ReAct 
Reproduce for ReAct shows that it can achieve a good result as the final paper shows

### Models

| Methods | Top1-Acc | FLOPs | Trained Model | 
| - | - | - | - |
XNOR-Net | 51.2% | 1.67 x 10^8 | - |
Bi-Real Net | 56.4% | 1.63 x 10^8 | - | 
Real-to-Binary | 65.4% | 1.83 x 10^8 | -
ReActNet (Bi-Real based) | 65.9% | 1.63 x 10^8 | Model-ReAct-ResNet
ReActNet-A | 69.5% | 0.87 x 10^8 | Model-ReAct-MobileNet

## Cifar10 
| Network | Learning-Rate | Epochs | Top1-acc | Methods 
| - | - | - | - | - |
| birealnet20 | 2e-2 | 200 | 78.99 | - | 
| reactnet20 | 1e-1 | 200 | 78.62 | - |
| fracbnn20 | 1e-1 | 200 | 77.38 | - |
| rbnn_resnet20 | 1e-1 | 200 | 84.38 | - | 
| rbnn_resnet18 | 1e-1 | 200 | | 91.30 | 
| sdbnn_resnet20 | 1e-1 | 200 |84.31 | - | 
| sdbnn_resnet18 | 1e-1 | 200 | 86.63 | - | 
| resnet20(fp) | 1e-1 | 200 | 92.65 | - | 