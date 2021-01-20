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
| reactnet20 | 2e-2 | 200 | 74.49 | - |
| reactnet20 | 5e-3 | 200 | - | - |
| reactnet20 | 1e-3 | 200 | - | - |
| fracbnn20 | 2e-2 | 200 | - | - |
| fracbnn20 | 1e-3 | 200 | - | - |