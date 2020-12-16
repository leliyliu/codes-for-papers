# Cifar100 Experiments 

| models | fp32 | LSQ | DPQ | KD | feature-KD | DPQ(ADMM)
| - | :-: | :-: | :-: | :-: | :-: | :-: | 
| Mobilenetv2 | 67.94 | 62.76 <br> (18.93 MMac) | 67.05 <br> (35.28 MMac) | 66.52 <br> (18.93 MMac) | 67.48 <br> (18.93 MMac) | - | 
| Mobilenetv3(large) | 71.63 | 71.09 <br> (20.24 MMac) |  - | - | - | - | 
| Vgg16 | 72.30 | 71.62 <br> (99.94 MMac) | - | - | - | - |
| ResNet18 | 76.26 <br> (557.02 MMac) |  - | - | - | - | - |