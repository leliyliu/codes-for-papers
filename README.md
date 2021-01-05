# codes-for-papers
Some basic model and codes for papers

## Todo List


- [x] 复现ReAct BNN 网络
- [x] 探索使用feature map 进行量化的思路 
- [x] 实现cifar100 的相关网络，并用于量化实验
- [ ] 利用迭代优化的方式探索混合精度量化内容
    - [x] 实现不包含flops 约束的混合精度量化探索
    - [ ] 添加对于量化精度的限制
- [ ] Blockwisely NAS搜索， on datasets cifar10
    - [ ] 以MobileNetv3 为基本框架实现blockwisely 搜索
    - [ ] 探索BNN 以其为目标的搜索可能性
    - [ ] 对于网络设计过程进行平均场分析