# Simple-implementation-of-Mobile-Former

At present, only the model but no trained. There may be some bug in the code, and some details may be different from the original paper, if you are interested in this, welcome to discuss.

Add: CutUp,MixUp,RandomErasing,SyncBatchNorm for DDP train

There are tow way for qkv aline in new code，A: Split token dim into heads; B: Broadcast x while product

Add: Make model by config(mf52, mf294, mf508) in config.py, the number of parameters almost same with paper

Train：python main.py --name mf294 --data path/to/ImageNet --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --batch-size 256 

Inference:

paper:https://arxiv.org/pdf/2108.05895.pdf

https://github.com/xiaolai-sqlai/mobilenetv3

https://github.com/lucidrains/vit-pytorch

https://github.com/Islanna/DynamicReLU
