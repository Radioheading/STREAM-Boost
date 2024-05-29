# STERAM-DANN Model

official pytorch implementation of the paper:

[STREAM: Sequential Training for Real-Time Efficient
Adaptation Model
](https://latex.sjtu.edu.cn/read/vcrhjyjsmwyn#6efa6e)

> Unsupervised domain adaptation (UDA) is a pivotal area in machine learning that aims to transfer knowledge across domains without supervision. Given the substantial time and GPU resources required for training UDA models, accelerating their training is crucial. In our paper, we introduce STREAM, a novel framework that optimizes training time by leveraging target training data upload and download processes. By incorporating confidence distillation and loss adaptation modules, STREAM-enhanced DANN achieves over 80% of the original model’s potential solely during dataset upload or download process under diverse network conditions. Notably, STREAM is independent from specific UDA methods, enabling seamless integration with most of GAN-based UDA techniques in the field.

## Getting started

#### Installation

Install library versions taht are compatible with your environment.

```bash
git clone https://github.com/Radioheading/STREAM-Boost.git STREAM
cd STREAM
conda create -n your_env python=3.7
conda activate your_env
```

#### Recommended configuration

```
python=3.7
pytorch=1.12.1
matplotlib=3.2.2
sklearn=1.0.2
cuda=11.3
```

you can simply run the code with `python main.py`, however, you need to manually change the hyper-parameters to simulate baseline or our model.

## Experimental results

|Baseline | $pps=200$ | $pps=500$ | $pps=2000$ |
|------------|-----------|-----------|------------|
| random order | $74.74\%$ | $72.80\%$ | $66.79\%$ |
| sequential order | $74.03\%$ | $67.50\%$ | $61.20\%$ |

|STREAM-DANN   | $pps=200$ | $pps=500$ | $pps=2000$ |
|------------|-----------|-----------|------------|
| random order | $82.81\%$ | $78.01\%$ | $68.86\%$ |
| sequential order | $78.54\%$ | $75.60\%$ | $68.08\%$ |
