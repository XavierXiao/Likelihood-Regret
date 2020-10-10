# Likelihood-Regret
Official implementation of [Likelihood Regret: An Out-of-Distribution Detection Score For Variational Auto-encoder](https://arxiv.org/abs/2003.02977) at NeurIPS 2020.
## Training

To train the VAEs, use appropriate arguments and run this command:

```train
python train_pixel.py
```

## Evaluation

To evaluate likelihood regret's OOD detection performance, run

```eval
python compute_LR.py
```

To evaluate [likelihood ratio](https://arxiv.org/abs/1906.02845), run
```eval
python test_likelihood_ratio.py
```

To evaluate [input complexity](https://openreview.net/forum?id=SyxIWpVYvr), run
```eval
python test_inputcomplexity.py
```
Above commands will save the numpy arrays containing the OOD scores for in-distribution and OOD samples in specific location, and to compute aucroc score, run
```eval
python aucroc.py
```

## Pre-trained Models

You can download pretrained VAE models on FMNIST and CIFAR-10 [here](https://drive.google.com/drive/folders/1nX7PmSUq7APE4hTNzAjspIbgAxkChPuM?usp=sharing).
