"""Concept-first Korean teaching manuscripts. No executable model code in the reader."""
SOURCES={
 'ALEXNET':('Krizhevsky et al. (2012) · AlexNet','https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf'),
 'HW':('Hubel & Wiesel (1962) · 고양이 시각피질의 수용 영역','https://pmc.ncbi.nlm.nih.gov/articles/PMC1359523/'),
 'F':('Fukushima (1980) · Neocognitron','https://pubmed.ncbi.nlm.nih.gov/7370364/'),
 'L':('LeCun et al. (1998) · Gradient-Based Learning Applied to Document Recognition','https://bottou.org/papers/lecun-98h'),
 'R':('He et al. (2015/2016) · Deep Residual Learning','https://arxiv.org/abs/1512.03385'),
 'T':('Geirhos et al. (2019) · Shape / texture bias','https://arxiv.org/abs/1811.12231'),
 'S':('Geirhos et al. (2020) · Shortcut Learning','https://arxiv.org/abs/2004.07780'),
 'REC':('Tang et al. (2018) · Recurrent computations for visual pattern completion','https://doi.org/10.1073/pnas.1719397115'),
 'BN':('Ioffe & Szegedy (2015) · Batch Normalization','https://arxiv.org/abs/1502.03167'),
 'D':('Srivastava et al. (2014) · Dropout','https://jmlr.org/papers/v15/srivastava14a.html'),
 'G':('Selvaraju et al. (2017) · Grad-CAM','https://arxiv.org/abs/1610.02391'),
 'TASKS':('Torchvision · Classification / Detection / Segmentation','https://docs.pytorch.org/vision/stable/models.html'),
 'MNIST':('MNIST · 데이터 구성과 공식 분할','https://www.tensorflow.org/datasets/catalog/mnist'),
 'MNISTCODE':('Torchvision MNIST · 공식 로더와 데이터 미러','https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html'),
}

def P(title,kicker,lead,paragraphs,visual,takeaway,ask='',sources=(),layout='split',notes=''):
    return dict(title=title,kicker=kicker,lead=lead,paragraphs=paragraphs,visual=visual,takeaway=takeaway,ask=ask,sources=sources,layout=layout,notes=notes)

from week1_mnist_content import make_week1
W1=make_week1(P)

from week2_alexnet_content import make_week2
W2=make_week2(P)
