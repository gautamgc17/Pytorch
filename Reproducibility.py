# REPRODUCIBLE RESULTS AND DETERMINISTIC BEHAVIOUR

# https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms
# https://discuss.pytorch.org/t/what-is-the-differenc-between-cudnn-deterministic-and-cudnn-benchmark/38054/2


seed = 42
import numpy as np
import random

np.random.seed(seed)
random.seed(seed)


# You can use torch.manual_seed() to seed the Random Number Generator for all devices (both CPU and CUDA)

import torch
torch.manual_seed(seed)
x = torch.rand(2,2)
print(x)


# If using CUDA, Sets the seed for generating random numbers on all GPUs
# Deterministic operations are often slower than nondeterministic operations, so single-run performance may decrease for your model. 
# However, determinism may save time in development by facilitating experimentation, debugging, and regression testing.

torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

