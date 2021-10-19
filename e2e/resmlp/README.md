End-to-end evaluation for res-mlp. The repo contains a model trained on CIFAR-10 using the provided train script. 
As it takes a long time to train, it is not recommended to train it in the Dockerfile.
The trial script provides options for selecting the number of trials to report.
The digest script processes and prints the results of the trials.

The ResMLP implementation is this one: https://github.com/lucidrains/res-mlp-pytorch
