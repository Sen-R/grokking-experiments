# Experiments on Grokking: Generalization Beyond Overfitting

## Introduction

Some experiments implementing and extending the results in the paper
[*Grokking: Generalization Beyond Overfitting on Small Algorithmic
Datasets](https://arxiv.org/pdf/2201.02177.pdf) (Power et al., 2022).

## Getting started

This code doesn't require any special dependencies beyond TensorFlow and
NumPy. In particular, it should work out-of-the box on Colab or similarly
configured ML environment. If setting up from scratch, you can use the
accompanying `requirements.txt` file to install the few dependencies you
need.

The entry point for running experiments is the module `scripts.train`. This
sets up and runs a single experiment, training a given model on a given
dataset using the given optimisation and training parameters and saving the
results of the run.

An example invokation of the script is as follows:

```bash
python -m scripts.train runs-dir \
	--dataset cubic_polynomial \
	--train-frac 0.5 \
	--p 97 \
	--model transformer \
	--learning-rate 0.001 \
	--epochs 200 \
	--steps-per-execution 100
```

Note that many of the scripts parameters have suitable default values (e.g.
`p` defaults to 97 and by default the transformer architecture uses the
same number of layers, heads and feature dimension as in Power et al.).
You can run `scripts.train --help` for hints on the arguments.

The output of a run is saved in `run-dir` (in the command above, or whatever
directory is provided ot the script), appended with a timestamp for that run
(so that calling the command again doesn't overwrite the results of the previous
run. The output directory contains the following contents:

* `params.json`: a JSON dictionary with all the input parameters (including
  defaults that weren't explicitly specified) used for this run.
* `data`: a directory containing the train and validation datasets used for
  this particular experiment.
* `history.json`: a "lines-style" JSON file, containing one JSON dictionary
  per line, corresponding to each epoch of training. Each dictionary stores
  various metrics for that epoch, such as loss and accuracy for the train
  and test sets, as well as some summaries of the model weights.
* `checkpoints`: TF model checkpoints saved every 10 epochs during training,
  allowing you to load model weights at various stages of training and, for
  example, evaluate its outputs on the train or val datasets.
