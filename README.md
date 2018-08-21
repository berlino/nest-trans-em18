# Implementation of Neural Transition-based Model for Nested Mention Recognition

We provide sample sentences from GENIA for illustration.
Requirement: PyTorch (tested on 0.4), Python(v3)

## Setup

1. Put the embedding file [glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/) into the folder of embeddings (or directly downlaod from [here](https://drive.google.com/open?id=1qDmFF0bUKHt5GpANj7jCUmDXgq50QJKw))
2. run the gen_data.py to generate the processed data for training, they will be defaultly placed at the ./data folder
3. run train.py to start training

We already generate the processed data for our samples. So you can skip the first and second steps if you only want to test our sample sentences.


## Configuration

Configurations of the model and training are in config.py

Some important configurations:

* if\_gpu: whether to use GPU for training
* input\_dropout: whether to add dropout layer
* if\_pretrained: whether to using Glove pretrained embeddings


