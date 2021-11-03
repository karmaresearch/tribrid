# Tribrid

## Code Structure

The code of the model is based on the [Hugging Face](https://github.com/huggingface) library of pretrained BERT. The structure is as follows:

* **run_classifier.py:**
The utility set of the model, containing a set of dedicated self-created classess and methods that are used in training and testing the model. It is based on the pretrained BERT from huggingface.

* **Tribrid.py:**
This file contains our proposed model for stance classificarion based on BERT with triplet (siamese) structure and a joint loss function with 3 losses.

* **Tribrid_pos.py:**
This file contains our proposed model for stance classificarion based on BERT with siamese structure and a joint loss function with 2 losses.

* **Post_Process:**
This folder contains the code to post process the logtis and distance for the final stance classification as well as the flyingsquid implementation.

* **requirements.txt:**
Dependencies to run the code. 

The datasets and the models are stored here: 

https://drive.google.com/drive/folders/14Er5Fzy9HaYwv3Hx-bLytlwsm8VHjxYy

## How to run the code

First fill in the model, dataset and output path in Tribrid.py/Tribrid_pos.py.

To run locally with a ```GPU``` :

```console
python -u Trbrid.py/Tribrid_pos.py
```

Or to run using a cluster that supports ```prun``` and ```GPU``` :

```console
prun -np 1 -native '-C TitanX --gres=gpu:1' python Trbrid.py/Tribrid_pos.py
```
