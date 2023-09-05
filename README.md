# Toxicity prediction🧪

The project is built upon libraries of pre-trained machine learning models from various fields.

### Table of contents:

- [Requirements](#requirements)
- [Getting Started](#getting-started)

## Requirements
- [NumPy](https://numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [SciPy](https://www.scipy.org/)
- [rdkit](https://www.rdkit.org/)

It is highly recommended to use a GPU.

any other requirements for specific models is written and explained in the coding files.

## Getting Started
To get started you need to choose the model you want to engage and open the colab notebook in the coding folder where the targeted file is placed.

from a general prespective, all coding files have the same chunk of code of importing the datasets which are placed in this repository project for your convenient use.

the datasets are splitted into 3 categories:
- original datasets
- train datasets
- test datasets

the train,test,and validation datasets are splitted 80/10/10 corrrespondingly. furthermore, in the train dataset folder their are files which was augmented (the operation of adding artifical data by a generator by code) and named after that. This was a part of balancing the data and essential part of the project. 

Also,if you want to save/load models, there is an option for that with google drive since the models are heavy and hence cannot be saved/loaded to github:

1.request premission here: https://drive.google.com/drive/folders/1UWhWA4_phu5alMNFuwgJpHsX9pzKMKYw?usp=sharing.

2.copy this folder to your google drive.

as for the models, they are divided into 3 main categories:
- models based on language structure: chemBerta and RNN.
- models based on graphs: GNN + molclr.
- models based on vector represantaion : DNN and XGB.
  
### chemBerta
To initiate the project you need to follow the commentes in the first block of code.

In addition,if you want to search for hyper parameters(with setting the corresponding variable in code to True) you need to create an account at the site of WANDB library: https://wandb.ai/site.

For your convenience, the full tutorial of chemBerta is [here](https://github.com/deepchem/deepchem/blob/master/examples/tutorials/Transfer_Learning_With_ChemBERTa_Transformers.ipynb).
In addition, because chemBerta utilizes simpleTransformers library, the wandb, that relies on it, outputs hyper paramaters data which was used in order to improve model.
the reports can be found here:https://wandb.ai/toxicityprediction/chemBerta/reportlist



### RNN
Instructions in comments in code.

### GNN
Instructions in comments in code.

### Molclr
somthing about this.....

### DNN
Instructions in comments in code.

### XGB
Instructions in comments in code.





  
