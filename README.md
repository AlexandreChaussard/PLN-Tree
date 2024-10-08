# Tree-based variational inference for Poisson log-normal models
>
> Hierarchical organization of entities, such as taxonomies in microbiology, provide crucial insights into ecosystems but are often ignored by current count-data models. To fill that gap, we propose the PLN-Tree model, an extension of the PLN model designed to incorporate hierarchical structures into count data analysis. The experiments conducted in this work with synthetic and microbiome data emphasize the value of integrating hierarchical organization of entities through PLN-Tree for count data modeling.
> 
> In its current form, PLN-Tree can be used for data augmentation tasks, preprocessing of compositional data (like CLR, ALR, ILR transforms), or to unveil interaction graphs of entities along their hierarchy.
> 

## ⭐ Paper experiments
This code aims at reproducing the results in [Chaussard et al. (2024). Tree-based variational inference for Poisson log-normal models](https://arxiv.org/abs/2406.17361).

Experiments made in the article can be found in the `experiments` notebook at the root of the repository.

## ⭐ Getting started
For a quick overview of the package functions, check the [getting started notebook](https://github.com/AlexandreChaussard/PLN-Tree/blob/master/Getting_started.ipynb).

## 🛠 Installation

Run
```pip install -r requirements.txt```

## 📦 Package structure

Please find the PLN-Tree models in `plntree.models`. 

Studied datasets are imported in `plntree.data`. 
Several pre-trained models can be found in `experiments/save` for each dataset in different settings,
see the experiments notebooks for more information.

## 📜 Citations
Please cite our work using the following reference:
```
Chaussard, A., Bonnet, A., Gassiat, E., & Corff, S. L. (2024). Tree-based variational inference for Poisson log-normal models. arXiv preprint arXiv:2406.17361.
```