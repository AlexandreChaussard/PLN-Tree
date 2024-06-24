# Tree-based variational inference for Poisson log-normal models
>
>When studying ecosystems, hierarchical trees are often used to organize entities based on proximity criteria, such as the taxonomy in microbiology, social classes in geography, or product types in retail businesses, offering valuable insights into entity relationships. Despite their significance, current count-data models do not leverage this structured information. In particular, the widely used Poisson log-normal (PLN) model, known for its ability to model interactions between entities from count data, lacks the possibility to incorporate such hierarchical tree structures, limiting its applicability in domains characterized by such complexities. To address this matter, we introduce the PLN-Tree model as an extension of the PLN model, specifically designed for modeling hierarchical count data. By integrating structured variational inference techniques, we propose an adapted training procedure and establish identifiability results, enhancisng both theoretical foundations and practical interpretability. Additionally, we extend our framework to classification tasks as a preprocessing pipeline, showcasing its versatility. Experimental evaluations on synthetic datasets as well as real-world microbiome data demonstrate the superior performance of the PLN-Tree model in capturing hierarchical dependencies and providing valuable insights into complex data structures, showing the practical interest of knowledge graphs like the taxonomy in ecosystems modeling.
>

## ⭐ Paper experiments
This code aims at reproducing the results in URL_ARTICLE.

Experiments made in the article can be found in the "experiments" notebook at the root of the repository.
Pre-trained models can be found in the "experiments/save" folder.

## ⭐ Getting started
The getting started guide can be found here (URL). For a quick overview of the package functions, check below.

## 🛠 Installation
- PyTorch (?.?.?)
- pyPLNmodels (?.?.?)

Run
```pip install -r requirements.txt```

## ⭐ Quick overview
Import the PLN-Tree model
```
from plntree.models import PLNTree
```
#### PLN-Tree model usage
```
model = PLNTree(
  ...
)
```

## 📜 Citations
Please cite our work using the following reference:
```
.... (...). PLN-Tree: a Tree-based variational inference for Poisson log-normal models. URL_ARTICLE
```