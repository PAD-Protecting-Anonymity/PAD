# PAD
PAD is a novel privacy-preserving data publication system that leverages interactions with data users to improve utility of privatized datasets.

The repository contains the source code and demos of PAD in Python.

For a detailed description of the algorithms, see the paper (https://ruoxijia.github.io/assets/papers/pad-buildsys-final.pdf).
To cite the paper:

```
@inproceedings{pad2017,
  author = {Jia, Ruoxi and Caleb Sangogboye, Fisayo and Hong, Tianzhen and Spanos, Costas and Baun Kj{\ae}rgaard, Mikkel},
  title = {PAD: Protecting Anonymity in Publishing Building Related Datasets},
  booktitle={Proceedings of the 4th ACM Conference on Embedded Systems for Energy-Efficient Buildings},
  year={2017},
  organization={ACM}
}
```

## Dependencies
- `python3.5` Python environment
- `sklearn` 
- `numpy` 
- `keras` 
- `pandas` 
- `tensorflow` 
- `scipy` 
- `matplotlib`


#PAD repository
The repo contains a number of foldes:
- [Dataset](#Dataset)
- [Demo](#Demo)
- [Utilities](#utilities)
- [K-ward](#K-ward)
- [Metric_Learning](#Metric_Learning)

## Dataset
Datasets made availible for showcasing PAD.

## Demo
Exsamples on how to use PAD.
Contines various exsample on have to use PAD, incluting linear, nonlinear data. Also on how to use the options of various kind of simularaty in the data.

## Utilities
Contines helper functions used in PAD.

## K-ward
Implamentation of K-ward algorithm.

## Metric_Learning
Learning metrics's which can be used in PAD.
Available is implementation for linear and nonlinear data.