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


# PAD repository
The repo contains a number of folders:
- [Dataset](#dataset)
- [Demo](#demo)
- [Utilities](#utilities)
- [K-ward](#k-ward)
- [Metric_Learning](#metric_Learning)

## Dataset
Datasets made available for showcasing PAD.

## Demo
Examples on how to use PAD.
Contains various examples on how to use PAD, including linear, nonlinear data. Also on how to use the options of various kind of similarity in the data.

## Utilities
Contains helper functions used in PAD.

## K-ward
Implementation of K-ward algorithm.

## Metric_Learning
Learning distance metrics which can be used in PAD.
Available is an implementation for linear and nonlinear data.
