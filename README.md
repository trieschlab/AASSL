# Self-supervised visual learning from interactions with objects

Source code used for the paper "Self-supervised visual learning from interactions with objects" accepted at ECCV 2024.


For now, the paper is available on [arxiv](https://arxiv.org/pdf/2407.06704).

---
## Datasets

**MVImgNet** : Publicly available

**CO3D-v1** : Publicly available

**RT4K** : Available soon

---
## Installation guide

Fork the repository: 

`git clone "PATH_TO_FORKED_REPOSITORY"`

Set up the environment:

```commandline
python3 -m venv ssltt
source ssltt/bin/activate
python3 -m pip install -r requirements.txt
```
---
## Examples

RT4K examples in order: AA-SimCLR, SimCLR, SimCLR-TT, Ciper-SimCLR, EquiMod-SimCLR :
```
python3 train.py --data_root {RT4K_ROOT} --dataset RT4K --modules classic,action,linear_eval --contrast combined
python3 train.py --data_root {RT4K_ROOT} --dataset RT4K --modules classic,linear_eval --contrast classic
python3 train.py --data_root {RT4K_ROOT} --dataset RT4K --modules classic,linear_eval --contrast combined
python3 train.py --data_root {RT4K_ROOT} --dataset RT4K --modules classic,ciper,linear_eval --contrast combined
python3 train.py --data_root {RT4K_ROOT} --dataset RT4K --modules classic,equivariant,linear_eval --contrast combined
```

MVImgNet examples coming soon.

---
## Pre-trained models

100-epochs MVImgNet-F pre-trained models are available there:
https://huggingface.co/aaubret/AASSL/tree/main

 ---
## Citation
```commandline
@article{aubret2024self,
  title={Self-supervised visual learning from interactions with objects},
  author={Aubret, Arthur and Teuli{\`e}re, C{\'e}line and Triesch, Jochen},
  journal={arXiv preprint arXiv:2407.06704},
  year={2024}
}
```

---
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details