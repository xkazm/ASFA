# Code for ASFA

## Prerequisites
* python == 3.8.5
* torch == 1.8.1
* numpy == 1.20.1
* scipy == 1.6.1
* mne == 0.22.0
* scikit-learn == 0.23.2
* pyriemann == 0.2.6

## Dataset
* Please manually download the datasets BNCI2014001, BNCI2014002, BNCI2014004 by [MOABB](https://github.com/NeuroTechX/moabb).

## Framework
* bci: common approaches in BCIs:
  * Common spatial pattern ([CSP](https://ieeexplore.ieee.org/document/895946))
  * Euclidean alignment ([EA](https://ieeexplore.ieee.org/document/8701679))
  * Minimum distance to mean ([MDRM](https://ieeexplore.ieee.org/document/6046114))
  * Riemannian alignment ([RA](https://ieeexplore.ieee.org/document/8013808))
  * [EEGNet](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta)
  * [DeepConvNet](https://onlinelibrary.wiley.com/doi/abs/10.1002/hbm.23730)
* bsfda: black-box source model for source-free domain adaptation:
  * Source: source only
  * Source HypOthesis Transfer ([SHOT-IM, SHOT](https://proceedings.mlr.press/v119/liang20a.html))
  * ASFA, ASFA-aug: our proposed approach, ASFA-aug add data augmentation when performing knowledge distillation
* libs: public function used in this project:
  * augment: augment functions
  * cdan, dan, dann, grl, jan, kernel: files for existing unsupervised domain adaptation approaches, code from https://github.com/thuml/Transfer-Learning-Library
  * dataLoad: load and compute tangent space features for EEG data
  * DataIterator: data iterator when training deep networks
  * network, eegnet, deepconvent, DomainDiscriminator: model definition
  * loss: loss functions
  * utils: common used functions
* sfda: approaches for source-free domain adaptation:
  * Source: source only
  * [BAIT](https://arxiv.org/abs/2010.12427)
  * Source HypOthesis Transfer ([SHOT-IM, SHOT](https://proceedings.mlr.press/v119/liang20a.html))
  * ASFA: our proposed approach
* uda: approaches for unsupervised domain adaptation:
  * Conditional domain adversarial network ([CDAN/CDAN-E](https://proceedings.neurips.cc/paper/2018/hash/ab88b15733f543179858600245108dd8-Abstract.html))
  * Domain adaptation network ([DAN](https://ieeexplore.ieee.org/abstract/document/8454781/))
  * Domain-adversarial neural network ([DANN](https://www.jmlr.org/papers/volume17/15-239/15-239.pdf))
  * Dynamic joint domain adaptation network ([DJDAN](https://ieeexplore.ieee.org/abstract/document/9354668/))
  * Joint adaptation netowrk ([JAN](http://proceedings.mlr.press/v70/long17a.html))
  * Minimum class confusion ([MCC](https://link.springer.com/chapter/10.1007/978-3-030-58589-1_28))

## Run
When you have prepared the datasets, you can directly run the corresponding .py file.

For example,
```angular2html
cd ASFA
python sfda/ASFA.py --gpu_id '0' --device 'cuda' --fileroot your_data_file_path --output ASFA
```

## Citation
If you find this code useful for your research, please cite our papers
```angular2html
@article{XiaASFA2022,
    title={Privacy-preserving domain adaptation for motor imagery-based brain-computer interfaces},
    author={Kun Xia and Lingfei Deng and Wlodzislaw Duch and Dongrui Wu},
    journal={IEEE Trans. on Biomedical Engineering},
    year={2022},
    note={in press}
}
```

## Contact
[kxia@hust.edu.cn](mailto:kxia@hust.edu.cn)
