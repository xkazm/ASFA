# Code for ASFA

## Prerequisites:
* python == 3.8.5
* pytorch == 

## Dataset:
* Please manually download the datasets BNCI201401, BNCI201402, BNCI201404 by [MOABB](https://github.com/NeuroTechX/moabb).

## Framework:
* bci: common approaches in BCIs:
  * CSP: 
  * EA:
  * MDRM:
  * RA:
  * EEGNet:
  * DeepConvNet:
* bsfda: black-box soure model for source-free domain adaptation:
  * Source: source only
  * SHOT-IM, SHOT: 
  * ASFA, ASFA-aug: ASFA-aug add data augmentation when performing knowledge distillation
* libs: public function used in this project:
  * augment: augment functions
  * cdan, dan, dann, grl, jan, kernel: files for domain adversarial training, code from https://github.com/thuml/Transfer-Learning-Library
  * dataLoad: load and compute tangent space features for EEG data
  * DataIterator: data iterator when training deep networks
  * network, eegnet, deepconvent, DomainDiscriminator: model definition
  * loss: loss functions
  * utils: common used functions
* sfda: approaches for source-free domain adaptation:
  * Source: source only
  * BAIT
  * SHOT-IM, SHOT:
  * ASFA: our proposed approaches
* uda: approaches for unsupervised domain adaptation:
  * CDAN: CDAN/CDAN-E,
  * DAN:
  * DANN:
  * DJDAN:
  * JAN:
  * MCC:

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