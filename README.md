# DyTox-HAR
Human activity recognition (HAR) has the potential to transform a wide array of sectors, including healthcare, fitness, surveillance, smart environments, and factory automation. It employs sensors and machine learning techniques to accurately identify and classify human behaviour and actions. The growing use of compact, wearable sensors, such as smartwatches, stresses the need for more resource-efficient machine learning models, given the constrained computational capacity of these devices. Simultaneously, the rapid expansion of HAR necessitates models capable of continually adapting to new data streams without the need to retrain the entire model, presenting a challenge to the conventional, resource-intensive continual learning architectures. Dynamic Mixture of Experts ([DMOE](https://github.com/rah-man/dmoe-har)) introduces a lightweight continual learning approach that achieves or surpasses the performance of many state-of-the-art machine learning architectures. This project aims to advance the DMOE model and proposes the substitution of DMOE’s gating mechanism with an adaptation of a cutting-edge transformer architecture, known as Dynamic Token Expansion ([DyTox](https://github.com/arthurdouillard/dytox)). The results show that this architecture improves upon the performance of DMOE, while maintaining a small parameter space and training time.

## Installation
The datasets DSADS, PAMAP2, HAPT, and WISDM are publicly available online for download. Links to the pages are provided below:
- DSADS and PAMAP2: https://www.kaggle.com/datasets/jindongwang92/crossposition-activity-recognition?resource=download
- HAPT: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- WISDM: https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset
The project runs on Python 3.10; the project dependencies can be installed from the requirements.txt file in the repository.

## Running
From the base project directory, the project can be run using:
```shell
python main.py --options <path-to-yaml-file>
```
YAML files for each of the datasets are provided in the options directory. Once the datasets have been downloaded, be sure to update the path to the relevant dataset in each of the YAML configurations. DSADS should point to the dsads.mat file, PAMAP2 should point to the pamap.mat file, HAPT should point to the directory in which the Train/ and Test/ directories are located, and WISDM should point to arff_files/phone/accel/all.csv.

Alternatively, the project can be run using arguments in the command line, with or without the use of a YAML file. For example:
```shell
python main.py --data-set wisdm --data-path har/WISDM/dataset/arff-files/phone/accel/all.csv --features 91 --embed-dim 768 --patch-size 48 --num-classes 18 --base-increment 2 --increment 2 --batch-size 32 --n-epochs 500
```
