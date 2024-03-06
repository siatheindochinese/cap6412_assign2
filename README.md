# cap6412_assign2
BLIP-2 Part 2

## 0) Requirements

### Python Packages:

Install salesforce-lavis from the original [lavis github repository](https://github.com/salesforce/LAVIS) by cloning it, accessing the cloned repository and perform `pip install -e . ` while inside the repository.

Do not use `pip install salesforce-lavis` as the Blip2 model in the PyPI version does not have `predict_answers` for performing VQA.

### Datasets
If you are a UCF student with access to Newton HPC, simply `scp` the dataset from  Newton. It is located in `/datasets/WGV/`. Alternatively, you may download the WGV dataset [from my google driver folder](https://drive.google.com/file/d/1mkef70zCDNvsSjpukLDb5Ojp3PfrJvoB/view?usp=sharing).

Merge the `datasets` folder in this repository with the `datasets` folder you downloaded. The `datasets/WGV` in this repository has 2 extra jsons, which are extracted from the original csv to aid in mapping and to save memory and computation.

## 1) Running evaluation
I have compiled all precomputed outputs in the folder `precomputed_wgv`. If you wish to have BLIP-2 re-generate the outputs (not recommended, it takes hours) in these folders, simply delete all the files inside and run the evaluation scripts. Otherwise, the evaluation scripts will simply use the saved outputs to compute performance metrics.

There are 5 evaluation scripts in this repository:

- `wgv_ret.py` evaluates zero-shot country classification on the WGV dataset using image-text retrieval with a pretrained BLIP-2 QFormer.
- `wgv_opt_openvqa.py` evaluates country classification as an open-ended VQA task on the WGV dataset using BLIP-2-OPT6.7b.
- `wgv_opt_closedvqa.py` evaluates country classification as a closed-ended VQA task on the WGV dataset using BLIP-2-OPT6.7b.
- `wgv_t5_openvqa.py` evaluates country classification as an open-ended VQA task on the WGV dataset using BLIP-2-FlanT5xl.
- `wgv_t5_closedvqa.py` evaluates country classification as a closed-ended VQA task on the WGV dataset using BLIP-2-FlanT5xl.

## 2) Video Result
A video evidence of running these scripts can be found in the `assets/` folder in this repository.
