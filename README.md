# Combining Diverse Feature Priors
This repository contains code for reproducing the results of our paper.

Paper: https://arxiv.org/abs/2110.08220

Blog Post: http://gradientscience.org/copriors/


Important files:
```
Scripts:
  pretrain_model.py: a script to pre-train the models on just the labeled data
  cotrain.py: a script to co-train pretrained model(s)
  sweep_final_models.py: a script to evaluate intermediate eras for a previously run cotrain
  
File Structure:
  datasets:
    datasets.py: the definition of the labeled/unlabeled/validation/test sets for our datasets
    transforms.py: describes the different prior transforms and spurious tinting
    co_training.py: contains the logic for model pre-training and co-training
   models:
    bagnet_custom.py: the architecture for the bagnets used in this paper
    model_utils.py: utilities for loading and building models
```

To generate the pre-trained priors, run:
```
python pretrain_model.py --dataset <DATASET NAME> --data-path <DATA PATH> --use_val --out-dir <OUTPUT PATH NAME> --arch <ARCHITECTURE NAME> --epochs 300 --lr <LR> --step_lr <STEP LR> --step_lr_gamma <STEP LR GAMMA> --additional-transform <TRANSFORM TYPE>

datasets: STLSub10, cifarsmallsub, celebaskewed 
data-path: use torchvision datasets from https://pytorch.org/vision/stable/index.html
use-val: determines whether to use validation or test set for tensorboard metrics
arch: vgg16_bn, bagnetcustom32 (bagnet for CIFAR), bagnetcustom96thin (bagnet for celeba/stl10)
lr, step-lr, step-lr-gamma are hyperparameters who's exact values can be found in our appendix.
additional-transform: which prior to use. possibilities are NONE, CANNY, SOBEL (use NONE and a bagnet architecture for the bagnet prior)

Add --spurious TINT to train with a tint (as in the tinted STL-10 experiments)
```

After generating the priors, the models can be self (include one prior directory) or co-trained (include both prior directories) by running:
```
python cotrain.py --dataset <DATASET NAME> --data-path <DATA PATH> --out-dir <OUTPUT PATH> --input-dirs <PRIOR DIRECTORY 1> --input-dirs <PRIOR DIRECTORY 2> --epochs_per_era 300 --fraction 0.05 --eras 20 --epochs 400 --arch vgg16_bn --additional-transform NONE --lr <LR> --step_lr <STEP LR> --step_lr_gamma <STEP LR GAMMA> --strategy STANDARD_CONSTANT 

This command will self/co-train the input prior directories, saving a checkpoint for each era, and then finally train a standard model on the pseudo-labels after the eras are complete.

To use the pure co-training strategy, add --pure
To use tinting as in the STL-10 tinting experiments
```
