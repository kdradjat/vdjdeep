# vdjDeep

# BCR sequence annotation using pre-trained DNABERT model

## Requirements
The models run on pytorch. We provide a .yml file that contains all the tools and dependancies.

## Installation
We recommand using conda and creating a separate environment or run it on a cluster.  
You can create an appropriate environment with the command line :
```
conda env create -f torch.yml
```

## Usage
Several scripts are available depending on the task to be performed.

### Fine-tuning DNABERT (train a new model)
If you want to train a new model with your own dataset. 
You need to launch the script [vdjdeep_train.py](https://github.com/kdradjat/vdjdeep/blob/main/single_task/vdjdeep_train.py) by specifying the training dataset, which type of gene you want to identify, the number of classes, and whether or not you want to consider alleles, etc. The training dataset has to be in FASTA format. 
For example, if you want to train a model that identify V genes without alleles, you can use this command line:
```
python vdjdeep_train.py --train_dir [training_file] --kmer 3 --batch_size 32 --weight_decay 0.0 --epoch 100  --nb_classes 76 --nb_seq_max 200000 --type 'V' --save_model --save_name 'model_Vgenes.pt
```
Other examples are available on the script [vdj_train.sh](https://github.com/kdradjat/vdjdeep/blob/main/vdj_train.sh) to launch the training of each type of model.

### Inference (using the trained model)
You need to launch the script [vdjdeep_predict.py](https://github.com/kdradjat/vdjdeep/blob/main/single_task/vdjdeep_predict.py) by specifying the model used and the test dataset and whether or not you consider the alleles. You can also scpecify the output file. 
The test dataset can be as the same format as the training dataset. If it is the case, the accuracy is computed directly. Otherwise, you have to specify --nolabel in the command line. 
You can also choose to apply clustering before passing the sequences into the model by adding --cluster. 
For example, if you want to run inference to identify V genes without alleles, you can use this command line:
```
python vdjdeep_predict.py --model [model_file_path] --test_dir [testing_file] --kmer 3 --batch_size 32 --weight_decay 0.0 --nb_seq_max 200000 --output "output_Vgenes.txt" --nolabel 
```
Other examples are given on the script [vdj_test.py](https://github.com/kdradjat/vdjdeep/blob/main/vdj_test.sh). 
Trained model .pt files for genes/alleles identification are available [here](https://dropsu.sorbonne-universite.fr/s/QztBEcMSX5RWTJi).
