# Multi-label Fundus Image Classification
Master’s Thesis : \
Improved Accuracy of Imbalanced Multilabel Classification in Medical Fundus Images

## Download the MuReD dataset from Kaggle
https://www.kaggle.com/datasets/abhirampolisetti/multi-label-retinal-disease-mured-dataset 

## Download packages
```bash
pip install -r requirements.txt
```

## Train the model
Using the following command to train the model

```bash
python train.py \
    --model "myconvnext_concatGAP" \
    --save_results_path "results/mured myconvnext_2layer_concatGAP.csv" \
    --transformer_layer 2 \
    --val \
    --weight_decay \
    --dataset "mured" \
    --save_model
```
* More training commands with different models can be found in the `train_all.sh` and `train_ours.sh` file.
* More training parameters can be found in the `train.py` file.

## Results and Model Weights
The results are saved in the `results` folder.
The model weights are saved in the `save_models` folder. (If the `--save_model` flag is set)
The results are saved in the csv file, and the model weights are saved in the `model` folder.


## Heatmap Visualization
If you want to visualize the heatmap of the model, you can use `model_heatmap.ipynb` file.
