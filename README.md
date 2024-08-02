# Multi-label Fundus Image Classification
NTU Master’s Thesis : \
Improved Accuracy of Imbalanced Multi-label Classification in Medical Fundus Images

## Abstract
- Proposes a multi-­label image classification model that combines ConvNeXt and Transformer encoder. Initially, the model uses convolutional layers to extract local features from images. Subsequently, it employs the Transformer to explore the interrelationships among these features, thereby facilitating accurate category predictions.
- Introduces a new data augmentation algorithm specifically designed to address the issue of class imbalance. The algorithm use the threshold (0.2) to enhances the dataset by dupli­cating fundus image including mi­nority classes.

## Public fundus image dataset (MuReD)
The data should be download in the path: `data\fundus\MuReD\` \
The dataset can be downloaded from the following link: \
https://www.kaggle.com/datasets/abhirampolisetti/multi-label-retinal-disease-mured-dataset \
The dataset have the `train_data.csv`, `test_data.csv` and `images`(`images\images\` with all fundus images).

## Download python packages
```bash
pip install -r requirements.txt
```

## Data Augmentation
The data augmentation algorithm(`My_proposed_ros.ipynb`) is implemented in the `data_augmentation` file. \
- Need to change the path of the dataset in the notebook.
    ```python
    df = pd.read_csv('../data/fundus/MuReD/train_data.csv')
    ```
- If you want to change the threshold, you can change the value in the notebook. (Default is 0.2)
    ```python
    threshold = label_counts.max() * 0.2
    ```

## Train the model
Using the following command to train the model

```bash
python train.py \
    --model "myconvnext_concatGAP" \
    --save_results_path "results/mured/myconvnext_2layer_concatGAP.csv" \
    --transformer_layer 2 \
    --val \
    --weight_decay \
    --dataset "mured" \
    --save_model
```

Traing the model with the proposed data augmentation algorithm
```bash
python train.py \
    --model "myconvnext_concatGAP" \
    --save_results_path "results/mured/myconvnext_2layer_concatGAP.csv" \
    --transformer_layer 2 \
    --val \
    --weight_decay \
    --dataset "mured" \
    --save_model \
    --data_aug "ros02" # new dataset name should be like [ros02]_train_data.csv in the data folder
```
* More training commands with different models can be found in the `train_all.sh` and `train_ours.sh` file.
* More training parameters can be found in the `train.py` file.

## Results and Model Weights
The results are saved in the `results` folder. \
The model weights are saved in the `saved_models` folder. (If the `--save_model` flag is set)

## Heatmap Visualization
If you want to visualize the heatmap of the model, you can use `model_heatmap.ipynb` file to see ConvNeXt backbone and Transformer encoder which attention in the fundus image. \
(Need to save the model weights in the `saved_models` folder and change the model name in the notebook)
