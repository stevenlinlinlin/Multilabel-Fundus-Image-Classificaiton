import argparse
import numpy as np
import math
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, average_precision_score, precision_score, recall_score
from sklearn.model_selection import KFold
import copy
from tqdm import tqdm 

# Custom imports
from utils import *
from dataloaders.multilabel_dataset import MultilabelDataset
from models.resnet import ResNet50, ResNet152
from models.densenet import DenseNet169, DenseNet161
from models.mobilenet import MobileNetV2
from models.efficientnet import EfficientNetB3, EfficientNetB5, EfficientNetB7
from models.inception import InceptionV3
from models.vit import ViTForMultiLabelClassification
from models.ctran import CTranModel
from models.utils import custom_replace
from models.swin_transformer import SwinTransformer
from models.convnext import ConvNeXt
from models.mydensenet import myDenseNet1, myDenseNet2, myDenseNet3, myDenseNet4
from models.myconvnext import ConvNeXtTransformer
# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))

prefetch_factor = 64
num_workers = 28
batch_size = 16
# RFMiD dataset
# num_classes = 28
# training_labels_path = 'data/fundus/RFMiD/Training_Set/new_RFMiD_Training_Labels.csv'
# evaluation_labels_path = 'data/fundus/RFMiD/Evaluation_Set/new_RFMiD_Validation_Labels.csv'
# training_images_dir = 'data/fundus/RFMiD/Training_Set/Training'
# evaluation_images_dir = 'data/fundus/RFMiD/Evaluation_Set/Validation'
selected_data  = 'augmented' # 'original' or 'augmented' to evaluate the model on the original or augmented dataset
# MuReD dataset
num_classes = 20
training_labels_path = 'data/fundus/MuReD/train_data.csv'
evaluation_labels_path = 'data/fundus/MuReD/test_data.csv'
training_images_dir = 'data/fundus/MuReD/images/images'
evaluation_images_dir = 'data/fundus/MuReD/images/images'
da_training_images_dir = 'data/fundus/MuReD/images/ros' # 'data/fundus/MuReD/images/xxxx' or None

# auc_fig_path = 'results/auc/densenet161.png'
# results_path = 'results/densenet161_90.csv'

# ctran_model = False # True for CTran, False for CNN
loss_labels = 'all' # 'all' or 'unk'for all labels or only unknown labels loss respectively

# Data transforms
## Transformations adapted for the dataset
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])
## Transformations from the original paper
#### For ViT
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])
#### For CNN 
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     # Resize
#     # transforms.Resize(232), # ResNet50/ResNet152
#     transforms.Resize(256),   # DenseNet169/161, MobileNetV2
#     # transforms.Resize(320),   # EfficientNet B3
#     # transforms.Resize(342),   # InceptionV3
    
#     # CenterCrop
#     transforms.CenterCrop(224),
#     # transforms.CenterCrop(299), # InceptionV3 
#     # transforms.CenterCrop(300), # EfficientNet B3
    
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# Models
def get_model(model_name):
    if model_name == 'resnet':
        model = ResNet152(num_classes).to(device)
    elif model_name == 'densenet':
        model = DenseNet161(num_classes).to(device)
    elif model_name == 'mobilenet':
        model = MobileNetV2(num_classes).to(device)
    elif model_name == 'efficientnet':
        model = EfficientNetB3(num_classes).to(device)
    elif model_name == 'inception':
        model = InceptionV3(num_classes).to(device)
    elif model_name == 'vit':
        model = ViTForMultiLabelClassification(num_labels=num_classes).to(device)
    elif model_name == 'ctran':
        model = CTranModel(num_labels=num_classes,use_lmt=True,pos_emb=False,layers=3,heads=4,dropout=0.1).to(device)
    elif model_name == 'swin':
        model = SwinTransformer(num_classes=num_classes).to(device)
    elif model_name == 'convnext':
        model = ConvNeXt(num_classes=num_classes).to(device)
    elif model_name == 'mydensenet4':
        model = myDenseNet4(num_classes).to(device)
    elif model_name == 'myconvnext':
        model = ConvNeXtTransformer(num_classes).to(device)
    
    return model

# datasets
def get_dataset():
    # train dataset
    train_dataset = MultilabelDataset(ann_dir=training_labels_path,
                                root_dir=training_images_dir,
                                num_labels=num_classes,
                                transform=transform, known_labels=1, testing=False, da_root_dir=da_training_images_dir)

    # val dataset
    test_dataset = MultilabelDataset(ann_dir=evaluation_labels_path,
                                root_dir=evaluation_images_dir,
                                num_labels=num_classes,
                                transform=transform, known_labels=0, testing=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, prefetch_factor=prefetch_factor, num_workers=num_workers)
    return train_dataset, test_dataset, test_loader


# trainset to train and validation (0.8, 0.2)   
def train(model, train_dataset, learning_rate, ctran_model=False, evaluation=False):
    num_epochs = 35
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=0.01) # for transformers
    # optimizer = optim.Adam(model.parameters(), lr=0.00001) # c-tran
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1) 
    
    if evaluation:
        # torch.manual_seed(13)
        total_size = len(train_dataset)
        val_size = int(total_size * 0.2)
        train_size = total_size - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(146))
        
        # train_label_counts = count_labels(train_dataset, num_classes)
        # val_label_counts = count_labels(val_dataset, num_classes)
        # sorted_train_label_counts = dict(sorted(train_label_counts.items()))
        # sorted_val_label_counts = dict(sorted(val_label_counts.items()))
        # print("Train Label Counts:     ", sorted_train_label_counts)
        # print("Validation Label Counts:", sorted_val_label_counts)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, prefetch_factor=prefetch_factor, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, prefetch_factor=prefetch_factor, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, prefetch_factor=prefetch_factor, num_workers=num_workers)

    best_train_loss = float('inf')
    best_val_loss = float('inf')
    best_model_state = None
    for epoch in tqdm(range(num_epochs), desc='Epoch'):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if ctran_model:
                labels = batch['labels'].float()
                images = batch['image'].float()
                mask = batch['mask'].float()
                unk_mask = custom_replace(mask,1,0,0)
                mask_in = mask.clone()
                
                optimizer.zero_grad()
                outputs,_,_ = model(images.to(device),mask_in.to(device))
                
                loss =  F.binary_cross_entropy_with_logits(outputs.view(labels.size(0),-1),labels.cuda(),reduction='none')
                if loss_labels == 'unk': 
                    # only use unknown labels for loss
                    loss_out = (unk_mask.cuda()*loss).sum()
                else: 
                    # use all labels for loss
                    loss_out = loss.sum()
                    
            else:
                inputs, labels = batch['image'].to(device), batch['labels'].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                # print(outputs.shape, labels.shape)
                loss_out = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none').sum() # sigmoid + BCELoss
            
            train_loss += loss_out.item()
            loss_out.backward()
            optimizer.step()
            
        scheduler.step()   

        if not evaluation:
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_model_state = copy.deepcopy(model.state_dict())
                
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss/len(train_loader):.6f}')
            continue
    
        # Evaluate the model on the validation set
        model.eval()
        val_loss = 0.0
        # correct_predictions = 0
        # total_jaccard_index = 0.0
        # total_samples = 0
        auc_scores = []

        with torch.no_grad():
            all_preds = []
            all_labels = []
            all_preds_4 = []
            all_labels_4 = []
            all_preds_5 = []
            all_labels_5 = []
            for batch in val_loader:
                if ctran_model:
                    labels = batch['labels'].float()
                    images = batch['image'].float()
                    mask = batch['mask'].float()
                    mask_in = mask.clone()
                    unk_mask = custom_replace(mask,1,0,0)
                    
                    outputs,int_pred,attns = model(images.to(device),mask_in.to(device))
                    
                    loss = F.binary_cross_entropy_with_logits(outputs.view(labels.size(0),-1),labels.cuda(), reduction='none')
                    loss_out = (unk_mask.cuda()*loss).sum()
                else:
                    inputs, labels = batch['image'].to(device), batch['labels'].to(device)
                    outputs = model(inputs)
                    loss_out = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none').sum()
                    
                val_loss += loss_out.item()

                # Calculate accuracy
                ## method 1. Strictly Accuracy
                # predicted_labels = (outputs > 0.5).float()
                # correct_predictions += (predicted_labels == labels).all(dim=1).sum().item()
                # total_samples += labels.size(0)
                
                ## method 2. Jaccard Accuracy
                # predicted = (outputs > 0.5).bool()
                # labels_bool = labels.bool()
                # intersection = (predicted & labels_bool).float().sum(dim=1)
                # union = (predicted | labels_bool).float().sum(dim=1)
                # jaccard_index_per_example = intersection / union
                # jaccard_index_per_example[union == 0] = 1.0
                # total_jaccard_index += jaccard_index_per_example.sum().item()
                # total_samples += labels.size(0)
                
                ## method 3. AUC
                outputs_np = F.sigmoid(outputs).cpu().numpy()
                # outputs_np = outputs.cpu().numpy()
                labels_np = labels.cpu().numpy()
                all_preds.extend(outputs_np)
                all_labels.extend(labels_np)
                
                ## method 4. mAP
                all_preds_4.append(F.sigmoid(outputs).cpu())
                # all_preds_4.append(outputs.cpu())
                all_labels_4.append(labels.cpu())
                
                ## method 5. F1 Score
                predicted = F.sigmoid(outputs).cpu() > 0.5
                # predicted = outputs.cpu() > 0.5
                all_preds_5.append(predicted.numpy())
                all_labels_5.append(labels.cpu().numpy())

        current_val_loss = val_loss / len(val_loader)
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
        
        if selected_data == 'original':
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss/len(train_loader):.6f}, Validation Loss: {val_loss/len(val_loader):.6f}')
            continue
        
        ## method 1.
        # accuracy = correct_predictions / total_samples
        ## method 2.
        # accuracy = total_jaccard_index / total_samples
        ## method 3.
        for i in range(labels_np.shape[1]):
                label_specific_auc = roc_auc_score([label[i] for label in all_labels], [pred[i] for pred in all_preds])
                auc_scores.append(label_specific_auc)
        average_auc = sum(auc_scores) / len(auc_scores)
        ## method 4. mAP
        all_preds_4 = torch.cat(all_preds_4).numpy()
        all_labels_4 = torch.cat(all_labels_4).numpy()
        mAP = 0
        for i in range(all_labels_4.shape[1]):
            AP = average_precision_score(all_labels_4[:, i], all_preds_4[:, i])
            mAP += AP

        mAP /= all_labels_4.shape[1]
        ## method 5. F1 Score
        all_preds_5 = np.vstack(all_preds_5)
        all_labels_5 = np.vstack(all_labels_5)
        f1_macro = f1_score(all_labels_5, all_preds_5, average='macro')
        
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss/len(train_loader):.6f}, Validation Loss: {val_loss/len(val_loader):.6f}, F1_macro: {f1_macro:.3f}, mAP: {mAP:.3f}, Average AUC: {average_auc:.3f}')
    return best_model_state


# Evaluate the model on the test set
def evaluate(model, best_model_state, test_loader, results_path, ctran_model=False, best_model=False):
    if best_model:
        print("------ Best model evaluation -----")
        model.load_state_dict(best_model_state)
        
    model.eval()
    # correct_predictions = 0
    # total_jaccard_index = 0.0
    # total_samples = 0
    auc_scores = []
    precision_scores = []
    recall_scores = []

    with torch.no_grad():
        
        all_preds = []
        all_labels = []
        all_preds_4 = []
        all_labels_4 = []
        all_preds_5 = []
        all_labels_5 = []
        
        for batch in test_loader:
            if ctran_model:
                labels = batch['labels'].float()
                images = batch['image'].float()
                mask = batch['mask'].float()
                mask_in = mask.clone()
                unk_mask = custom_replace(mask,1,0,0)
                
                outputs,int_pred,attns = model(images.to(device),mask_in.to(device))
            else:
                inputs, labels = batch['image'].to(device), batch['labels'].to(device)
                outputs = model(inputs)
                
            # Calculate accuracy
            ## method 1. Strictly Accuracy
            # predicted_labels = (outputs > 0.5).float()
            # correct_predictions += (predicted_labels == labels).all(dim=1).sum().item()
            # total_samples += labels.size(0)
            
            ## method 2. Jaccard Accuracy
            # predicted = (outputs > 0.5).bool()
            # labels_bool = labels.bool()
            # intersection = (predicted & labels_bool).float().sum(dim=1)
            # union = (predicted | labels_bool).float().sum(dim=1)
            # jaccard_index_per_example = intersection / union
            # jaccard_index_per_example[union == 0] = 1.0
            # total_jaccard_index += jaccard_index_per_example.sum().item()
            # total_samples += labels.size(0)
            
            ## method 3. AUC
            outputs_np = F.sigmoid(outputs).cpu().numpy()
            # outputs_np = outputs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_preds.extend(outputs_np)
            all_labels.extend(labels_np)
            
            ## method 4. mAP
            all_preds_4.append(F.sigmoid(outputs).cpu())
            # all_preds_4.append(outputs.cpu())
            all_labels_4.append(labels.cpu())
            
            ## method 5. F1 Score
            predicted = F.sigmoid(outputs).cpu() > 0.5
            # predicted = outputs.cpu() > 0.5
            all_preds_5.append(predicted.numpy())
            all_labels_5.append(labels.cpu().numpy())


    ## method 1.
    # accuracy = correct_predictions / total_samples
    ## method 2.
    # accuracy = total_jaccard_index / total_samples
    ## method 3.
    # print(len(all_preds), all_preds[0].shape, labels_np.shape)
    for i in range(labels_np.shape[1]):
        # print(len(all_labels))
        # print([label[i] for label in all_labels], [pred[i] for pred in all_preds])
        label_specific_auc = roc_auc_score([label[i] for label in all_labels], [pred[i] for pred in all_preds])
        auc_scores.append(label_specific_auc)
        
        # Precision and Recall
        label_specific_precision = precision_score([label[i] for label in all_labels], [pred[i] > 0.5 for pred in all_preds], zero_division=0)
        label_specific_recall = recall_score([label[i] for label in all_labels], [pred[i] > 0.5 for pred in all_preds], zero_division=0)
        precision_scores.append(label_specific_precision)
        recall_scores.append(label_specific_recall)
        
    average_auc = sum(auc_scores) / len(auc_scores)
    average_precision = sum(precision_scores) / len(precision_scores)
    average_recall = sum(recall_scores) / len(recall_scores)
    ## method 4. mAP
    all_preds_4 = torch.cat(all_preds_4).numpy()
    all_labels_4 = torch.cat(all_labels_4).numpy()
    mAP = 0
    mAP_per_label = []
    for i in range(all_labels_4.shape[1]):
        AP = average_precision_score(all_labels_4[:, i], all_preds_4[:, i])
        mAP_per_label.append(AP)
        mAP += AP

    mAP /= all_labels_4.shape[1]
    ## method 5. F1 Score
    all_preds_5 = np.vstack(all_preds_5)
    all_labels_5 = np.vstack(all_labels_5)
    f1_macro = f1_score(all_labels_5, all_preds_5, average='macro')
    f1_list = list(f1_score(all_labels_5, all_preds_5, average=None))
    result2csv(results_path, evaluation_labels_path, precision_scores, recall_scores, f1_list, mAP_per_label, auc_scores)
    # print(f'Evaluation - Average Precision: {average_precision:.3f}, Average Recall: {average_recall:.3f}, F1_macro: {f1_macro:.3f}, mAP: {mAP:.3f}, Average AUC: {average_auc:.3f}, ML Scores: {(mAP + average_auc) / 2:.3f}')
    normal_auc = auc_scores.pop(1)
    average_auc = sum(auc_scores) / len(auc_scores)
    normal_f1 = f1_list.pop(1)
    f1_macro = sum(f1_list) / len(f1_list)
    mAP_per_label.pop(1)
    mAP = sum(mAP_per_label) / len(mAP_per_label)
    ML_score = (mAP + average_auc) / 2
    eval_results = [f1_macro, mAP, average_auc, ML_score, normal_f1, normal_auc, (ML_score + normal_auc) / 2]
    eval_results = [str(round(result, 3)) for result in eval_results]
    results2allcsv(results_path, eval_results)
    print(f'Evaluation - Average Precision: ML_F1: {f1_macro:.3f}, ML_mAP: {mAP:.3f}, ML_AUC: {average_auc:.3f}, ML_Score: {ML_score:.3f}, Bin_F1: {normal_f1:.3f}, Bin_AUC: {normal_auc:.3f}, Model_Score: {(ML_score + normal_auc) / 2:.3f}')
    # plot_auc_curve(all_preds, all_labels, evaluation_labels_path, auc_fig_path)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--save_results_path', type=str)
    # parser.add_argument('--training_labels_path', type=str)
    parser.add_argument('--ctran_model', action='store_true')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--val', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    train_dataset, test_dataset, test_loader = get_dataset()
    model = get_model(args.model)
    print(f"===== Model: {model.__class__.__name__} =====")
    print(f"<training_labels_path: {training_labels_path}>")
    print("******************** Training   ********************")
    best_model_state = train(model, train_dataset, args.lr, ctran_model=args.ctran_model, evaluation=args.val)
    # best_model_state = train_plm(model, train_dataset, args.lr, ctran_model=args.ctran_model, evaluation=args.val, num_classes=num_classes, batch_size=batch_size, prefetch_factor=prefetch_factor, num_workers=num_workers, device=device)
    # best_model_state = train_kfold(model, train_dataset, args.lr, ctran_model=args.ctran_model)
    print("******************** Testing ********************")
    evaluate(model, best_model_state, test_loader, args.save_results_path, ctran_model=args.ctran_model)
    # evaluate(model, best_model_state, test_loader, args.save_results_path, ctran_model=args.ctran_model, best_model =True)
