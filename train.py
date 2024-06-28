import argparse
import numpy as np
import math
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR, OneCycleLR
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, average_precision_score, precision_score, recall_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
import copy
from tqdm import tqdm 

# Custom imports
from utils import *
from dataloaders.multilabel_dataset import MultilabelDataset
from loss_functions.focal import FocalLoss
from loss_functions.asymmetric import AsymmetricLossOptimized
from loss_functions.polyloss import Poly1CrossEntropyLoss, Poly1FocalLoss
from models.resnet import ResNet50, ResNet152
from models.densenet import DenseNet169, DenseNet161, DenseNet121
from models.mobilenet import MobileNetV2
from models.efficientnet import EfficientNetB3, EfficientNetB5, EfficientNetB7, EfficientNet_v2
from models.inception import InceptionV3
from models.vit import ViTForMultiLabelClassification, ViT
from models.c_tran.ctran import CTranModel
from models.utils import custom_replace
from models.swin_transformer import SwinTransformer
from models.convnext import ConvNeXt
from models.mydensenet import myDenseNet1, myDenseNet2, myDenseNet3, myDenseNet4
from models.myconvnext import ConvNeXtTransformer, ConvNeXtTransformer_concatGAP
from models.maxvit import MaxViT
# from models.mvit import MViT_v2
from models.coatnet import CoAtNet
from models.add_gcn import ADD_GCN
from models.query2label.query2label import build_q2l
from models.ml_decoder import create_model
from models.tresnet import create_tresnet_model
from models.mcar import mcar_resnet101

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))

prefetch_factor = 64
num_workers = 28
# selected_data  = 'augmented' # 'original' or 'augmented' to evaluate the model on the original or augmented dataset
# auc_fig_path = 'results/auc/densenet161.png'
# results_path = 'results/densenet161_90.csv'
# ctran_model = False # True for CTran, False for CNN
loss_labels = 'all' # 'all' or 'unk'for all labels or only unknown labels loss respectively

# Data transforms
## Transformations adapted for the dataset training
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((384, 384)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])
## Transformations adapted for the dataset testing
transform4test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((384, 384)),
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


def dataset2train(dataset_name, data_aug=None):
    valid_labels_path = ''
    valid_images_dir = ''
    
    if data_aug:
        # print(f"[Data Augmentation: {data_aug}]")
        if dataset_name == 'rfmid':
            # RFMiD dataset
            num_classes = 29
            normal_class_index = 0
            training_labels_path = f"data/fundus/RFMiD/Training_Set/{data_aug}_new_RFMiD_Training_Labels.csv"
            evaluation_labels_path = 'data/fundus/RFMiD/Evaluation_Set/new_RFMiD_Validation_Labels.csv'
            training_images_dir = 'data/fundus/RFMiD/Training_Set/Training'
            evaluation_images_dir = 'data/fundus/RFMiD/Evaluation_Set/Validation'
            da_training_images_dir = f"data/fundus/RFMiD/Training_Set/{data_aug}"
        elif dataset_name == 'mured':
            # MuReD dataset
            num_classes = 20
            normal_class_index = 1
            training_labels_path = f"data/fundus/MuReD/{data_aug}_train_data.csv"
            evaluation_labels_path = 'data/fundus/MuReD/test_data.csv'
            training_images_dir = 'data/fundus/MuReD/images/images'
            evaluation_images_dir = 'data/fundus/MuReD/images/images'
            da_training_images_dir = f"data/fundus/MuReD/images/{data_aug}"
        elif dataset_name == 'itri':
            num_classes = 15
            normal_class_index = 1
            training_labels_path = f"data/fundus/MuReD/{data_aug}_train_data.csv"
            evaluation_labels_path = 'data/fundus/MuReD/test_data.csv'
            training_images_dir = 'data/fundus/MuReD/images/images'
            evaluation_images_dir = 'data/fundus/MuReD/images/images'
            da_training_images_dir = f"data/fundus/MuReD/images/{data_aug}"
            valid_labels_path = ''
            valid_images_dir = ''
    else:
        # print("[Original Data]")
        if dataset_name == 'rfmid':
            # RFMiD dataset
            num_classes = 29
            normal_class_index = 0
            training_labels_path = 'data/fundus/RFMiD/Training_Set/new_RFMiD_Training_Labels.csv'
            evaluation_labels_path = 'data/fundus/RFMiD/Evaluation_Set/new_RFMiD_Validation_Labels.csv'
            training_images_dir = 'data/fundus/RFMiD/Training_Set/Training'
            evaluation_images_dir = 'data/fundus/RFMiD/Evaluation_Set/Validation'
            da_training_images_dir = 'data/fundus/RFMiD/Training_Set/Training'
        elif dataset_name == 'mured':
            # MuReD dataset
            num_classes = 20
            normal_class_index = 1
            training_labels_path = 'data/fundus/MuReD/train_data.csv'
            evaluation_labels_path = 'data/fundus/MuReD/test_data.csv'
            training_images_dir = 'data/fundus/MuReD/images/images'
            evaluation_images_dir = 'data/fundus/MuReD/images/images'
            da_training_images_dir = 'data/fundus/MuReD/images/images' # 'data/fundus/MuReD/images/xxxx' or None
        elif dataset_name == 'itri':
            num_classes = 15
            normal_class_index = 1
            training_labels_path = 'data/fundus/MuReD/train_data.csv'
            evaluation_labels_path = 'data/fundus/MuReD/test_data.csv'
            training_images_dir = 'data/fundus/MuReD/images/images'
            evaluation_images_dir = 'data/fundus/MuReD/images/images'
            da_training_images_dir = 'data/fundus/MuReD/images/images'
            valid_labels_path = ''
            valid_images_dir = ''
        
    return num_classes, training_labels_path, evaluation_labels_path, training_images_dir, evaluation_images_dir, da_training_images_dir, normal_class_index, valid_labels_path, valid_images_dir


# Models
def get_model(model_name, transformer_layer, num_classes):
    if model_name == 'resnet':
        model = ResNet152(num_classes).to(device)
    elif model_name == 'densenet':
        model = DenseNet121(num_classes).to(device)
        # model = DenseNet161(num_classes).to(device)
    elif model_name == 'mobilenet':
        model = MobileNetV2(num_classes).to(device)
    elif model_name == 'efficientnet':
        model = EfficientNet_v2(num_classes).to(device)
    elif model_name == 'inception':
        model = InceptionV3(num_classes).to(device)
    elif model_name == 'vit':
        model = ViT(num_classes).to(device)
    elif model_name == 'ctran':
        model = CTranModel(num_labels=num_classes,use_lmt=True,pos_emb=False,layers=3,heads=4,dropout=0.1).to(device)
    elif model_name == 'swin':
        model = SwinTransformer(num_classes=num_classes).to(device)
    elif model_name == 'convnext':
        model = ConvNeXt(num_classes=num_classes).to(device)
    elif model_name == 'mydensenet4':
        model = myDenseNet4(num_classes).to(device)
    elif model_name == 'myconvnext':
        model = ConvNeXtTransformer(num_classes, num_transformer_layers=transformer_layer).to(device)
    elif model_name == 'myconvnext_concatGAP':
        model = ConvNeXtTransformer_concatGAP(num_classes, num_transformer_layers=transformer_layer).to(device)
    elif model_name == 'maxvit':
        model = MaxViT(num_classes=num_classes).to(device)
    elif model_name == 'coatnet':
        model = CoAtNet(num_classes=num_classes).to(device)
    elif model_name == 'add_gcn':
        model = ADD_GCN(num_classes=num_classes).to(device)
    elif model_name == 'q2l':
        model = build_q2l(num_class=num_classes).to(device)
    elif model_name == 'ml_decoder':
        model = create_model(num_classes=num_classes).to(device)
    elif model_name == 'tresnet':
        model = create_tresnet_model(num_classes=num_classes).to(device)
    elif model_name == 'mcar':
        model = mcar_resnet101(num_classes=num_classes,  ps='gwp', topN=4, threshold=0.5, pretrained=True).to(device)
    
    return model

# datasets
def get_dataset(num_classes, batch_size, training_labels_path, training_images_dir, da_training_images_dir, evaluation_labels_path, evaluation_images_dir, valid_labels_path, valid_images_dir):
    train_loader = None
    val_loader = None
    # train dataset
    train_dataset = MultilabelDataset(ann_dir=training_labels_path,
                                root_dir=training_images_dir,
                                num_labels=num_classes,
                                transform=transform, known_labels=1, testing=False, da_root_dir=da_training_images_dir)
    
    if valid_labels_path:
        valid_dataset = MultilabelDataset(ann_dir=valid_labels_path,
                                root_dir=valid_images_dir,
                                num_labels=num_classes,
                                transform=transform4test, known_labels=0, testing=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, prefetch_factor=prefetch_factor, num_workers=num_workers)
        val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, prefetch_factor=prefetch_factor, num_workers=num_workers)

    # test dataset
    test_dataset = MultilabelDataset(ann_dir=evaluation_labels_path,
                                root_dir=evaluation_images_dir,
                                num_labels=num_classes,
                                transform=transform4test, known_labels=0, testing=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, prefetch_factor=prefetch_factor, num_workers=num_workers)
    return train_dataset, test_dataset, test_loader, train_loader, val_loader


# trainset to train and validation (0.8, 0.2)   
def train(model, num_classes, train_dataset, train_loader, val_loader, learning_rate, batch_size, ctran_model=False, evaluation=False, weight_decay=False, warmup=False, loss='bce'):
    num_epochs = 15
    if weight_decay:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    # Loss function
    if model.__class__.__name__ == "MCARResnet":
        criterion = nn.BCELoss()
    elif loss == 'focal_loss':
        print("[Focal Loss]")
        criterion = FocalLoss()
    elif loss == 'asymmetric_loss':
        print("[Asymmetric Loss]")
        criterion = AsymmetricLossOptimized(gamma_neg=1, gamma_pos=0)
        # criterion = AsymmetricLossOptimized()
    elif loss == 'bce':
        print("[BCE Loss]")
        criterion = nn.BCEWithLogitsLoss(reduction='sum')
    elif loss == 'poly_ce':
        print("[Poly Loss (bce)]")
        criterion = Poly1CrossEntropyLoss(num_classes, reduction='sum')
    elif loss == 'poly_focal':
        print("[Poly Loss (Focal)]")
        criterion = Poly1FocalLoss(num_classes, reduction='sum')
    
    if warmup:
        num_epochs += 5
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=linear_warmup)
        
    step_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # step_scheduler = OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=10, epochs=num_epochs, div_factor=10, final_div_factor=100)
    
    
    if val_loader is None:
        if evaluation:
            # torch.manual_seed(13)
            total_size = len(train_dataset)
            val_size = int(total_size * 0.2)
            train_size = total_size - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
            
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
    else:
        evaluation = True

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
                # loss_out = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none').sum() # sigmoid + BCELoss
                if model.__class__.__name__ == "MCARResnet":
                    loss_out = criterion(outputs[0], labels) + criterion(outputs[1], labels)
                else:
                    loss_out = criterion(outputs, labels)
            
            train_loss += loss_out.item()
            loss_out.backward()
            optimizer.step()
            
        # scheduler.step()
        if epoch < 5:
            if warmup:
                # print(warmup_scheduler.get_last_lr())
                warmup_scheduler.step()
            else:
                step_scheduler.step()
                # print(step_scheduler.get_last_lr())
        else:
            step_scheduler.step()
            # print(step_scheduler.get_last_lr())

        if not evaluation:
            current_train_loss = train_loss / len(train_loader)
            if current_train_loss < best_train_loss:
                best_train_loss = current_train_loss
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
                    outputs = F.sigmoid(outputs)
                else:
                    inputs, labels = batch['image'].to(device), batch['labels'].to(device)
                    outputs = model(inputs)
                    # loss_out = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none').sum()
                    if model.__class__.__name__ == "MCARResnet":
                        loss_out = criterion(outputs[0], labels) + criterion(outputs[1], labels)
                        outputs  = torch.max(outputs[0], outputs[1])
                    else:
                        loss_out = criterion(outputs, labels)
                        outputs = F.sigmoid(outputs)
                    
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
                outputs_np = outputs.cpu().numpy()
                # outputs_np = outputs.cpu().numpy()
                labels_np = labels.cpu().numpy()
                all_preds.extend(outputs_np)
                all_labels.extend(labels_np)
                
                ## method 4. mAP
                all_preds_4.append(outputs.cpu())
                # all_preds_4.append(outputs.cpu())
                all_labels_4.append(labels.cpu())
                
                ## method 5. F1 Score
                predicted = outputs.cpu() > 0.5
                # predicted = outputs.cpu() > 0.5
                all_preds_5.append(predicted.numpy())
                all_labels_5.append(labels.cpu().numpy())

        current_val_loss = val_loss / len(val_loader)
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
        
        # if rfmid_ori:
        #     print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss/len(train_loader):.6f}, Validation Loss: {val_loss/len(val_loader):.6f}')
        #     continue
        
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
def evaluate(model, best_model_state, test_loader, results_path, evaluation_labels_path, dataset_name, normal_index=1, ctran_model=False, best_model=False):
    if best_model:
        print()
        print("~~~~~~~~~~ Best model evaluation ~~~~~~~~~~")
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
                
            if model.__class__.__name__ == "MCARResnet":
                outputs  = torch.max(outputs[0], outputs[1])
            else:
                outputs = F.sigmoid(outputs)
                
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
            outputs_np = outputs.cpu().numpy()
            # outputs_np = outputs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_preds.extend(outputs_np)
            all_labels.extend(labels_np)
            
            ## method 4. mAP
            all_preds_4.append(outputs.cpu())
            # all_preds_4.append(outputs.cpu())
            all_labels_4.append(labels.cpu())
            
            ## method 5. F1 Score
            predicted = outputs.cpu() > 0.5
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
    ## method 6. Overall precision and recall and F1
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(all_labels_5.ravel(), all_preds_5.ravel(), average='binary')
    overall_precision = round(overall_precision, 3)
    overall_recall = round(overall_recall, 3)
    overall_f1 = round(overall_f1, 3)
    # print(overall_precision, overall_recall, overall_f1)
    
    if dataset_name == 'rfmid':
        os.makedirs('results/rfmid', exist_ok=True)
    elif dataset_name == 'mured':
        os.makedirs('results/mured', exist_ok=True)
    avg_results = result2csv(results_path, evaluation_labels_path, precision_scores, recall_scores, f1_list, mAP_per_label, auc_scores, best_model)
    # print(f'Evaluation - Average Precision: {average_precision:.3f}, Average Recall: {average_recall:.3f}, F1_macro: {f1_macro:.3f}, mAP: {mAP:.3f}, Average AUC: {average_auc:.3f}, ML Scores: {(mAP + average_auc) / 2:.3f}')
    
    normal_auc = auc_scores.pop(normal_index)
    average_auc = sum(auc_scores) / len(auc_scores)
    normal_f1 = f1_list.pop(normal_index)
    f1_macro = sum(f1_list) / len(f1_list)
    mAP_per_label.pop(normal_index)
    mAP = sum(mAP_per_label) / len(mAP_per_label)
    ML_score = (mAP + average_auc) / 2
    eval_results = [f1_macro, mAP, average_auc, ML_score, normal_f1, normal_auc, (ML_score + normal_auc) / 2]
    eval_results = [str(round(result, 3)) for result in eval_results]
    results2allcsv(results_path, eval_results, avg_results, dataset_name, overall_precision, overall_recall, overall_f1, best_model)
    print(f'===== Evaluation results =====')
    print(f'OP: {overall_precision}, OR: {overall_recall}, OF1: {overall_f1}, CP: {avg_results[0]}, CR: {avg_results[1]}, CF1: {avg_results[2]}, mAP: {avg_results[3]}, Average AUC: {avg_results[4]}')
    print(f'ML_F1: {f1_macro:.3f}, ML_mAP: {mAP:.3f}, ML_AUC: {average_auc:.3f}, ML_Score: {ML_score:.3f}, Bin_F1: {normal_f1:.3f}, Bin_AUC: {normal_auc:.3f}, Model_Score: {(ML_score + normal_auc) / 2:.3f}')
    # plot_auc_curve(all_preds, all_labels, evaluation_labels_path, auc_fig_path)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='myconvnext_concatGAP', help='Model name')
    parser.add_argument('--save_results_path', type=str, default='results/myconvnext_concatGAP.csv', help='Path to save the evaluation results')
    parser.add_argument('--ctran_model', action='store_true', help='Use CTran model')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--val', action='store_true', help='split the training set into training and validation')
    parser.add_argument('--transformer_layer', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--dataset', type=str, default='mured', help='Dataset name: mured or rfmid')
    parser.add_argument('--weight_decay', action='store_true')
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--data_aug', type=str, default=None, help='Data augmentation methods or None')
    parser.add_argument('--plm', action='store_true', help='Partial Label Masking training')
    parser.add_argument('--loss', type=str, default='bce', help='Loss function: bce, focal, asymmetric')
    # parser.add_argument('--normal_class', type=int, default=1, help='Normal class index')
    # parser.add_argument('--training_labels_path', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    num_classes, training_labels_path, evaluation_labels_path, training_images_dir, evaluation_images_dir, da_training_images_dir, normal_class_index, valid_labels_path, valid_images_dir = dataset2train(args.dataset, args.data_aug)
    train_dataset, test_dataset, test_loader, train_loader, val_loader = get_dataset(num_classes=num_classes, batch_size=args.batch_size, training_labels_path=training_labels_path, training_images_dir=training_images_dir, da_training_images_dir=da_training_images_dir, evaluation_labels_path=evaluation_labels_path, evaluation_images_dir=evaluation_images_dir, valid_labels_path=valid_labels_path, valid_images_dir=valid_images_dir)
    model = get_model(args.model, args.transformer_layer, num_classes)
    print(f"===== Model: {model.__class__.__name__} =====")
    print(f"<training_labels_path: {training_labels_path}>")
    print("******************** Training   ********************")
    if args.plm:
        best_model_state = train_plm(model, train_dataset, args.lr, ctran_model=args.ctran_model, warmup=args.warmup, evaluation=args.val, num_classes=num_classes, batch_size=args.batch_size, prefetch_factor=prefetch_factor, num_workers=num_workers, device=device, loss=args.loss)
    else:
        best_model_state = train(model, num_classes, train_dataset, train_loader, val_loader, args.lr, batch_size=args.batch_size, ctran_model=args.ctran_model, evaluation=args.val, weight_decay=args.weight_decay, warmup=args.warmup, loss=args.loss)
    # best_model_state = train_kfold(model, train_dataset, args.lr, ctran_model=args.ctran_model)
    print("******************** Testing ********************")
    evaluate(model, best_model_state, test_loader, args.save_results_path, evaluation_labels_path, args.dataset, normal_index=normal_class_index, ctran_model=args.ctran_model)
    evaluate(model, best_model_state, test_loader, args.save_results_path, evaluation_labels_path, args.dataset, normal_index=normal_class_index, ctran_model=args.ctran_model, best_model =True)
