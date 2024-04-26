import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, average_precision_score, precision_score, recall_score
from sklearn.model_selection import KFold
import copy

from models.utils import custom_replace


def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

# Kfold cross validation (k=5)
def train_kfold(model, train_dataset, ctran_model=False, batch_size=32, prefetch_factor=2, num_workers=4, device='cuda', loss_labels='unk'):
    num_epochs = 10
    best_val_loss = float('inf')
    best_model_state = None

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(train_dataset)))):
        print(f"----- Fold {fold + 1}/{kf.get_n_splits()} -----")
        
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, prefetch_factor=prefetch_factor, num_workers=num_workers)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler, prefetch_factor=prefetch_factor, num_workers=num_workers)
        
        if ctran_model:
            optimizer = optim.Adam(model.parameters(), lr=0.00001)
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
        model.apply(reset_weights)
        
        for epoch in range(num_epochs):
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
                    outputs,int_pred,attns = model(images.to(device),mask_in.to(device))
                    
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
                    # print(outputs)
                    loss_out = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none').sum() # sigmoid + BCELoss
                    
                train_loss += loss_out.item()
                loss_out.backward()
                optimizer.step()

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
                        # print(outputs.shape, labels.shape)
                        loss_out = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none').sum() # sigmoid + BCELoss
                        
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
                    # outputs_np = F.sigmoid(outputs).cpu().numpy()
                    outputs_np = outputs.cpu().numpy()
                    labels_np = labels.cpu().numpy()
                    all_preds.extend(outputs_np)
                    all_labels.extend(labels_np)
                    
                    ## method 4. mAP
                    # all_preds_4.append(F.sigmoid(outputs).cpu())
                    all_preds_4.append(outputs.cpu())
                    all_labels_4.append(labels.cpu())
                    
                    ## method 5. F1 Score
                    # predicted = F.sigmoid(outputs).cpu() > 0.5
                    predicted = outputs.cpu() > 0.5
                    all_preds_5.append(predicted.numpy())
                    all_labels_5.append(labels.cpu().numpy())


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
            
            current_val_loss = val_loss / len(val_loader)
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
            
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss/len(train_loader):.6f}, Validation Loss: {val_loss/len(val_loader):.6f}, F1_macro: {f1_macro:.3f}, Average AUC: {average_auc:.3f}, mAP: {mAP:.3f}')
    return best_model_state