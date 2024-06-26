import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, average_precision_score, precision_score, recall_score
from sklearn.model_selection import KFold
import copy
from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd
import csv
from torch.optim.lr_scheduler import StepLR, LambdaLR
from tqdm import tqdm 
from collections import Counter, defaultdict

from models.utils import custom_replace
from loss_functions.asymmetric import AsymmetricLossOptimized
from loss_functions.focal import FocalLoss
eps = np.finfo(float).eps

def count_labels(dataset, num_classes):
    label_counts = defaultdict(int)
    for data in dataset:
        labels = data['labels']
        for label in range(num_classes):
            if labels[label] == 1:
                label_counts[label] += 1
    return label_counts

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()
    
def plot_auc_curve(all_preds, all_labels, evaluation_labels_path, auc_fig_path):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    n_classes = all_labels.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    # colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive']
    class_names = pd.read_csv(evaluation_labels_path).columns.tolist()[1:]
    for i, cls_name in zip(range(n_classes), class_names):
        plt.plot(fpr[i], tpr[i], lw=2,
                label=f'{cls_name} ({roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-label ROC')
    plt.legend(loc="lower right")
    plt.savefig(auc_fig_path)
    
def result2csv(results_path, evaluation_labels_path, precision_list, recall_list, f1_list, ap_list, auc_list, best_model):
    if best_model:
        results_path = results_path.replace('.csv', '_best.csv')
        
    print(f"[Writing each label results to {results_path}]")
    class_names = pd.read_csv(evaluation_labels_path).columns.tolist()[1:]
    with open(results_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['diseases', 'Precision', 'Recall', 'F1', 'AP', 'AUC'])
        for class_name, precision, recall, f1, ap, auc in zip(class_names, precision_list, recall_list, f1_list, ap_list, auc_list):
            formatted_class_name = f"{class_name:<8}"  
            formatted_precision = f"{precision:.3f}".ljust(8)
            formatted_recall = f"{recall:.3f}".ljust(8)
            formatted_f1 = f"{f1:.3f}".ljust(8)
            formatted_ap = f"{ap:.3f}".ljust(8)
            formatted_auc = f"{auc:.3f}".ljust(8)
            writer.writerow([formatted_class_name, formatted_precision, formatted_recall, formatted_f1, formatted_ap, formatted_auc])
            
        avg_precision = sum(precision_list) / len(precision_list)
        avg_recall = sum(recall_list) / len(recall_list)
        avg_f1 = sum(f1_list) / len(f1_list)
        avg_ap = sum(ap_list) / len(ap_list)
        avg_auc = sum(auc_list) / len(auc_list)
        
        writer.writerow([
            f"Average".ljust(8),
            f"{avg_precision:.3f}".ljust(8),
            f"{avg_recall:.3f}".ljust(8),
            f"{avg_f1:.3f}".ljust(8),
            f"{avg_ap:.3f}".ljust(8),
            f"{avg_auc:.3f}".ljust(8)
        ])
    
    avg_results = [avg_precision, avg_recall, avg_f1, avg_ap, avg_auc]
    avg_results = [str(round(result, 3)) for result in avg_results]
    return avg_results
        
def results2allcsv(results_path, all_results, avg_results, dataset_name, overall_precision, overall_recall, overall_f1, best_model):
    if dataset_name == 'mured':
        csv_file_path = 'results/all_models_results_mured.csv'
    elif dataset_name == 'rfmid':
        csv_file_path = 'results/all_models_results_rfmid.csv'
        
    if best_model:
        csv_file_path = csv_file_path.replace('.csv', '_best.csv')
    
    if not os.path.exists(csv_file_path):
        column_names = ['name', 'OP', 'OR', 'OF1', 'CP', 'CR', 'CF1', 'mAP', 'Avg_AUC', 'ML_F1',  'ML_mAP', 'ML_AUC', 'ML_Score', 'Bin_F1', 'Bin_AUC', 'Model_Score']
        df = pd.DataFrame(columns=column_names)
        df.to_csv(csv_file_path, index=False)
    
    overall_results = [overall_precision ,overall_recall ,overall_f1]
    filename = results_path.split('/')[-1]
    basename = filename.split('.')[0]
    result_with_name = [basename] + overall_results + avg_results + all_results
    
    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result_with_name)
        
    print(f"[Evaluation all results has been written to *{csv_file_path}*]")

def linear_warmup(epoch):
    if epoch < 5:
        return epoch * 5e-2
    return 1.0

# Kfold cross validation (k=5)
def train_kfold(model, train_dataset, learning_rate, ctran_model=False, batch_size=32, prefetch_factor=2, num_workers=4, device='cuda', loss_labels='unk'):
    print(f"[Training with KFold cross validation]")
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
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
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

def get_positive_ratio(dataset) -> np.ndarray:
    y = get_y_true(dataset)
    n_samples = len(y)
    n_positives = np.sum(y > 0, axis=0)
    positive_ratio = n_positives / n_samples
    # print(f"Positive ratio: {positive_ratio}")
    return positive_ratio

def get_y_true(dataset):
    y = list(map(lambda x: x['labels'], dataset))
    return np.concatenate(y, axis=0)

class RandomMultiHotGenerator:
    def __init__(self, seed=None):
        super(RandomMultiHotGenerator, self).__init__()
        self.rng = np.random.default_rng(seed)

    def generate(self, prob: np.ndarray) -> np.ndarray:
        """
        Generate multi-hot vector where some elements are 1 with a certain probability
        *prob* and the others is 0.

        Args:
            prob: probability. [0, 1]

        Returns:
            ones_with_probability (np.ndarray): whose shape is the same as *prob*
        """
        return np.where(
            self.rng.uniform(low=0.0, high=1.0, size=prob.shape) <= prob, 1, 0
        ).astype(np.int_)


class MaskGenerator:
    def __init__(self, generator: Optional[RandomMultiHotGenerator] = None):
        if generator is None:
            generator = RandomMultiHotGenerator()

        self.generator = generator

    def generate(self, y_true, positive_ratio, positive_ratio_ideal):
        y_true = y_true.numpy()
        is_head_class = np.full(
            y_true.shape, positive_ratio > positive_ratio_ideal, np.bool_
        )
        is_tail_class = np.logical_not(is_head_class)

        mask_for_head_class = self.generator.generate(
            np.full(y_true.shape, positive_ratio_ideal / positive_ratio)
        )
        mask_for_tail_class = self.generator.generate(
            np.full(y_true.shape, positive_ratio / positive_ratio_ideal)
        )

        mask = np.ones_like(y_true)
        mask = np.where(is_head_class & (y_true > 0), mask_for_head_class, mask)
        mask = np.where(is_tail_class & (y_true == 0), mask_for_tail_class, mask)

        return mask


class ProbabilityHistograms:
    def __init__(self, n_classes: int, n_bins: int = 10):
        self.n_classes = n_classes
        self.n_bins = n_bins

        dtype = np.int_
        self.ground_truth_positive = np.zeros((self.n_bins, self.n_classes), dtype)
        self.prediction_positive = np.zeros((self.n_bins, self.n_classes), dtype)
        self.ground_truth_negative = np.zeros((self.n_bins, self.n_classes), dtype)
        self.prediction_negative = np.zeros((self.n_bins, self.n_classes), dtype)

    def reset(self):
        dtype = np.int_
        self.ground_truth_positive = np.zeros((self.n_bins, self.n_classes), dtype)
        self.prediction_positive = np.zeros((self.n_bins, self.n_classes), dtype)
        self.ground_truth_negative = np.zeros((self.n_bins, self.n_classes), dtype)
        self.prediction_negative = np.zeros((self.n_bins, self.n_classes), dtype)

    def update_histogram(self, y_true: np.ndarray, y_pred: np.ndarray):
        value_range = [0.0, 1.0]

        for class_i in range(self.n_classes):
            y_true_class = y_true[:, class_i]
            y_pred_class = y_pred[:, class_i]

            pos_indices = y_true_class > 0
            neg_indices = ~pos_indices

            # NOTE: np.histogram returns *hist* and *bin_edges*
            ground_truth_positive, _ = np.histogram(
                y_true_class[pos_indices], self.n_bins, value_range
            )
            self.ground_truth_positive[:, class_i] += ground_truth_positive
            
            ground_truth_negative, _ = np.histogram(
                y_true_class[neg_indices], self.n_bins, value_range
            )
            self.ground_truth_negative[:, class_i] += ground_truth_negative

            prediction_positive, _ = np.histogram(
                y_pred_class[pos_indices], self.n_bins, value_range
            )
            self.prediction_positive[:, class_i] += prediction_positive

            prediction_negative, _ = np.histogram(
                y_pred_class[neg_indices], self.n_bins, value_range
            )
            self.prediction_negative[:, class_i] += prediction_negative

    def divergence_difference(self):
        divergence_positive = self._divergence_between_histograms(
            self.prediction_positive, self.ground_truth_positive
        )
        divergence_negative = self._divergence_between_histograms(
            self.prediction_negative, self.ground_truth_negative
        )

        divergence_positive = self._standardize_among_classes(divergence_positive)
        divergence_negative = self._standardize_among_classes(divergence_negative)

        return divergence_positive - divergence_negative

    @staticmethod
    def _divergence_between_histograms(hist_pred, hist_true):
        # normalize histogram
        hist_true = hist_true / (np.sum(hist_true, axis=0) + eps)
        hist_pred = hist_pred / (np.sum(hist_pred, axis=0) + eps)

        kl_div = kullback_leibler_divergence(hist_pred, hist_true)
        return kl_div

    @staticmethod
    def _standardize_among_classes(x):
        return (x - np.mean(x)) / (np.std(x) + eps)

def kullback_leibler_divergence(p, q):
    """
    Kullback-Leibler divergence.

    Args:
        p, q: discrete probability distributions, whose shape is (n_bins, n_classes)

    Returns:
        kl_div: Kullback-Leibler divergence (relative entropy from q to p)
    """
    q = np.where(q > 0, q, eps)
    kl_div = np.sum(p * np.log(p / q + eps), axis=0)
    return kl_div

# train with Partial Label Masking
def train_plm(model, train_dataset, learning_rate, ctran_model=False, loss_labels='all', warmup=False, evaluation=False, num_classes=20, batch_size=16, prefetch_factor=64, num_workers=28, device='cuda', loss='bce'):
    print(f"[Training with Partial Label Masking]")
    num_epochs = 35
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01) # for transformers
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Loss function
    if loss == 'focal_loss':
        criterion = FocalLoss()
    elif loss == 'asymmetric_loss':
        criterion = AsymmetricLossOptimized()
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='sum')
    
    if warmup:
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=linear_warmup)
    step_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    if evaluation:
        # torch.manual_seed(13)
        total_size = len(train_dataset)
        val_size = int(total_size * 0.2)
        train_size = total_size - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(146))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, prefetch_factor=prefetch_factor, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, prefetch_factor=prefetch_factor, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, prefetch_factor=prefetch_factor, num_workers=num_workers)
    
    change_rate = 1
    positive_ratio = get_positive_ratio(train_loader).astype(np.float32)
    ideal_positive_ratio = copy.deepcopy(positive_ratio)
    hist = ProbabilityHistograms(n_classes=num_classes, n_bins=4)
    mask_generator = MaskGenerator(
        generator=RandomMultiHotGenerator(seed=146)
    )

    best_train_loss = float('inf')
    best_val_loss = float('inf')
    best_model_state = None
    for epoch in tqdm(range(num_epochs), desc='Epoch'):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            plm_mask = mask_generator.generate(
                batch['labels'], positive_ratio, ideal_positive_ratio
            )
            # print(plm_mask)
            plm_mask = torch.from_numpy(plm_mask).to(device)
            
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
                # loss_out = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none') # sigmoid + BCELoss
                loss_out = criterion(outputs, labels)
            
            hist.update_histogram(batch['labels'], outputs.sigmoid().detach().cpu().numpy())
            loss_out = (loss_out * plm_mask).sum()
            train_loss += loss_out.item()
            loss_out.backward()
            optimizer.step()
        
        divergence_difference = hist.divergence_difference()
        ideal_positive_ratio *= np.exp(change_rate * divergence_difference)
        hist.reset()
        
        if epoch < 5:
            if warmup:
                print(warmup_scheduler.get_last_lr())
                warmup_scheduler.step()
            else:
                print(step_scheduler.get_last_lr())
                step_scheduler.step()
        else:
            print(step_scheduler.get_last_lr())
            step_scheduler.step()  

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
                    # loss_out = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none').sum()
                    loss_out = criterion(outputs, labels)
                    
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


