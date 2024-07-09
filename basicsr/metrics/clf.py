import torch
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_f1(predictions, labels,  **kwargs):
    predictions = torch.sigmoid(predictions) > 0.5
    # print(predictions, labels)
    predictions = predictions.int()
    
    tp = (labels * predictions).sum(dim=0)
    fp = ((1 - labels) * predictions).sum(dim=0)
    fn = (labels * (1 - predictions)).sum(dim=0)
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    return f1.mean()

@METRIC_REGISTRY.register()
def calculate_precision(predictions, labels,  **kwargs):
    predictions = torch.sigmoid(predictions) > 0.5
    # print(predictions, labels)
    predictions = predictions.int()
    
    tp = (labels * predictions).sum(dim=0)
    fp = ((1 - labels) * predictions).sum(dim=0)
    
    precision = tp / (tp + fp + 1e-7)
    return precision.mean()

@METRIC_REGISTRY.register()
def calculate_recall(predictions, labels,  **kwargs):
    predictions = torch.sigmoid(predictions) > 0.5
    # print(predictions, labels)
    predictions = predictions.int()
    
    tp = (labels * predictions).sum(dim=0)
    fn = (labels * (1 - predictions)).sum(dim=0)
    
    recall = tp / (tp + fn + 1e-7)

    return recall.mean()

# # Example true labels and predicted labels
# true_labels = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
# predicted_labels = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1]])

# # Calculate F1-score per label
# f1_scores = f1_score(true_labels, predicted_labels)
# print("F1-Scores per Label:", f1_scores)

# # Calculate micro-averaged F1-score
# micro_f1 = f1_score(true_labels, predicted_labels).mean()
# print("Micro-Averaged F1-Score:", micro_f1)

# # Calculate macro-averaged F1-score
# macro_f1 = f1_scores.mean()
# print("Macro-Averaged F1-Score:", macro_f1)
