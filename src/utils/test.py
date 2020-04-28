import torch
import torch.nn as nn

import numpy as np

from sklearn.metrics import confusion_matrix, classification_report

from torch.utils.data import DataLoader

from src.datasets.SOLDataset import SOLDataset
   
def evaluate(model, device, dataset, label_to_int, criterion):
    print("Evaluating Model Accuracy ...")
    model.eval()

    batch_size = 32
    generator = DataLoader(dataset, batch_size=batch_size)

    total = 0
    correct = 0
    class_correct = list(0. for i in range(len(label_to_int)))
    class_total   = list(0. for i in range(len(label_to_int)))

    y_true = []
    y_pred = []
    for i, batch in enumerate(generator, 0):
        model.zero_grad()

        # this code is horribly hacked and need refactoring after this
        X, y = batch["audio"], batch["label"]         
        X = X.to(device=device)                 
        y = y.numpy()
        outputs = model(X)                            
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()
    
        total += y.shape[0]
        pairwise_eq = (preds == y)
        correct += np.sum(pairwise_eq)
    
        y_true += list(y)
        y_pred += list(preds)
            
        # classwise accuracy
        for i, p in np.ndenumerate(preds):
            label = y[i]
            class_correct[label] += pairwise_eq[i]
            class_total[label] += 1

    # get confusion matrix
    labels = list(label_to_int.values())
    int_to_label = {v: k for k, v in label_to_int.items()}

    confusions = confusion_matrix(y_true, 
                                  y_pred, 
                                  labels)

    report = classification_report([int_to_label[y] for y in y_true],
                                    [int_to_label[y] for y in y_pred])

    print(report)

    '''
    print('Accuracy of network on %d test samples: %d %%' % (total, 
        100 * correct / total))
    for i in range(len(int_to_label)):
        print('Accuracy of %5s : %2d %%' % (
            int_to_label[i], 100 * class_correct[i] / class_total[i]))
    '''

    return confusions
