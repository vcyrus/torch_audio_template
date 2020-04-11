import torch
import torch.nn as nn

import numpy as np

from torch.utils.data import DataLoader
   
def evaluate(model, device, dataset, label_to_int, criterion):
    model.eval()

    int_to_label = {i: label for label, i in label_to_int.items()}

    batch_size = 32
    generator = DataLoader(dataset, batch_size=batch_size)

    total = 0
    correct = 0
    class_correct = list(0. for i in range(len(label_to_int)))
    class_total   = list(0. for i in range(len(label_to_int)))

    for i, batch in enumerate(generator, 0):
        model.zero_grad()
        X, y = batch["audio"], batch["label"]         
        X = X.to(device=device)                 
        y = y.numpy()

        outputs = model(X)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()

        total += y.shape[0]
        pairwise_eq = (preds == y)
        correct += np.sum(pairwise_eq)
        
        for i, p in np.ndenumerate(preds):
            label = y[i]
            class_correct[label] += pairwise_eq[i]
            class_total[label] += 1
            
    print('Accuracy of network on %d test samples: %d %%' % (total, 
        100 * correct / total))
    for i in range(len(int_to_label)):
        print('Accuracy of %5s : %2d %%' % (
            int_to_label[i], 100 * class_correct[i] / class_total[i]))
