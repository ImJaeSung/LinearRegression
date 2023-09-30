import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import torch.optim as optim
from data_utils import x_train, y_train, x_test
from data_utils import CustomDataset
from model import RegressionModel

submission = pd.read_csv('../dataset/sample_submission.csv')
submission['SalePrice'] = 0

epochs = 3000
best_loss = 10**8
patience_check = 0
patience_limit = 3

K = 5
random_seed = 42
cross = KFold(n_splits = K, shuffle = True, random_state = random_seed)
fold = []
valid_losses = []

for train_idx, valid_idx in cross.split(x_train, y_train):
    fold.append((train_idx, valid_idx))

for i in range(len(fold)):
    print(i)
    train_idx, valid_idx = fold[i]
    X_train = x_train.iloc[train_idx]
    Y_train = y_train[train_idx]
    X_valid = x_train.iloc[valid_idx]
    Y_valid = y_train[valid_idx]

    train_dataset = CustomDataset(X_train, Y_train)
    train_dataloader = DataLoader(train_dataset, batch_size = X_train.shape[0], shuffle = True)       
    valid_dataset = CustomDataset(X_valid, Y_valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size = X_valid.shape[0])
    
    model = RegressionModel([288, 512, 1024, 1024, 512, 1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', min_lr = 0.00001, 
                                                       threshold = 0.01, factor = 0.1, patience = 3)
    
    train_loss = []
    valid_loss = []

    for epoch in range(epochs + 1):
        model.train()
        for batch_idx, samples in enumerate(train_dataloader):
            train_x, train_y = samples
            
            pred = model(train_x)
            pred = torch.log(1 + pred)

            loss = criterion(pred, train_y)
            train_loss.append(loss.item())

            if epoch % 500 == 0:
                print('Epoch {:4d}/{} train_Loss : {:.6f}'.format(
                    epoch, epochs, loss
                ))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            
            for batch_idx, samples in enumerate(valid_dataloader):
                valid_x, valid_y = samples
                preds = model(valid_x)
                preds = torch.log(1 + preds)

                val_loss = criterion(preds, valid_y)
                valid_loss.append(val_loss.item())
                # scheduler.step(val_loss)
                
                if epoch % 500 == 0:
                    print('Epoch{:4d}/{} valid_Loss : {:.6f}'.format(
                    epoch, epochs, val_loss
                    ))

            if val_loss > best_loss:
                patience_check +=1
                
                if patience_check >= patience_limit:
                    break
            else:
                best_loss = val_loss
                patience_check = 0
                best_model = copy.deepcopy(model)
    
    valid_losses.append(valid_loss)

    print('best_epoch: {} best_valid_loss: {:.6f}'.format(
        np.argmin(valid_loss), np.min(valid_loss)))
    
    preds = best_model(torch.FloatTensor(x_test.values)).detach().squeeze(dim = 1).numpy()
    submission['SalePrice'] += preds/len(fold)