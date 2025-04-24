import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model import ClassicalModel
from preprocessing_for_bot import Tokenize



device = "cuda" if torch.cuda_is_avaliable() else "cpu"

df = pd.read_csv("data\cut_data_for_study.csv")
max_len = np.max(df['text_len'])

vocabulary, tokenized_column = Tokenize(df["text_clean"], max_len)

EMBEDDING_DIM = 200
VOCAB_SIZE = len(vocabulary) + 1
BATCH_SIZE = 32
hidden_dim = 32
NUM_CLASSES = 2
num_epochs = 3


X = tokenized_column
y = df['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
train_loader = DataLoader(X_train, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
test_loader = DataLoader(X_test, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)


def train(model, criterion, optimizer,
          train_dataloader, test_dataloader, num_epochs):

    train_losses = np.zeros(num_epochs)
    test_losses = np.zeros(num_epochs)

    train_accuracy_arr = np.zeros(num_epochs)
    test_accuracy_arr = np.zeros(num_epochs)

    for i_epoch in range(num_epochs):
        it = 0
        train_loss = 0
        test_loss = 0

        train_accuracy = 0
        test_accuracy = 0

        # train step
        model.train()
        for batch in train_dataloader:
            X = batch[0]
            y = batch[1]

            h = model.init_hidden(y.size(0))
            preds, h = model(X, h)

            optimizer.zero_grad()
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().cpu().numpy()
            # print(f'batch: {it+1}/{len(train_dataloader)}, loss: {train_loss/(it+1):.4f}, time: {execution_time:.4f}')
            it += 1
            train_accuracy += (preds.argmax(-1).detach() == y).cpu().numpy().mean()


        train_loss /= len(train_dataloader)
        train_accuracy /= len(train_dataloader)
        train_losses[i_epoch] = train_loss
        train_accuracy_arr[i_epoch] = train_accuracy

        model.eval()
        for batch in test_dataloader:
            X = batch[0]
            y = batch[1]

            h = model.init_hidden(y.size(0))

            with torch.no_grad():
                preds, h = model(X, h)
                loss = criterion(preds, y)

                test_loss += loss.detach().cpu().numpy()
                test_accuracy += (preds.argmax(-1) == y).cpu().numpy().mean()


        test_loss /= len(test_dataloader)
        test_accuracy /= len(test_dataloader)

        test_losses[i_epoch] = test_loss
        test_accuracy_arr[i_epoch] = test_accuracy

    return train_losses, test_losses, train_accuracy_arr, test_accuracy_arr

criterion = nn.NLLLoss()

model = ClassicalModel(VOCAB_SIZE, EMBEDDING_DIM, hidden_dim, NUM_CLASSES, LSTM_LAYERS=1)
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
train_losses, _, _, test_accuracy_arr = train(model, criterion=criterion,
                        optimizer=optimizer,
                        train_dataloader=train_loader,
                        test_dataloader=test_loader,
                        num_epochs=num_epochs)

save_dir = 'models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

weights_path = os.path.join(save_dir, 'cl_lstm_weights.pth')

torch.save(model.state_dict(), weights_path)