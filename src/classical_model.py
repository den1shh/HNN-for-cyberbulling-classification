import torch
import torch.nn as nn
import pandas as pd


from transformers import DistilBertTokenizer, DistilBertModel

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden = hidden[-1]
        hidden_repeated = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        attn_weights = torch.tanh(self.attn(torch.cat((hidden_repeated, encoder_outputs), dim=2)))
        attn_weights = self.v(attn_weights).squeeze(2)
        return nn.functional.softmax(attn_weights, dim=1)

class CyberbullyingDetector(nn.Module):
    def __init__(self, hidden_size, num_classes, pretrained_model_name='distilbert-base-uncased'):
        super(CyberbullyingDetector, self).__init__()
        self.bert = DistilBertModel.from_pretrained(pretrained_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(input_size=768,
                            hidden_size=hidden_size,
                            num_layers=1, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        out, hidden = self.lstm(sequence_output)
        attn_weights = self.attention(hidden[0], out)
        context = attn_weights.unsqueeze(1).bmm(out).squeeze(1)
        logits = self.softmax(self.fc(context))
        return logits