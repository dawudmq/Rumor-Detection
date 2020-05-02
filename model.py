from torch import nn
from torch.functional import F
import torch
import math


class Attention(nn.Module):  # output: tuple: (context, attn)
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, X, X_padding_mask=None, coverage=None, dropout=0.1):
        """
        K / key: (L, B, H) encoder_outputs, encoder feature
        V / value: (L, B, H) to calculate the context vector
        Q / query: (L, B, H) last_hidden, deocder feature
        X_padding_mask: (B, 1, L)
        coverage: (B, L)
        """
        X_dim = X.size(-1)
        X_query = X.transpose(0, 1)  # -> (B, L, H)
        X_key = X.transpose(0, 1)  # -> (B, L, H)
        X_value = X.transpose(0, 1)  # -> (B, L, H)

        scores = torch.matmul(X_query, X_key.transpose(-2, -1)) / math.sqrt(X_dim)  # (B, L, H) x (B, H, L) -> (B, L, L)

        attn_dist = F.softmax(scores, dim=-1)  # (B, L, L)
        attn_dist = F.dropout(attn_dist, p=dropout)
        context = torch.matmul(attn_dist, X_value)  # (B, L, L) x (B, L, H) -> (B, L, H)

        # calculate average
        context = context.sum(1)/context.size(1)
        return context, attn_dist

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.embedding_dim=300
        self.hidden_dim=300
        self.batch_size=1

        self.content = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim, bidirectional=True,num_layers=1)
        self.content_linear = nn.Linear(self.hidden_dim*2,self.hidden_dim)
        self.output_linear = nn.Linear(1 * self.batch_size * self.hidden_dim, 1)
        self.attention = Attention()

    def forward(self, event):
        multi_view = torch.tensor([])

        X_content_data,label =event[0],event[1]
        #X_content_data = X_content_data.unsqueeze(1)

        output_content, hidden_content = self.content(X_content_data.view(1,-1,300))
        output_content = self.content_linear(output_content)
        output_content, attn_dist_content = self.attention(output_content.transpose(0,1))
        output_content = output_content.flatten()
        multi_view = torch.cat((multi_view, output_content))
        multi_view = self.output_linear(multi_view)
        result = torch.sigmoid(multi_view)
        loss = (label - result)**2
        return loss, result