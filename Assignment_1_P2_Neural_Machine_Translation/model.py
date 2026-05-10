import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class bilstm_encoder(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, num_layers, hidden_dim, pretrained_embeddings
    ):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.embeddings.weight.data.copy_(pretrained_embeddings)

        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_hidden = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.fc_cell = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):
        x = self.embeddings(x)
        outputs, (hidden, cell) = self.bilstm(x)

        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        cell = torch.cat((cell[-2], cell[-1]), dim=1)

        hidden = torch.tanh(self.fc_hidden(hidden))
        cell = torch.tanh(self.fc_cell(cell))

        return outputs, hidden, cell


class bert_encoder(nn.Module):
    def __init__(self, hidden_dim, freeze_bert=True):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")

        for param in self.bert.parameters():
            param.requires_grad = not freeze_bert

        self.encoder_proj = nn.Sequential(
            nn.Linear(768, hidden_dim * 2), nn.LayerNorm(hidden_dim * 2)
        )
        self.fc_hidden = nn.Sequential(
            nn.Linear(768, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.fc_cell = nn.Sequential(
            nn.Linear(768, hidden_dim), nn.LayerNorm(hidden_dim)
        )

    def forward(self, input_ids, attention_mask):
        bert_embed = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        cls_out = bert_embed[:, 0, :]
        encoder_out = self.encoder_proj(bert_embed)

        hidden = torch.tanh(self.fc_hidden(cls_out))
        cell = torch.tanh(self.fc_cell(cls_out))

        return encoder_out, hidden, cell


class bahdanau_attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.v_out = nn.Linear(hidden_dim, 1)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, decoder_hidden, encoder_out, enc_mask):
        enc_len = encoder_out.size(1)
        dec_exp = decoder_hidden.unsqueeze(1).expand(-1, enc_len, -1)

        energy = torch.tanh(
            self.layer_norm(self.attention(torch.cat((dec_exp, encoder_out), dim=2)))
        )
        attention_scores = self.v_out(energy).squeeze(2)

        if enc_mask is not None:
            enc_mask = enc_mask.bool()
            attention_scores = attention_scores.masked_fill(~enc_mask, float("-inf"))

        return F.softmax(attention_scores, dim=1)


class lstm_decoder_vanilla(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, hidden_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim + hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, cell, encoder_out):
        x_embed = self.embeddings(x.unsqueeze(1))
        context = encoder_out.mean(dim=1, keepdim=True)
        lstm_input = torch.cat((x_embed, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell


class lstm_decoder_attention(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, hidden_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.attention = bahdanau_attention(hidden_dim=hidden_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim + hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim + hidden_dim * 2, vocab_size)

    def forward(self, x, hidden, cell, encoder_out, src_mask):
        x_embed = self.embeddings(x.unsqueeze(1))
        attn_weights = self.attention(hidden[-1], encoder_out, src_mask)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_out)

        lstm_input = torch.cat((x_embed, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        prediction = self.fc_out(torch.cat((output, context), dim=2).squeeze(1))
        return prediction, hidden, cell, attn_weights


class seq2seq_vanilla(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        pretrained_embeddings,
    ):
        super().__init__()
        self.encoder = bilstm_encoder(
            input_vocab_size, embed_dim, num_layers, hidden_dim, pretrained_embeddings
        )
        self.decoder = lstm_decoder_vanilla(
            output_vocab_size, embed_dim, num_layers, hidden_dim
        )
        self.num_layers = num_layers

    def forward(self, src, trg, teacher_forcing_ratio=0.6):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.fc_out.out_features

        encoder_outputs, hidden, cell = self.encoder(src)
        hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell = cell.unsqueeze(0).repeat(self.num_layers, 1, 1)

        use_tf = torch.rand(1).item() < teacher_forcing_ratio

        if use_tf:
            x_embed = self.decoder.embeddings(trg[:, :-1])
            context = encoder_outputs.mean(dim=1, keepdim=True).expand(
                -1, trg_len - 1, -1
            )
            lstm_input = torch.cat((x_embed, context), dim=2)
            lstm_out, _ = self.decoder.lstm(lstm_input, (hidden, cell))
            outputs = self.decoder.fc_out(lstm_out)
        else:
            outputs = torch.zeros(
                batch_size, trg_len - 1, vocab_size, device=src.device
            )
            input_token = trg[:, 0]
            for t in range(1, trg_len):
                output, hidden, cell = self.decoder(
                    input_token, hidden, cell, encoder_outputs
                )
                outputs[:, t - 1] = output
                input_token = output.argmax(dim=1)

        return outputs


class seq2seq_attention(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        pretrained_embeddings,
    ):
        super().__init__()
        self.encoder = bilstm_encoder(
            input_vocab_size, embed_dim, num_layers, hidden_dim, pretrained_embeddings
        )
        self.decoder = lstm_decoder_attention(
            output_vocab_size, embed_dim, num_layers, hidden_dim
        )
        self.num_layers = num_layers

    def forward(self, src, trg, teacher_forcing_ratio=0.6, src_mask=None):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.fc_out.out_features

        encoder_outputs, hidden, cell = self.encoder(src)
        hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell = cell.unsqueeze(0).repeat(self.num_layers, 1, 1)

        outputs = torch.zeros(batch_size, trg_len - 1, vocab_size, device=src.device)
        input_token = trg[:, 0]
        use_tf = torch.rand(1).item() < teacher_forcing_ratio

        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(
                input_token, hidden, cell, encoder_outputs, src_mask
            )
            outputs[:, t - 1] = output
            input_token = trg[:, t] if use_tf else output.argmax(dim=1)

        return outputs


class seq2seq_bert(nn.Module):
    def __init__(
        self, output_vocab_size, embed_dim, hidden_dim, num_layers, freeze_bert=True
    ):
        super().__init__()
        self.encoder = bert_encoder(hidden_dim, freeze_bert=freeze_bert)
        self.decoder = lstm_decoder_attention(
            output_vocab_size, embed_dim, num_layers, hidden_dim
        )
        self.num_layers = num_layers

    def forward(self, input_ids, attention_mask, trg, teacher_forcing_ratio=0.6):
        batch_size = input_ids.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.fc_out.out_features

        encoder_outputs, hidden, cell = self.encoder(input_ids, attention_mask)
        hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell = cell.unsqueeze(0).repeat(self.num_layers, 1, 1)

        src_mask = attention_mask
        outputs = torch.zeros(
            batch_size, trg_len - 1, vocab_size, device=input_ids.device
        )
        input_token = trg[:, 0]
        use_tf = torch.rand(1).item() < teacher_forcing_ratio

        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(
                input_token, hidden, cell, encoder_outputs, src_mask
            )
            outputs[:, t - 1] = output
            input_token = trg[:, t] if use_tf else output.argmax(dim=1)

        return outputs
