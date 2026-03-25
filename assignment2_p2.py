"""
CSL7640 - NLU Assignment 2
Problem 2 - Character Level Name Generation using RNN Variants
Name: [Your Name]
Roll: [Your Roll No]

three models using PyTorch:
  1. Vanilla RNN
  2. Bidirectional LSTM (BLSTM)
  3. RNN with Attention

how to run:
    python assignment2_p2.py           # full pipeline
    python assignment2_p2.py --task 0  # generate dataset (skips if file exists)
    python assignment2_p2.py --task 1  # train all models
    python assignment2_p2.py --task 2  # evaluate

the name generation (task 0) only hits the API once.
if TrainingNames.txt already exists it just loads from disk.
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# config - change these if needed
# ---------------------------------------------------------------

NAMES_FILE  = "TrainingNames.txt"
HIDDEN_SIZE = 256
N_LAYERS    = 2
DROPOUT     = 0.3
EPOCHS      = 60
LR          = 0.001
BATCH_SIZE  = 64
N_GENERATE  = 200   # how many names to generate during evaluation
MAX_LEN     = 20    # max length of a generated name

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {DEVICE}")

PAD = '<PAD>'
SOS = '<SOS>'
EOS = '<EOS>'



def load_names():
    if not os.path.exists(NAMES_FILE):
        raise FileNotFoundError(f"{NAMES_FILE} not found — run task 0 first")
    names = [l.strip() for l in open(NAMES_FILE) if l.strip()]
    print(f"  loaded {len(names)} names from {NAMES_FILE}")
    return names


def run_task0():
    print("\n===== TASK 0 : Dataset =====")
    names = [l.strip() for l in open(NAMES_FILE) if l.strip()]
    lengths = [len(n) for n in names]
    chars   = set(c.lower() for n in names for c in n)
    print(f"  total names  : {len(names)}")
    print(f"  avg length   : {sum(lengths)/len(lengths):.1f} chars")
    print(f"  unique chars : {len(chars)}")
    print(f"  first 10     : {names[:10]}")


# ===============================================================
# CHARACTER VOCABULARY
# ===============================================================

class CharVocab:
    """
    char <-> index mapping.
    adds PAD, SOS, EOS special tokens before the actual alphabet.

    encode() wraps a name as:  [SOS, c1, c2, ..., EOS]
    training splits this into:
        input  = [SOS, c1, c2, ...]
        target = [c1, c2, ..., EOS]
    """
    def __init__(self, names):
        chars  = sorted(set(c.lower() for n in names for c in n))
        tokens = [PAD, SOS, EOS] + chars

        self.c2i        = {c: i for i, c in enumerate(tokens)}
        self.i2c        = tokens
        self.pad_idx    = self.c2i[PAD]
        self.sos_idx    = self.c2i[SOS]
        self.eos_idx    = self.c2i[EOS]
        self.vocab_size = len(tokens)

        print(f"  vocab: {self.vocab_size} tokens ({len(chars)} chars + 3 special)")

    def encode(self, name):
        idxs = [self.c2i.get(c.lower(), self.pad_idx) for c in name]
        return [self.sos_idx] + idxs + [self.eos_idx]

    def decode(self, idxs):
        out = []
        for i in idxs:
            if i == self.eos_idx:
                break
            if i not in (self.sos_idx, self.pad_idx):
                out.append(self.i2c[i])
        return ''.join(out)


# ===============================================================
# DATASET + DATALOADER
# ===============================================================

class NameDataset(Dataset):
    """
    wraps the list of encoded names.
    getitem returns (input_seq, target_seq) as tensors.
    collate_fn pads them to the same length within a batch.
    """
    def __init__(self, names, vocab):
        self.data = []
        for n in names:
            enc = vocab.encode(n)
            x   = torch.tensor(enc[:-1], dtype=torch.long)
            y   = torch.tensor(enc[1:],  dtype=torch.long)
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    xs, ys = zip(*batch)
    xs_pad = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    ys_pad = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0)
    return xs_pad, ys_pad


# ===============================================================
# MODEL 1 — Vanilla RNN
# ===============================================================

class VanillaRNN(nn.Module):
    """
    Basic stacked RNN for character-level name generation.

    Architecture:
        embedding(vocab -> hidden)
        -> RNN (hidden -> hidden, 2 layers)
        -> dropout
        -> linear (hidden -> vocab)
        -> softmax (during generation)

    The RNN update at each step is:
        h_t = tanh(W_ih * x_t  +  W_hh * h_{t-1}  +  b)

    Known weakness: vanishing gradients over long sequences because
    gradients get multiplied by W_hh repeatedly. For names (short
    sequences) this isn't too bad but the model does converge slower
    than LSTM.

    Hyperparameters used:
        hidden_size = 256,  n_layers = 2,
        dropout = 0.3,      lr = 0.001 (Adam)
    """
    def __init__(self, vocab_size, hidden_size, n_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        self.embed   = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.rnn     = nn.RNN(
            hidden_size, hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, vocab_size)

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  VanillaRNN     | hidden={hidden_size} layers={n_layers} "
              f"dropout={dropout} params={self.n_params:,}")

    def forward(self, x, h=None):
        # x: (B, T) -> embed -> (B, T, H) -> rnn -> (B, T, H) -> fc -> (B, T, V)
        emb     = self.dropout(self.embed(x))
        out, h  = self.rnn(emb, h)
        logits  = self.fc(self.dropout(out))
        return logits, h

    def generate(self, vocab, temperature=0.8):
        self.eval()
        with torch.no_grad():
            idx = torch.tensor([[vocab.sos_idx]], device=DEVICE)
            h   = None
            out = []
            for _ in range(MAX_LEN):
                logits, h = self.forward(idx, h)
                probs     = F.softmax(logits[0, -1] / temperature, dim=-1)
                idx       = torch.multinomial(probs, 1).unsqueeze(0)
                c = idx.item()
                if c == vocab.eos_idx: break
                if c != vocab.pad_idx: out.append(vocab.i2c[c])
        return ''.join(out).capitalize()


# ===============================================================
# MODEL 2 — Bidirectional LSTM
# ===============================================================

class BidirectionalLSTM(nn.Module):
    """
    LSTM with bidirectional encoding.

    LSTM gates at each timestep (one direction):
        f = sigmoid(W_f @ [h, x] + b_f)   <- forget gate
        i = sigmoid(W_i @ [h, x] + b_i)   <- input gate
        o = sigmoid(W_o @ [h, x] + b_o)   <- output gate
        g = tanh   (W_g @ [h, x] + b_g)   <- cell candidate
        c = f * c_prev  +  i * g           <- update cell state
        h = o * tanh(c)                    <- hidden state

    The cell state c gives gradients a direct path backwards through
    time (no squashing), which is why LSTMs handle longer sequences
    better than vanilla RNNs.

    Bidirectional: forward LSTM (left→right) + backward LSTM (right→left),
    outputs concatenated at each position -> size 2*hidden_size.

    During generation only the forward direction is used since we
    can't look at future characters while sampling.

    Hyperparameters:
        hidden_size = 256, n_layers = 2,
        dropout = 0.3,     lr = 0.001 (Adam)
    """
    def __init__(self, vocab_size, hidden_size, n_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.lstm  = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        # 2*hidden because concat of fwd + bwd
        self.fc      = nn.Linear(2 * hidden_size, vocab_size)

        # separate forward-only LSTM for generation
        self.lstm_fwd = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False
        )
        # project forward hidden to vocab (we double it to match fc input shape)
        self.fc_fwd = nn.Linear(hidden_size, vocab_size)

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  BidirectionalLSTM | hidden={hidden_size} layers={n_layers} "
              f"dropout={dropout} params={self.n_params:,}")

    def forward(self, x, state=None):
        emb        = self.dropout(self.embed(x))
        out, state = self.lstm(emb, state)
        logits     = self.fc(self.dropout(out))
        return logits, state

    def copy_fwd_weights(self):
        """
        Copy forward direction weights from bidirectional lstm into
        lstm_fwd. For multi-layer LSTMs, we must slice the input-to-hidden
        weights of layers > 0 because they expect concatenated inputs.
        """
        sd_bidir = self.lstm.state_dict()
        sd_fwd   = self.lstm_fwd.state_dict()

        for key in sd_fwd:
            if key in sd_bidir:
                # Check if this is an input-to-hidden weight for layers above 0
                # These have shape [4*H, 2*H] in Bi-LSTM but [4*H, H] in fwd-only
                if 'weight_ih_l' in key and not key.endswith('_l0'):
                    # Slice the weight to only take the first 'hidden_size' columns
                    sd_fwd[key] = sd_bidir[key][:, :self.hidden_size]
                else:
                    # For all other weights (Layer 0 or hidden-to-hidden), shapes match
                    sd_fwd[key] = sd_bidir[key]

        self.lstm_fwd.load_state_dict(sd_fwd)

    def generate(self, vocab, temperature=0.8):
        """use the forward-only lstm for generation"""
        self.eval()
        self.copy_fwd_weights()
        with torch.no_grad():
            idx = torch.tensor([[vocab.sos_idx]], device=DEVICE)
            h   = None
            out = []
            for _ in range(MAX_LEN):
                emb       = self.dropout(self.embed(idx))
                rnn_out, h= self.lstm_fwd(emb, h)
                logits    = self.fc_fwd(rnn_out[:, -1, :])
                probs     = F.softmax(logits[0] / temperature, dim=-1)
                idx       = torch.multinomial(probs, 1).unsqueeze(0)
                c = idx.item()
                if c == vocab.eos_idx: break
                if c != vocab.pad_idx: out.append(vocab.i2c[c])
        return ''.join(out).capitalize()


# ===============================================================
# MODEL 3 — RNN with Basic Attention
# ===============================================================

class AttentionRNN(nn.Module):
    """
    GRU-based RNN with additive (Bahdanau-style) attention.

    Why GRU instead of vanilla RNN here:
    GRU has update and reset gates (similar idea to LSTM but simpler),
    which helps it retain context better than tanh RNN, making it
    a more reasonable base for an attention mechanism.

    At each decode step t, attention computes:
        e_{t,k} = v^T * tanh(W_a * enc_out[k]  +  U_a * h_t  +  b_a)
        alpha_t  = softmax(e_t)
        context  = sum_k(alpha_{t,k} * enc_out[k])

    Then output:
        y_t = W_out([h_t ; context_t]) + b

    The attention weights alpha tell us which encoder positions the
    model is focusing on when generating each character. You can
    visualize these as a heatmap (done in task 2).

    Hyperparameters:
        hidden_size = 256, n_layers = 2,
        dropout = 0.3,     lr = 0.001 (Adam)
    """
    def __init__(self, vocab_size, hidden_size, n_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        self.embed   = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.gru     = nn.GRU(
            hidden_size, hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)

        # attention params
        self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_a = nn.Linear(hidden_size, 1,           bias=False)

        # output: [h_t ; context] => 2*H -> vocab
        self.fc  = nn.Linear(2 * hidden_size, vocab_size)

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  AttentionRNN   | hidden={hidden_size} layers={n_layers} "
              f"dropout={dropout} params={self.n_params:,}")

    def attend(self, h_dec, enc_outs):
        """
        h_dec    : (B, H)    current decoder hidden
        enc_outs : (B, T, H) all encoder outputs
        returns context (B, H) and weights (B, T)
        """
        h_exp   = h_dec.unsqueeze(1).expand_as(enc_outs)      # (B, T, H)
        energy  = self.v_a(
            torch.tanh(self.W_a(enc_outs) + self.U_a(h_exp))
        ).squeeze(-1)                                           # (B, T)
        alpha   = F.softmax(energy, dim=-1)                    # (B, T)
        context = torch.bmm(alpha.unsqueeze(1), enc_outs).squeeze(1)  # (B, H)
        return context, alpha

    def forward(self, x, h=None):
        emb          = self.dropout(self.embed(x))   # (B, T, H)
        enc_outs, h  = self.gru(emb, h)              # (B, T, H)

        outputs = []
        for t in range(enc_outs.size(1)):
            h_t     = enc_outs[:, t, :]              # (B, H)
            ctx, _  = self.attend(h_t, enc_outs)
            combined= torch.cat([h_t, ctx], dim=-1)  # (B, 2H)
            outputs.append(combined.unsqueeze(1))

        out    = torch.cat(outputs, dim=1)            # (B, T, 2H)
        logits = self.fc(self.dropout(out))
        return logits, h

    def generate(self, vocab, temperature=0.8):
        self.eval()
        with torch.no_grad():
            idx      = torch.tensor([[vocab.sos_idx]], device=DEVICE)
            h        = None
            all_outs = []
            out      = []

            for _ in range(MAX_LEN):
                emb         = self.dropout(self.embed(idx))
                step_out, h = self.gru(emb, h)              # (1, 1, H)
                all_outs.append(step_out)

                enc_outs = torch.cat(all_outs, dim=1)        # (1, t, H)
                h_t      = step_out[:, 0, :]
                ctx, _   = self.attend(h_t, enc_outs)

                combined = torch.cat([h_t, ctx], dim=-1)
                logits   = self.fc(combined)
                probs    = F.softmax(logits[0] / temperature, dim=-1)
                idx      = torch.multinomial(probs, 1).unsqueeze(0)
                c = idx.item()
                if c == vocab.eos_idx: break
                if c != vocab.pad_idx: out.append(vocab.i2c[c])

        return ''.join(out).capitalize()


# ===============================================================
# TRAINING
# ===============================================================

def train_model(model, dataloader, vocab, label):
    """
    Standard training loop:
    - Adam optimizer
    - cross entropy loss (ignoring PAD tokens)
    - gradient clipping at 1.0 to avoid exploding gradients
    - ReduceLROnPlateau scheduler (halves lr if loss stalls for 5 epochs)
    - prints a sample generated name every 10 epochs
    """
    model.to(DEVICE)
    opt       = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=5, factor=0.5
    )
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    losses    = []

    print(f"\n  training [{label}]  epochs={EPOCHS}  lr={LR}  batch={BATCH_SIZE}")

    for ep in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        n_batch = 0

        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits, _ = model(x)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            n_batch += 1

        avg = ep_loss / n_batch
        losses.append(avg)
        scheduler.step(avg)

        cur_lr = opt.param_groups[0]['lr']
        print(f"    ep {ep+1:>3}/{EPOCHS}  loss={avg:.4f}  lr={cur_lr:.5f}", end='')

        if (ep+1) % 10 == 0:
            model.eval()
            s = model.generate(vocab)
            print(f"  -> {s}", end='')
        print()

    return losses


def run_task1():
    print("\n===== TASK 1 : Model Training =====")

    names = load_names()
    vocab = CharVocab(names)

    dataset    = NameDataset(names, vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=collate_fn)

    print(f"\n  Architecture Summary:")
    print(f"  {'Model':<20} {'Hidden':>7} {'Layers':>7} {'Dropout':>9} "
          f"{'Params':>12}  LR")
    print(f"  {'-'*65}")

    rnn   = VanillaRNN      (vocab.vocab_size, HIDDEN_SIZE, N_LAYERS, DROPOUT)
    blstm = BidirectionalLSTM(vocab.vocab_size, HIDDEN_SIZE, N_LAYERS, DROPOUT)
    attn  = AttentionRNN    (vocab.vocab_size, HIDDEN_SIZE, N_LAYERS, DROPOUT)

    for mname, m in [("VanillaRNN",rnn),("BidirLSTM",blstm),("AttentionRNN",attn)]:
        print(f"  {mname:<20} {HIDDEN_SIZE:>7} {N_LAYERS:>7} {DROPOUT:>9} "
              f"{m.n_params:>12,}  {LR}")

    rnn_l   = train_model(rnn,   dataloader, vocab, "VanillaRNN")
    blstm_l = train_model(blstm, dataloader, vocab, "BLSTM")
    attn_l  = train_model(attn,  dataloader, vocab, "AttentionRNN")

    # save all three
    for fname, m, l in [("rnn_model.pt",   rnn,   rnn_l),
                        ("blstm_model.pt",  blstm, blstm_l),
                        ("attn_model.pt",   attn,  attn_l)]:
        torch.save({
            'model_state': m.state_dict(),
            'losses':      l,
            'vocab':       vocab,
            'config': dict(vocab_size=vocab.vocab_size,
                           hidden_size=HIDDEN_SIZE,
                           n_layers=N_LAYERS,
                           dropout=DROPOUT)
        }, fname)
        print(f"  saved {fname}")

    # loss curve
    plt.figure(figsize=(10, 5))
    for l, lbl, col in [(rnn_l,  'Vanilla RNN',   '#1565C0'),
                        (blstm_l,'BLSTM',          '#2E7D32'),
                        (attn_l, 'Attention RNN',  '#B71C1C')]:
        plt.plot(l, label=lbl, color=col, linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Avg Loss')
    plt.title('Training Loss — Character-level Name Generation')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('p2_loss_curves.png', dpi=150)
    plt.close()
    print("  saved p2_loss_curves.png")


# ===============================================================
# TASK 2 — Evaluation
# ===============================================================

def novelty_rate(generated, train_names):
    """% of generated names that don't appear in training set"""
    train_set = set(n.lower() for n in train_names)
    novel     = sum(1 for n in generated if n.lower() not in train_set)
    return novel / len(generated) if generated else 0.0


def diversity(generated):
    """unique names / total generated"""
    return len(set(n.lower() for n in generated)) / len(generated) if generated else 0.0


def is_valid(name):
    return len(name) >= 2 and name.replace(' ', '').isalpha()


def load_model(fname, ModelClass):
    # 1. Register your custom class as "safe"
    torch.serialization.add_safe_globals([CharVocab])
    
    # 2. Now load as usual
    ckpt = torch.load(fname, map_location=DEVICE, weights_only=True)
    
    cfg = ckpt['config']
    m = ModelClass(cfg['vocab_size'], cfg['hidden_size'],
                   cfg['n_layers'],   cfg['dropout'])
    m.load_state_dict(ckpt['model_state'])
    m.to(DEVICE)
    return m, ckpt['vocab'], ckpt['losses']


def run_task2():
    print("\n===== TASK 2 : Evaluation =====")

    for f in ["rnn_model.pt", "blstm_model.pt", "attn_model.pt"]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} not found — run task 1 first")

    names   = load_names()
    results = {}
    all_gen = {}

    for fname, ModelClass, label in [
        ("rnn_model.pt",   VanillaRNN,        "VanillaRNN"),
        ("blstm_model.pt", BidirectionalLSTM, "BLSTM"),
        ("attn_model.pt",  AttentionRNN,       "AttentionRNN"),
    ]:
        model, vocab, losses = load_model(fname, ModelClass)
        model.eval()

        gen = []
        for _ in range(N_GENERATE):
            t    = float(np.random.choice([0.6, 0.8, 1.0, 1.2]))
            name = model.generate(vocab, temperature=t)
            if is_valid(name):
                gen.append(name)

        nov = novelty_rate(gen, names)
        div = diversity(gen)

        results[label] = dict(novelty=nov, diversity=div,
                              final_loss=losses[-1],
                              n_generated=len(gen),
                              samples=gen[:20])
        all_gen[label] = gen

        print(f"\n  {label}")
        print(f"    generated    : {len(gen)}")
        print(f"    novelty rate : {nov:.3f}  ({nov*100:.1f}% new names)")
        print(f"    diversity    : {div:.3f}  ({div*100:.1f}% unique)")
        print(f"    final loss   : {losses[-1]:.4f}")
        print(f"    samples      : {', '.join(gen[:8])}")

    # summary table
    print(f"\n  {'Model':<16} {'Novelty':>9} {'Diversity':>11} "
          f"{'Loss':>8} {'Generated':>11}")
    print(f"  {'-'*58}")
    for label, r in results.items():
        print(f"  {label:<16} {r['novelty']:>8.3f}  {r['diversity']:>10.3f}  "
              f"{r['final_loss']:>7.4f}  {r['n_generated']:>10}")

    # save generated name txt files
    for label, gen in all_gen.items():
        fname = f"generated_{label}.txt"
        with open(fname, 'w') as f:
            f.write('\n'.join(gen) + '\n')
        print(f"  saved {fname}")

    # failure mode analysis
    print("\n  --- Qualitative Analysis ---")
    print("\n  Sample generated names (realism check):")
    for label, gen in all_gen.items():
        print(f"    {label}: {', '.join(gen[:10])}")

    print("\n  Failure modes:")
    for label, gen in all_gen.items():
        short = [n for n in gen if len(n) <= 2]
        long_ = [n for n in gen if len(n) > 12]
        # names where one char dominates (e.g. "aaaaaaa")
        rept  = [n for n in gen
                 if max(Counter(n.lower()).values()) > len(n)//2]
        print(f"\n  {label}:")
        print(f"    too short (<=2): {len(short):>3}  e.g. {short[:4]}")
        print(f"    too long  (>12): {len(long_):>3}  e.g. {long_[:4]}")
        print(f"    repetitive char: {len(rept):>3}  e.g. {rept[:4]}")

    # bar charts
    labels = list(results.keys())
    nov_v  = [results[l]['novelty']    for l in labels]
    div_v  = [results[l]['diversity']  for l in labels]
    loss_v = [results[l]['final_loss'] for l in labels]
    cols   = ['#1565C0', '#2E7D32', '#B71C1C']

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Evaluation — Character-level Name Generation',
                 fontsize=13, fontweight='bold')

    for ax, vals, title, ylabel in [
        (axes[0], nov_v,  'Novelty Rate',       'Fraction not in training set'),
        (axes[1], div_v,  'Diversity',           'Unique / Total generated'),
        (axes[2], loss_v, 'Final Training Loss', 'Avg cross-entropy'),
    ]:
        bars = ax.bar(labels, vals, color=cols, edgecolor='white', width=0.5)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        if title != 'Final Training Loss':
            ax.set_ylim(0, 1.1)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.25, linestyle='--')
        ax.tick_params(axis='x', labelrotation=10)

    plt.tight_layout()
    plt.savefig('p2_evaluation.png', dpi=150)
    plt.close()
    print("\n  saved p2_evaluation.png")

    # save metrics json
    with open('p2_eval_results.json', 'w') as f:
        save = {l: {k: v for k, v in r.items() if k != 'samples'}
                for l, r in results.items()}
        json.dump(save, f, indent=2)
    print("  saved p2_eval_results.json")


# ===============================================================
# main
# ===============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, choices=[0, 1, 2],
                        help='0=dataset  1=train  2=eval  (default: all)')
    args = parser.parse_args()

    if args.task is not None:
        {0: run_task0, 1: run_task1, 2: run_task2}[args.task]()
    else:
        run_task0()   # skips automatically if TrainingNames.txt exists
        run_task1()
        run_task2()
        print("\nall done!")


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    main()
