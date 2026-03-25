import os
import re
import sys
import time
import json
import pickle
import argparse
from collections import Counter
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# -------------------------------------------------------------------
# some global settings, change these if needed
# -------------------------------------------------------------------

EMBED_DIMS   = [50, 100]     # trying two sizes
WINDOW_SIZES = [2, 5]
NEG_SAMPLES  = [5, 10]
EPOCHS       = 5
LR           = 0.025
MIN_FREQ     = 2             # ignore words that appear less than this
SUBSAMPLE_T  = 1e-4          # for subsampling frequent words

RAW_DIR      = "raw_data"
CORPUS_FILE  = "corpus.txt"
STATS_FILE   = "corpus_stats.json"

# words to analyze in task 3
# note: "exam" appears twice in the assignment pdf, keeping 5 unique ones
PROBE_WORDS  = ["research", "student", "phd", "exam", "department"]

STOP_WORDS = set(stopwords.words('english'))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
}

# seed urls - tried to cover departments, regulations (mandatory),
# research pages and faculty. added more than minimum just to get
# enough data

# NOTE: iitj.ac.in main site is JS-rendered, requests can't scrape it.
# old.iitj.ac.in serves real static HTML and has the same content.
# academics.iitj.ac.in also works fine.
# All URLs below are verified to return actual text content.

SEED_URLS = {
    "departments": [
        "https://old.iitj.ac.in/department/index.php?id=cse",
        "https://old.iitj.ac.in/department/index.php?id=ece",
        "https://old.iitj.ac.in/department/index.php?id=me",
        "https://old.iitj.ac.in/department/index.php?id=physics",
        "https://old.iitj.ac.in/department/index.php?id=chemistry",
        "https://old.iitj.ac.in/department/index.php?id=maths",
        "https://old.iitj.ac.in/department/index.php?id=bioscience",
        "https://old.iitj.ac.in/department/index.php?id=civil",
        "https://old.iitj.ac.in/department/index.php?id=cse&cat=people",
        "https://old.iitj.ac.in/department/index.php?id=ece&cat=people",
        "https://old.iitj.ac.in/department/index.php?id=me&cat=people",
        "https://old.iitj.ac.in/department/index.php?id=cse&cat=research",
        "https://old.iitj.ac.in/department/index.php?id=ece&cat=research",
        "https://old.iitj.ac.in/department/index.php?id=physics&cat=research",
    ],
    # assignment says this is MUST - academics.iitj.ac.in serves static HTML
    "regulations": [
        "http://academics.iitj.ac.in/?page_id=54",   # academic regulations
        "http://academics.iitj.ac.in/?page_id=52",   # circulars
        "http://academics.iitj.ac.in/?page_id=49",   # academic divisions
        "http://academics.iitj.ac.in/?page_id=56",   # PhD regulations
        "http://academics.iitj.ac.in/",
        "https://old.iitj.ac.in/academics/index.php?id=regulations",
        "https://old.iitj.ac.in/academics/index.php?id=programmes",
        "https://old.iitj.ac.in/academics/index.php?id=curriculum",
        "https://old.iitj.ac.in/academics/",
    ],
    "research": [
        "https://old.iitj.ac.in/research/",
        "https://old.iitj.ac.in/research/index.php?id=overview",
        "https://old.iitj.ac.in/research/index.php?id=centres",
        "https://old.iitj.ac.in/research/index.php?id=projects",
        "https://old.iitj.ac.in/department/index.php?id=cse&cat=research",
        "https://old.iitj.ac.in/department/index.php?id=me&cat=research",
    ],
    "institute_general": [
        "https://old.iitj.ac.in/",
        "https://old.iitj.ac.in/institute/index.php?id=about",
        "https://old.iitj.ac.in/institute/index.php?id=vision",
        "https://old.iitj.ac.in/announcement/",
        "https://old.iitj.ac.in/people/index.php?id=faculty",
        # techscape newsletter - good source for institute news text
        "https://old.iitj.ac.in/techscape/vol04/issue01/commemorating_iitj/the_glorious_journey/",
        "https://old.iitj.ac.in/schools/execfaq.php",
    ],
    "faculty_profiles": [
        "https://old.iitj.ac.in/department/index.php?id=cse&cat=people",
        "https://old.iitj.ac.in/department/index.php?id=ece&cat=people",
        "https://old.iitj.ac.in/department/index.php?id=me&cat=people",
        "https://old.iitj.ac.in/department/index.php?id=physics&cat=people",
        "https://old.iitj.ac.in/department/index.php?id=maths&cat=people",
    ],
}


# ===================================================================
# TASK 1 - Data Collection and Preprocessing
# ===================================================================

def fetch_url(url):
    # just a simple wrapper, returns None if anything goes wrong
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            return r.text
        print(f"    got {r.status_code} for {url}, skipping")
        return None
    except Exception as e:
        print(f"    failed to fetch {url}: {e}")
        return None


def get_text_from_html(html):
    """
    Parse html and get the main text content. Remove nav, footer etc because they add garbage like
    "Home > Departments > CSE" repeated on every page
    """
    soup = BeautifulSoup(html, 'html.parser')

    # remove noise tags
    for t in soup(['script', 'style', 'nav', 'footer',
                   'header', 'noscript', 'iframe', 'form']):
        t.decompose()

    # try to find main content div, fall back to body
    content = (soup.find('div', id='content') or
               soup.find('div', class_=re.compile('content|main', re.I)) or
               soup.find('main') or
               soup.find('body'))

    if not content:
        return ""
    return content.get_text(separator=' ', strip=True)


def is_english(text, thresh=0.85):
    # check if text is mostly ascii (english)
    # iitj site has some hindi text so need to filter that out
    if not text:
        return False
    ascii_c = sum(1 for c in text if ord(c) < 128)
    return (ascii_c / len(text)) >= thresh


def find_links(html, base_url):
    # follow links on old.iitj.ac.in and academics.iitj.ac.in
    # these are the domains that actually serve static HTML
    soup  = BeautifulSoup(html, 'html.parser')
    found = set()

    ALLOWED_DOMAINS = ('old.iitj.ac.in', 'academics.iitj.ac.in')

    for a in soup.find_all('a', href=True):
        full = urljoin(base_url, a['href'].strip())
        p    = urlparse(full)
        if p.scheme not in ('http', 'https'):
            continue
        if not any(d in p.netloc for d in ALLOWED_DOMAINS):
            continue
        if re.search(r'\.(pdf|jpg|png|zip|doc|ppt|xls)$', p.path, re.I):
            continue
        found.add(full)
    return found


def scrape_iitj():
    """
    Scrape text from old.iitj.ac.in across different page types.
    Follows internal links one level deep so we get subpages too without hardcoding every url.
    Saves each page as a separate txt file.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    total = 0

    for cat, seeds in SEED_URLS.items():
        print(f"\n  [{cat}]")
        visited = set()
        queue   = list(seeds)
        idx     = 0

        while queue and idx < 25:   # cap at 25 pages per category
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            print(f"    fetching: {url}")
            html = fetch_url(url)
            if not html:
                continue

            text = get_text_from_html(html)

            if not is_english(text):
                print("    skipping - not english")
                continue
            if len(text.split()) < 50:
                print("    skipping - too short")
                continue

            fname = os.path.join(RAW_DIR, f"{cat}_{idx}.txt")
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"    saved: {fname}  ({len(text.split())} words)")
            idx  += 1
            total += 1

            # crawl one level of sub-links
            for link in find_links(html, url):
                if link not in visited:
                    queue.append(link)

            time.sleep(1.5)  # be polite, don't hammer the server

        print(f"  done - saved {idx} pages for {cat}")

    print(f"\n  total pages scraped: {total}")


def clean_text(text):
    # remove all the web garbage - urls, emails, phone numbers etc
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)
    text = re.sub(r'&[a-z]+;', ' ', text)
    text = re.sub(r'[\+\(]?[0-9][0-9\s\-\(\)]{8,}[0-9]', ' ', text)
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text):
    """
    Split text into sentences, tokenize each sentence.
    Keep only alphabetic tokens, lowercase, min length 2.
    Drop sentences with fewer than 4 words.
    Returns list of token lists (one per sentence).
    """
    result = []
    for sent in sent_tokenize(text):
        tokens = []
        for tok in word_tokenize(sent):
            tok = tok.lower()
            if tok.isalpha() and len(tok) >= 2:
                tokens.append(tok)
        if len(tokens) >= 4:
            result.append(tokens)
    return result


def build_corpus():
    """
    Read all txt files from raw_data/, clean and tokenize them,
    write final corpus.txt (one sentence per line, space separated).
    Also compute and print stats + save word cloud.
    """
    if not os.path.exists(RAW_DIR):
        raise FileNotFoundError(f"'{RAW_DIR}' not found, run scraper first")

    all_sents  = []
    all_tokens = []
    ndocs      = 0

    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith('.txt'):
            continue
        with open(os.path.join(RAW_DIR, fname), 'r',
                  encoding='utf-8', errors='replace') as f:
            raw = f.read()
        if not raw.strip():
            continue
        ndocs += 1
        sents  = tokenize(clean_text(raw))
        all_sents.extend(sents)
        for s in sents:
            all_tokens.extend(s)

    # save corpus
    with open(CORPUS_FILE, 'w', encoding='utf-8') as f:
        for sent in all_sents:
            f.write(' '.join(sent) + '\n')

    vocab     = Counter(all_tokens)
    vocab_sz  = len(vocab)

    # top words excluding common stopwords
    top_words = [(w, c) for w, c in vocab.most_common(60)
                 if w not in STOP_WORDS][:20]

    stats = {
        "num_documents" : ndocs,
        "num_sentences" : len(all_sents),
        "num_tokens"    : len(all_tokens),
        "vocab_size"    : vocab_sz,
        "top_20_words"  : top_words
    }

    print(f"\n  --- Corpus Statistics ---")
    print(f"  Documents  : {ndocs}")
    print(f"  Sentences  : {len(all_sents)}")
    print(f"  Tokens     : {len(all_tokens)}")
    print(f"  Vocab size : {vocab_sz}")
    print(f"\n  Top 20 words (no stopwords):")
    for i, (w, c) in enumerate(top_words, 1):
        print(f"    {i:>2}. {w:<20} {c}")

    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n  stats saved to {STATS_FILE}")

    # word cloud - excluding stopwords so actual content words show up
    freq = {w: c for w, c in vocab.items()
            if w not in STOP_WORDS and len(w) > 2}
    wc = WordCloud(width=1200, height=600, background_color='white',
                   colormap='Blues', max_words=150).generate_from_frequencies(freq)
    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Frequent Words - IIT Jodhpur Corpus', fontsize=14)
    plt.tight_layout()
    plt.savefig('wordcloud.png', dpi=150)
    plt.close()
    print("  word cloud saved to wordcloud.png")

    return all_sents, all_tokens, vocab


def run_task1():
    print("\n========== TASK 1 : Data Collection & Preprocessing ==========")
    scrape_iitj()
    build_corpus()
    print("\nTask 1 done.")


# ===================================================================
# TASK 2 - Word2Vec from scratch (no gensim, pure numpy as required)
# ===================================================================

# --- vocabulary ---

class Vocab:
    """
    Stores the vocabulary and related stuff needed during training.
    Filters out rare words (below min_freq).
    Also computes the noise distribution for negative sampling
    using the f(w)^0.75 trick from the original paper.
    """
    def __init__(self, sentences, min_freq=2, t=1e-4):
        counts = Counter(w for s in sentences for w in s)
        # drop rare words
        counts = {w: c for w, c in counts.items() if c >= min_freq}

        # sort by freq descending
        words = sorted(counts, key=lambda x: -counts[x])

        self.w2i = {w: i for i, w in enumerate(words)}
        self.i2w = words

        freq_arr = np.array([counts[w] for w in words], dtype=np.float64)
        self.counts = freq_arr

        # noise distribution - f(w)^0.75 as in Mikolov 2013
        noise = freq_arr ** 0.75
        self.noise_dist = noise / noise.sum()

        # subsampling: drop frequent words with some probability
        # this helps because words like "the" appear too often and
        # don't add much signal
        total = freq_arr.sum()
        p_keep = np.sqrt(t / (freq_arr / total + 1e-10))
        self.keep_prob = np.clip(p_keep, 0, 1)

        print(f"  vocab size (min_freq={min_freq}): {len(words)}")

    def __len__(self):
        return len(self.i2w)

    def encode(self, sentences):
        # convert words to indices, apply subsampling
        rng = np.random.default_rng(0)
        encoded = []
        for sent in sentences:
            ids = []
            for w in sent:
                if w not in self.w2i:
                    continue
                i = self.w2i[w]
                if rng.random() < self.keep_prob[i]:
                    ids.append(i)
            if ids:
                encoded.append(ids)
        return encoded

    def neg_sample(self, exclude, k):
        # draw k negatives, avoid the exclude set
        negs = []
        while len(negs) < k:
            s = np.random.choice(len(self), p=self.noise_dist)
            if s not in exclude:
                negs.append(s)
        return negs


def sigmoid(x):
    # stable sigmoid - handles very negative values without overflow
    return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))


# --- CBOW ---

class CBOW:
    """
    CBOW: predict target word from average of context words.
    Uses negative sampling as the training objective.

    W_in  = context/input embeddings
    W_out = target/output embeddings
    Final embeddings = average of both (common practice)
    """
    def __init__(self, vocab_size, dim, window, n_neg, lr):
        self.V      = vocab_size
        self.dim    = dim
        self.window = window
        self.n_neg  = n_neg
        self.lr     = lr

        # small uniform init for W_in, zeros for W_out
        s = 0.5 / dim
        self.W_in  = np.random.uniform(-s, s, (vocab_size, dim))
        self.W_out = np.zeros((vocab_size, dim))

        nparams = 2 * vocab_size * dim
        print(f"  CBOW | dim={dim} window={window} neg={n_neg} | params={nparams:,}")

    def train_sent(self, sent, vocab, lr):
        loss = 0.0
        for pos in range(len(sent)):
            target = sent[pos]

            # gather context indices
            lo  = max(0, pos - self.window)
            hi  = min(len(sent), pos + self.window + 1)
            ctx = [sent[j] for j in range(lo, hi) if j != pos]
            if not ctx:
                continue

            # context vector = mean of input embeddings
            h = self.W_in[ctx].mean(axis=0)

            negs = vocab.neg_sample(set(ctx) | {target}, self.n_neg)

            # positive pair update
            s_pos = np.dot(h, self.W_out[target])
            g_pos = sigmoid(s_pos) - 1.0
            grad_h = g_pos * self.W_out[target]
            self.W_out[target] -= lr * g_pos * h
            loss -= np.log(sigmoid(s_pos) + 1e-10)

            # negative pairs
            for neg in negs:
                s_neg = np.dot(h, self.W_out[neg])
                g_neg = sigmoid(s_neg)
                grad_h += g_neg * self.W_out[neg]
                self.W_out[neg] -= lr * g_neg * h
                loss -= np.log(1 - sigmoid(s_neg) + 1e-10)

            # update all context word embeddings (shared gradient, split equally)
            self.W_in[ctx] -= lr * (grad_h / len(ctx))

        return loss

    @property
    def embeddings(self):
        return (self.W_in + self.W_out) / 2.0


# --- Skip-gram with negative sampling (SGNS) ---

class SGNS:
    """
    SGNS: for each target word, predict surrounding context words.
    Opposite direction to CBOW - each context word gets individual update.

    Generally better at capturing rare word semantics than CBOW
    because each (target, context) pair is a separate training example.
    """
    def __init__(self, vocab_size, dim, window, n_neg, lr):
        self.V      = vocab_size
        self.dim    = dim
        self.window = window
        self.n_neg  = n_neg
        self.lr     = lr

        s = 0.5 / dim
        self.W_in  = np.random.uniform(-s, s, (vocab_size, dim))
        self.W_out = np.zeros((vocab_size, dim))

        nparams = 2 * vocab_size * dim
        print(f"  SGNS | dim={dim} window={window} neg={n_neg} | params={nparams:,}")

    def train_sent(self, sent, vocab, lr):
        loss = 0.0
        for pos in range(len(sent)):
            target = sent[pos]
            v_t    = self.W_in[target]

            lo  = max(0, pos - self.window)
            hi  = min(len(sent), pos + self.window + 1)
            ctx_positions = [j for j in range(lo, hi) if j != pos]

            for cp in ctx_positions:
                ctx  = sent[cp]
                negs = vocab.neg_sample({target, ctx}, self.n_neg)

                # positive
                s_pos = np.dot(v_t, self.W_out[ctx])
                g_pos = sigmoid(s_pos) - 1.0
                grad_in = g_pos * self.W_out[ctx]
                self.W_out[ctx] -= lr * g_pos * v_t
                loss -= np.log(sigmoid(s_pos) + 1e-10)

                # negatives
                for neg in negs:
                    s_neg = np.dot(v_t, self.W_out[neg])
                    g_neg = sigmoid(s_neg)
                    grad_in += g_neg * self.W_out[neg]
                    self.W_out[neg] -= lr * g_neg * v_t
                    loss -= np.log(1 - sigmoid(s_neg) + 1e-10)

                self.W_in[target] -= lr * grad_in

        return loss

    @property
    def embeddings(self):
        return (self.W_in + self.W_out) / 2.0


def train(model, encoded_sents, vocab, epochs, init_lr):
    """
    Training loop for both CBOW and SGNS.
    Linear learning rate decay over the whole training run.
    Shuffle sentences every epoch.
    """
    rng        = np.random.default_rng(42)
    total_w    = sum(len(s) for s in encoded_sents)
    word_count = 0
    losses     = []

    for ep in range(epochs):
        order      = rng.permutation(len(encoded_sents))
        ep_loss    = 0.0

        for i, idx in enumerate(order):
            sent = encoded_sents[idx]
            # linear lr decay - same as original word2vec
            progress   = word_count / (total_w * epochs + 1)
            cur_lr     = max(init_lr * (1 - progress), init_lr * 1e-4)

            ep_loss   += model.train_sent(sent, vocab, cur_lr)
            word_count += len(sent)

            if (i+1) % max(1, len(encoded_sents)//10) == 0:
                print(f"    ep {ep+1}/{epochs}  "
                      f"{100*(i+1)/len(encoded_sents):.0f}%  "
                      f"loss={ep_loss/(i+1):.4f}  lr={cur_lr:.5f}", end='\r')

        avg = ep_loss / len(encoded_sents)
        losses.append(avg)
        print(f"\n    epoch {ep+1} done - avg loss: {avg:.4f}")

    return losses


def run_task2():
    print("\n========== TASK 2 : Word2Vec Training (pure numpy) ==========")

    if not os.path.exists(CORPUS_FILE):
        raise FileNotFoundError("corpus.txt not found, run task 1 first")

    # load corpus
    sents = []
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            toks = line.strip().split()
            if toks:
                sents.append(toks)
    print(f"  loaded {len(sents):,} sentences")

    vocab   = Vocab(sents, min_freq=MIN_FREQ, t=SUBSAMPLE_T)
    encoded = vocab.encode(sents)

    results        = []
    best_cbow      = None
    best_sgns      = None
    best_cbow_loss = float('inf')
    best_sgns_loss = float('inf')

    # run over hyperparameter combinations
    for dim in EMBED_DIMS:
        for win in WINDOW_SIZES:
            for neg in NEG_SAMPLES:

                print(f"\n  -- CBOW  dim={dim} win={win} neg={neg} --")
                cbow = CBOW(len(vocab), dim, win, neg, LR)
                cbow_l = train(cbow, encoded, vocab, EPOCHS, LR)
                fl = cbow_l[-1]
                results.append(('CBOW', dim, win, neg, fl))
                if fl < best_cbow_loss:
                    best_cbow_loss = fl
                    best_cbow      = cbow

                print(f"\n  -- SGNS  dim={dim} win={win} neg={neg} --")
                sgns = SGNS(len(vocab), dim, win, neg, LR)
                sgns_l = train(sgns, encoded, vocab, EPOCHS, LR)
                fl = sgns_l[-1]
                results.append(('SGNS', dim, win, neg, fl))
                if fl < best_sgns_loss:
                    best_sgns_loss = fl
                    best_sgns      = sgns

    # print results table for the report
    print("\n  Results summary:")
    print(f"  {'model':<6} {'dim':>4} {'win':>4} {'neg':>4}  {'final_loss':>10}")
    print("  " + "-"*36)
    for (m, d, w, n, l) in results:
        best_marker = ""
        if m == 'CBOW' and l == best_cbow_loss:
            best_marker = " <- best"
        if m == 'SGNS' and l == best_sgns_loss:
            best_marker = " <- best"
        print(f"  {m:<6} {d:>4} {w:>4} {n:>4}  {l:>10.4f}{best_marker}")

    # save best models
    with open('cbow_model.pkl', 'wb') as f:
        pickle.dump({'model': best_cbow, 'vocab': vocab}, f)
    with open('sgns_model.pkl', 'wb') as f:
        pickle.dump({'model': best_sgns, 'vocab': vocab}, f)

    print("\n  best models saved to cbow_model.pkl and sgns_model.pkl")
    print("Task 2 done.")


# ===================================================================
# TASK 3 - Semantic Analysis
# ===================================================================

def cos_sim(vec, matrix):
    # cosine similarity between one vector and all rows of matrix
    dots     = matrix @ vec
    norm_v   = np.linalg.norm(vec) + 1e-10
    norm_m   = np.linalg.norm(matrix, axis=1) + 1e-10
    return dots / (norm_v * norm_m)


def nearest_neighbours(word, emb, vocab, k=5):
    if word not in vocab.w2i:
        return []
    idx       = vocab.w2i[word]
    sims      = cos_sim(emb[idx], emb)
    sims[idx] = -1.0   # exclude the word itself
    top       = np.argsort(sims)[::-1][:k]
    return [(vocab.i2w[i], float(sims[i])) for i in top]


def analogy(a, b, c, emb, vocab, k=5):
    """
    Solve a:b :: c:?  using vector arithmetic: W[b] - W[a] + W[c]
    then find nearest neighbours of the result.
    """
    missing = [w for w in [a, b, c] if w not in vocab.w2i]
    if missing:
        return None, missing

    va = emb[vocab.w2i[a]]
    vb = emb[vocab.w2i[b]]
    vc = emb[vocab.w2i[c]]

    query = vb - va + vc
    sims  = cos_sim(query, emb)

    for w in [a, b, c]:
        sims[vocab.w2i[w]] = -1.0

    top = np.argsort(sims)[::-1][:k]
    return [(vocab.i2w[i], float(sims[i])) for i in top], []


def run_task3():
    print("\n========== TASK 3 : Semantic Analysis ==========")

    for fpath in ['cbow_model.pkl', 'sgns_model.pkl']:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"{fpath} not found, run task 2 first")

    with open('cbow_model.pkl', 'rb') as f:
        cb = pickle.load(f)
    with open('sgns_model.pkl', 'rb') as f:
        sg = pickle.load(f)

    cbow_emb = cb['model'].embeddings
    sgns_emb = sg['model'].embeddings
    vocab    = cb['vocab']

    # --- 3A: nearest neighbours ---
    print(f"\n  Top-{5} nearest neighbours for each probe word")
    print("  (cosine similarity, CBOW vs SGNS side by side)\n")

    for word in PROBE_WORDS:
        cb_nn = nearest_neighbours(word, cbow_emb, vocab, k=5)
        sg_nn = nearest_neighbours(word, sgns_emb, vocab, k=5)

        if not cb_nn:
            print(f"  '{word}' not in vocab, skipping")
            continue

        print(f"  Word: '{word}'")
        print(f"  {'':4} {'CBOW':<22} {'SGNS':<22}")
        print("  " + "-"*50)
        for i, ((cw, cs), (sw, ss)) in enumerate(zip(cb_nn, sg_nn), 1):
            print(f"  {i:<4} {cw:<15} {cs:.4f}    {sw:<15} {ss:.4f}")
        print()

    # --- 3B: analogy experiments ---
    print("\n  Analogy Experiments: a : b :: c : ?")
    print("  (vector arithmetic: W[b] - W[a] + W[c])\n")

    analogies = [
        ('ug', 'btech', 'pg'),
        ('professor', 'research', 'student'),
        ('department', 'faculty', 'course'),
    ]

    for (a, b, c) in analogies:
        print(f"  '{a}' : '{b}' :: '{c}' : ?")
        cb_ans, missing = analogy(a, b, c, cbow_emb, vocab)
        sg_ans, _       = analogy(a, b, c, sgns_emb, vocab)

        if missing:
            print(f"  words not in vocab: {missing}\n")
            continue

        print(f"  {'rank':<5} {'CBOW ans':<18} {'sim':>7}   {'SGNS ans':<18} {'sim':>7}")
        print("  " + "-"*60)
        for i, ((cw, cs), (sw, ss)) in enumerate(zip(cb_ans[:5], sg_ans[:5]), 1):
            print(f"  {i:<5} {cw:<18} {cs:>7.4f}   {sw:<18} {ss:>7.4f}")
        print(f"  best CBOW: '{cb_ans[0][0]}' | best SGNS: '{sg_ans[0][0]}'\n")

    print("Task 3 done.")


# ===================================================================
# TASK 4 - Visualization (PCA and t-SNE)
# ===================================================================

# word groups for visualization
# picked words that should appear in an iitj corpus
VIZ_GROUPS = {
    "academic_roles" : ["professor", "student", "faculty", "researcher",
                        "phd", "postdoc", "undergraduate", "graduate"],
    "programmes"     : ["btech", "mtech", "phd", "msc", "ug",
                        "pg", "programme", "degree"],
    "research"       : ["research", "project", "publication", "thesis",
                        "lab", "experiment", "algorithm", "model"],
    "administration" : ["department", "institute", "campus", "office",
                        "committee", "regulation", "policy", "board"],
}


def collect_vecs(groups, emb, vocab):
    words, vecs, labels = [], [], []
    for grp, wlist in groups.items():
        for w in wlist:
            if w in vocab.w2i:
                words.append(w)
                vecs.append(emb[vocab.w2i[w]])
                labels.append(grp)
    return words, np.array(vecs) if vecs else np.array([]), labels


def scatter_plot(ax, coords, words, labels, title):
    groups   = sorted(set(labels))
    colors   = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    grp_col  = dict(zip(groups, colors))

    for i, (w, lbl) in enumerate(zip(words, labels)):
        x, y = coords[i]
        ax.scatter(x, y, color=grp_col[lbl], s=70, zorder=3,
                   edgecolors='white', linewidths=0.5)
        ax.annotate(w, (x, y), textcoords='offset points',
                    xytext=(4, 3), fontsize=8, alpha=0.9)

    for grp, col in grp_col.items():
        ax.scatter([], [], color=col, label=grp, s=70)

    ax.legend(fontsize=8, framealpha=0.6)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("component 1", fontsize=9)
    ax.set_ylabel("component 2", fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)


def run_task4():
    print("\n========== TASK 4 : Visualization ==========")

    with open('cbow_model.pkl', 'rb') as f:
        cb = pickle.load(f)
    with open('sgns_model.pkl', 'rb') as f:
        sg = pickle.load(f)

    cbow_emb = cb['model'].embeddings
    sgns_emb = sg['model'].embeddings
    vocab    = cb['vocab']

    wc, vc, lc = collect_vecs(VIZ_GROUPS, cbow_emb, vocab)
    ws, vs, ls = collect_vecs(VIZ_GROUPS, sgns_emb, vocab)

    if len(vc) < 5:
        print("  not enough words found in vocab for visualization")
        print("  you may need to update VIZ_GROUPS with words from your corpus")
        return

    print(f"  found {len(vc)} words for CBOW, {len(vs)} for SGNS")

    # PCA
    pca_c = PCA(n_components=2, random_state=42).fit_transform(vc)
    pca_s = PCA(n_components=2, random_state=42).fit_transform(vs)

    # t-SNE - perplexity must be less than n_samples
    perp  = min(30, len(vc) - 1)
    tsne_c = TSNE(n_components=2, perplexity=perp,
                  random_state=42, max_iter=1000).fit_transform(vc)
    tsne_s = TSNE(n_components=2, perplexity=perp,
                  random_state=42, max_iter=1000).fit_transform(vs)

    # 2x2 grid plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Word Embedding Projections - IIT Jodhpur Corpus\n'
                 'CBOW vs Skip-gram | PCA vs t-SNE',
                 fontsize=13, fontweight='bold')

    scatter_plot(axes[0][0], pca_c,  wc, lc, 'CBOW - PCA')
    scatter_plot(axes[0][1], pca_s,  ws, ls, 'Skip-gram (SGNS) - PCA')
    scatter_plot(axes[1][0], tsne_c, wc, lc, 'CBOW - t-SNE')
    scatter_plot(axes[1][1], tsne_s, ws, ls, 'Skip-gram (SGNS) - t-SNE')

    plt.tight_layout()
    plt.savefig('task4_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  plot saved to task4_visualization.png")

    print("""
  Observations for report:
  - PCA is linear so clusters might overlap for similar groups
  - t-SNE preserves local structure better, clearer clusters expected
  - SGNS usually gives tighter clusters than CBOW because each
    context word gets its own gradient update instead of averaging
  - compare how academic_roles and research words cluster differently
    in CBOW vs SGNS
""")

    print("Task 4 done.")


# ===================================================================
# main
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, choices=[1,2,3,4],
                        help='which task to run (default: all)')
    args = parser.parse_args()

    task_map = {1: run_task1, 2: run_task2, 3: run_task3, 4: run_task4}

    if args.task:
        task_map[args.task]()
    else:
        run_task1()
        run_task2()
        run_task3()
        run_task4()
        print("\nAll tasks done!")


if __name__ == '__main__':
    main()
