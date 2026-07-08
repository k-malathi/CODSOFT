import os
import pickle
from collections import Counter

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence


# =====================================================================
# 1. VOCABULARY
# Converts words <-> integer indices so captions can be fed to the LSTM.
# =====================================================================
class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        # Simple whitespace tokenizer (lowercased, punctuation-stripped)
        return text.lower().replace(".", "").replace(",", "").strip().split()

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4  # 0-3 reserved for special tokens

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized = self.tokenize(text)
        return [
            self.stoi.get(token, self.stoi["<unk>"]) for token in tokenized
        ]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


# =====================================================================
# 2. DATASET
# Expects a captions file with lines: "image_filename,caption text"
# (this matches the common Flickr8k/Flickr30k format)
# =====================================================================
class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir, captions_file, vocab, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.vocab = vocab

        self.image_paths = []
        self.captions = []
        with open(captions_file, "r", encoding="utf-8") as f:
            next(f)  # skip header if present
            for line in f:
                if "," not in line:
                    continue
                img_name, caption = line.strip().split(",", 1)
                self.image_paths.append(img_name)
                self.captions.append(caption)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_path = os.path.join(self.root_dir, self.image_paths[index])
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        numericalized_caption = [self.vocab.stoi["<start>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<end>"])

        return image, torch.tensor(numericalized_caption)


def collate_fn(batch, pad_idx):
    """Pads variable-length captions within a batch to the same length."""
    images = [item[0].unsqueeze(0) for item in batch]
    images = torch.cat(images, dim=0)

    captions = [item[1] for item in batch]
    lengths = [len(cap) for cap in captions]
    padded_captions = torch.full(
        (len(captions), max(lengths)), pad_idx, dtype=torch.long
    )
    for i, cap in enumerate(captions):
        padded_captions[i, : len(cap)] = cap

    return images, padded_captions, torch.tensor(lengths)


# =====================================================================
# 3. ENCODER — Pre-trained ResNet-50 (CNN feature extractor)
# =====================================================================
class EncoderCNN(nn.Module):
    """
    Uses a pre-trained ResNet-50 with the final classification layer
    removed, keeping the spatial feature map (for attention) instead
    of collapsing to a single vector.
    """

    def __init__(self, encoded_image_size=14, fine_tune=False):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Drop the last two layers (avgpool + fc) to keep spatial feature maps
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune(fine_tune)

    def forward(self, images):
        # images: (batch, 3, H, W)
        features = self.resnet(images)               # (batch, 2048, H', W')
        features = self.adaptive_pool(features)       # (batch, 2048, enc_size, enc_size)
        features = features.permute(0, 2, 3, 1)        # (batch, enc_size, enc_size, 2048)
        batch_size = features.size(0)
        features = features.view(batch_size, -1, features.size(-1))  # (batch, num_pixels, 2048)
        return features

    def fine_tune(self, fine_tune=False):
        """By default, freeze ResNet weights (only train the decoder)."""
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Optionally fine-tune the last conv block for better performance
        if fine_tune:
            for module in list(self.resnet.children())[6:]:
                for param in module.parameters():
                    param.requires_grad = True


# =====================================================================
# 4. ATTENTION MODULE (Bahdanau-style additive attention)
# =====================================================================
class Attention(nn.Module):
    """
    At each decoding step, computes a weighted sum of image regions,
    letting the model "look at" the most relevant part of the image
    for the next word it's about to generate.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: (batch, num_pixels, encoder_dim)
        # decoder_hidden: (batch, decoder_dim)
        att1 = self.encoder_att(encoder_out)                     # (batch, num_pixels, attn_dim)
        att2 = self.decoder_att(decoder_hidden)                  # (batch, attn_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch, num_pixels)
        alpha = self.softmax(att)                                # attention weights
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha


# =====================================================================
# 5. DECODER — LSTM with Attention (generates captions word by word)
# =====================================================================
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, encoder_dim=2048,
                 decoder_dim=512, attention_dim=256, dropout=0.5):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim)

        # Initialize hidden/cell state from the mean image features
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # Gate that scales how much attention context to use (soft gating)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, captions, lengths):
        """
        encoder_out: (batch, num_pixels, encoder_dim) from EncoderCNN
        captions: (batch, max_len) numericalized ground-truth captions
        lengths: actual lengths of each caption (for teacher forcing)
        """
        batch_size = encoder_out.size(0)
        vocab_size = self.vocab_size

        embeddings = self.embedding(captions)  # (batch, max_len, embed_size)
        h, c = self.init_hidden_state(encoder_out)

        # We won't decode the <end> token, so max decode length = lengths - 1
        decode_lengths = [length - 1 for length in lengths]
        max_dec_len = max(decode_lengths)

        predictions = torch.zeros(batch_size, max_dec_len, vocab_size, device=encoder_out.device)

        for t in range(max_dec_len):
            batch_size_t = sum(l > t for l in decode_lengths)
            attention_weighted_encoding, _ = self.attention(
                encoder_out[:batch_size_t], h[:batch_size_t]
            )
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding

            lstm_input = torch.cat(
                [embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1
            )
            h_t, c_t = self.lstm_cell(lstm_input, (h[:batch_size_t], c[:batch_size_t]))
            h = torch.cat([h_t, h[batch_size_t:]], dim=0) if batch_size_t < batch_size else h_t
            c = torch.cat([c_t, c[batch_size_t:]], dim=0) if batch_size_t < batch_size else c_t

            preds = self.fc(self.dropout(h_t))
            predictions[:batch_size_t, t, :] = preds

        return predictions, decode_lengths

    def generate_caption(self, encoder_out, vocab, max_len=20):
        """Greedy decoding: generate a caption for a single image at inference time."""
        h, c = self.init_hidden_state(encoder_out)
        word = torch.tensor([vocab.stoi["<start>"]], device=encoder_out.device)
        caption = []

        for _ in range(max_len):
            embed = self.embedding(word)  # (1, embed_size)
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding

            lstm_input = torch.cat([embed, attention_weighted_encoding], dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            preds = self.fc(h)
            predicted = preds.argmax(1)

            token = vocab.itos[predicted.item()]
            if token == "<end>":
                break
            if token not in ("<start>", "<pad>"):
                caption.append(token)
            word = predicted

        return " ".join(caption)


# =====================================================================
# 6. FULL MODEL WRAPPER
# =====================================================================
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, vocab_size, encoder_dim=2048,
                 decoder_dim=512, attention_dim=256, fine_tune_encoder=False):
        super().__init__()
        self.encoder = EncoderCNN(fine_tune=fine_tune_encoder)
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size=vocab_size,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim,
        )

    def forward(self, images, captions, lengths):
        encoder_out = self.encoder(images)
        predictions, decode_lengths = self.decoder(encoder_out, captions, lengths)
        return predictions, decode_lengths

    def caption_image(self, image_tensor, vocab, max_len=20):
        self.eval()
        with torch.no_grad():
            encoder_out = self.encoder(image_tensor)
            return self.decoder.generate_caption(encoder_out, vocab, max_len)


# =====================================================================
# 7. IMAGE TRANSFORM (standard ImageNet normalization for ResNet input)
# =====================================================================
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


# =====================================================================
# 8. TRAINING LOOP (skeleton — requires a captions dataset, e.g. Flickr8k)
# =====================================================================
def train(
    root_dir,
    captions_file,
    num_epochs=5,
    batch_size=32,
    embed_size=256,
    learning_rate=3e-4,
    freq_threshold=5,
    save_path="image_captioning_model.pth",
    vocab_path="vocab.pkl",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build vocabulary from all captions
    with open(captions_file, "r", encoding="utf-8") as f:
        next(f)
        all_captions = [line.strip().split(",", 1)[1] for line in f if "," in line]

    vocab = Vocabulary(freq_threshold=freq_threshold)
    vocab.build_vocabulary(all_captions)
    vocab.save(vocab_path)
    print(f"Vocabulary size: {len(vocab)}")

    dataset = ImageCaptionDataset(root_dir, captions_file, vocab, transform=image_transform)
    pad_idx = vocab.stoi["<pad>"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
    )

    model = ImageCaptioningModel(embed_size=embed_size, vocab_size=len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
    )

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, captions, lengths in loader:
            images, captions = images.to(device), captions.to(device)

            predictions, decode_lengths = model(images, captions, lengths)
            targets = captions[:, 1:]  # shift right (skip <start>)
            targets = pack_padded_sequence(
                targets, decode_lengths, batch_first=True, enforce_sorted=False
            ).data
            predictions = pack_padded_sequence(
                predictions, decode_lengths, batch_first=True, enforce_sorted=False
            ).data

            loss = criterion(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model, vocab


# =====================================================================
# 9. INFERENCE HELPER (once a model is trained/loaded)
# =====================================================================
def caption_single_image(image_path, model, vocab, device="cpu"):
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(device)
    model.to(device)
    return model.caption_image(image_tensor, vocab)


if __name__ == "__main__":
    print("This file defines the model architecture and training pipeline.")
    print("To train: call train(root_dir='images/', captions_file='captions.txt')")
    print("For a quick, no-training demo using a pre-trained model, run caption_demo.py")
