# Task 3: Image Captioning AI

## CODSOFT Artificial Intelligence Internship

###  Overview
This project combines **Computer Vision** and **Natural Language
Processing** to automatically generate descriptive captions for
images. It ships with **two complementary implementations**:

| File                    | Purpose                                                                 |
|-------------------------|--------------------------------------------------------------------------|
| `caption_demo.py`        | **Ready-to-run inference** using a pre-trained BLIP transformer model. No training required — works immediately. Use this for your demo video. |
| `image_captioning.py`    | **Full trainable architecture** — ResNet-50 CNN encoder + LSTM decoder with attention, built from scratch. Use this to train your own model on a dataset like Flickr8k/COCO. |

This structure demonstrates both an understanding of how image
captioning models are *built* (encoder-decoder + attention) and the
ability to *ship a working product* using state-of-the-art pre-trained
models — a realistic ML engineering workflow.

---

###  Architecture & Model Choices

#### 1. Quick Demo — `caption_demo.py` (BLIP)
- **Model:** [BLIP](https://arxiv.org/abs/2201.12086) (`Salesforce/blip-image-captioning-base`), loaded via Hugging Face `transformers`.
- **Why BLIP:** It's a transformer-based vision-language model
  pre-trained on large-scale image-caption pairs, so it produces
  fluent, accurate captions out-of-the-box — no GPU training run
  needed. This satisfies the task's "transformer-based model" option
  directly.
- **Pipeline:** `image → BLIP visual encoder → transformer decoder → caption text`

#### 2. Full Custom Model — `image_captioning.py` (ResNet + LSTM + Attention)
- **Encoder (CNN):** Pre-trained **ResNet-50** (ImageNet weights) with
  the final classification layers removed. Instead of collapsing to a
  single feature vector, it keeps a spatial feature map (14×14×2048)
  so the decoder can attend to different image regions.
- **Attention:** A Bahdanau-style additive attention module computes,
  at every decoding step, a weighted combination of image regions —
  letting the model "look" at the relevant part of the image for each
  word it generates.
- **Decoder (RNN):** An **LSTM** that takes the previous word embedding
  + attention context at each step and predicts the next word,
  trained end-to-end with **teacher forcing**.
- **Why this combo:** ResNet gives strong, proven visual features;
  attention lets the RNN decoder focus dynamically rather than relying
  on one static global feature vector (a known weakness of vanilla
  CNN→RNN captioning).

##### Pipeline diagram
```
   Image
     │
     ▼
 ResNet-50 (frozen, pre-trained)
     │  spatial feature map (14x14x2048)
     ▼
 Attention  ◄──────────────┐
     │                     │  hidden state
     ▼                     │
   LSTM Decoder ───────────┘
     │  word-by-word
     ▼
 "a dog running through a field"
```

---

###  How to Run

#### Install dependencies
```bash
pip install -r requirements_captioning.txt
```

#### Option A — Instant captioning (recommended for your demo video)
```bash
python caption_demo.py path/to/image1.jpg path/to/image2.jpg
```
Example output:
```
============================================================
IMAGE CAPTIONING RESULTS
============================================================

Image: sample_images/dog.jpg
Caption: a dog running through a grassy field

Image: sample_images/city.jpg
Caption: a city street with tall buildings and cars
============================================================
```

#### Option B — Train your own model from scratch
1. Download a captioning dataset, e.g. [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k).
2. Arrange it as:
   ```
   images/               # all .jpg files
   captions.txt          # format: image_filename,caption text
   ```
3. Train:
   ```python
   from image_captioning import train

   model, vocab = train(
       root_dir="images/",
       captions_file="captions.txt",
       num_epochs=20,
       batch_size=32,
   )
   ```
4. Run inference with your trained model:
   ```python
   from image_captioning import caption_single_image, ImageCaptioningModel, Vocabulary
   import torch

   vocab = Vocabulary.load("vocab.pkl")
   model = ImageCaptioningModel(embed_size=256, vocab_size=len(vocab))
   model.load_state_dict(torch.load("image_captioning_model.pth"))

   caption = caption_single_image("test.jpg", model, vocab)
   print(caption)
   ```

>  Training the custom model to strong quality requires a GPU and
> several hours over a full dataset (COCO/Flickr8k/Flickr30k). If you
> don't have GPU access, `caption_demo.py` gives you a fully working,
> high-quality captioning system for your submission and demo video.

---

###  Files
```
image_captioning.py           # Full CNN+LSTM+Attention architecture & training loop
caption_demo.py                # Ready-to-run inference using pre-trained BLIP
requirements_captioning.txt    # Python dependencies
README_imagecaptioning.md      # Project documentation (this file)
```

---

###  Sample Results
| Image                | Generated Caption                                  |
|-----------------------|------------------------------------------------------|
| A dog in a park        | "a dog running through a grassy field"              |
| A city skyline          | "a city street with tall buildings and cars"        |
| A plate of food          | "a plate of food with rice and vegetables on a table" |

*(Add your own screenshots/sample images here after running the demo.)*

---

###  Error Handling
`caption_demo.py` handles common failure cases gracefully:
- Missing/invalid file paths → raises a clear `FileNotFoundError`
- Corrupted or unreadable image files → caught and reported per-image
  in `caption_batch()` without crashing the whole batch
- Automatically uses GPU if available, otherwise falls back to CPU

---

###  Possible Extensions
- Fine-tune BLIP on a domain-specific dataset (e.g. medical images, products)
- Replace the LSTM decoder in `image_captioning.py` with a Transformer decoder
- Add BLEU/METEOR/CIDEr evaluation metrics against ground-truth captions
- Deploy as a simple Flask/Streamlit web app for drag-and-drop captioning

---

###  Internship Info
This task was completed as part of the **CODSOFT Artificial Intelligence
Internship**. Repository maintained under: `CODSOFT`

#codsoft #internship #artificialintelligence #computervision #nlp
