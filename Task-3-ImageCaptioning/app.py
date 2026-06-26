from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

# Load pretrained model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_caption(image_path):
    image = Image.open(image_path)

    if image.mode != "RGB":
        image = image.convert("RGB")

    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, max_length=16)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return caption

# Run program
image_path = input("Enter image path: ")
caption = generate_caption(image_path)

print("\nGenerated Caption:", caption)
