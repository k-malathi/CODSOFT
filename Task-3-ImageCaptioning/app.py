import sys
import os

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class ImageCaptioner:
    """Wraps a pre-trained BLIP model behind a simple .caption(image_path) API."""

    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model '{model_name}' on {self.device} ...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Model loaded successfully.\n")

    def caption(self, image_path, conditional_prompt=None, max_new_tokens=30):
        """
        Generate a caption for a single image.

        Args:
            image_path: path to a local image file (jpg/png/etc.)
            conditional_prompt: optional text prefix to steer the caption,
                e.g. "a photo of" -- pass None for unconditional captioning
            max_new_tokens: maximum caption length

        Returns:
            A string caption describing the image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Could not open image '{image_path}': {e}")

        if conditional_prompt:
            inputs = self.processor(image, conditional_prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()

    def caption_batch(self, image_paths):
        """Generate captions for a list of images, handling errors per-image."""
        results = {}
        for path in image_paths:
            try:
                results[path] = self.caption(path)
            except Exception as e:
                results[path] = f"[ERROR] {e}"
        return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python caption_demo.py <image_path> [image_path2 ...]")
        print("Example: python caption_demo.py sample_images/dog.jpg")
        sys.exit(1)

    image_paths = sys.argv[1:]
    captioner = ImageCaptioner()

    print("=" * 60)
    print("IMAGE CAPTIONING RESULTS")
    print("=" * 60)

    results = captioner.caption_batch(image_paths)
    for path, caption in results.items():
        print(f"\nImage: {path}")
        print(f"Caption: {caption}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
