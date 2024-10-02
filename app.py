import os
import requests
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from config import *

os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:/hf_cache_directory'

processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b").to("cuda" if torch.cuda.is_available() else "cpu")


def read_image_paths_from_directory(directory):
    supported_extensions = (".jpg", ".jpeg", ".png")
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(supported_extensions)]
    return image_paths


def load_image(image_path):
    if image_path.startswith("http"):
        image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
    else: 
        image = Image.open(image_path).convert('RGB')
    print(f"Loaded image: {image_path}")
    return image


def caption_image_conditional(image_path, instruction, processor, model):
    image = load_image(image_path)
    image = image.resize((MODEL_INPUT_IMAGE_SIZE, MODEL_INPUT_IMAGE_SIZE))  # Resize the image to 256x256 pixels
    inputs = processor(images=image, text=instruction, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    outputs = model.generate(**inputs)
    
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption


def caption_images_and_save(image_paths, output_file, processor, model, instruction=None):
    with open(output_file, 'w') as file:
        for image_path in image_paths:
            try:
                if instruction:
                    caption = caption_image_conditional(image_path, instruction, processor, model)
                    file.write(f"Image: {image_path}\nCaption: {caption}\n\n")
                    print(f"Captioned: {image_path}")
                else:
                    print(f"No instruction provided for image: {image_path}")
            except Exception as e:
                print(f"Failed to caption image {image_path}: {e}")


def main(directory, output_file, instruction="Describe the content of this image"):
    image_paths = read_image_paths_from_directory(directory)
    
    caption_images_and_save(image_paths, output_file, processor, model, instruction)


if __name__ == "__main__":
    print(f"Image source: {IMAGE_SOURCE}")
    print(f"Output file: {OUTPUT_FILE}")
    instruction = "Describe this image in detail"
    
    main(IMAGE_SOURCE, OUTPUT_FILE, instruction)
