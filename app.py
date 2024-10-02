import os
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


model_id = "xtuner/llava-llama-3-8b-v1_1-transformers"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
).to("cuda")

processor = AutoProcessor.from_pretrained(model_id)

def load_image(image_path):
    """Load an image from a local file path."""
    if os.path.exists(image_path):
        return Image.open(image_path).convert('RGB')
    else:
        print(f"Error: The file '{image_path}' does not exist.")
        return None

def chat_with_ai(image_path=None):
    """Facilitate continuous chat with the AI model."""

    prompt = "<|start_header_id|>user<|end_header_id|>\n\n<image>\nWhat are these?<|eot_id|>" \
             "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    raw_image = load_image(image_path) if image_path else None

    while True:
        if raw_image:
            inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)
        else:
            user_query = input("You: ")
            if user_query.lower() in ['exit', 'quit']:
                print("Exiting the chat.")
                break

            prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_query}\n<|eot_id|>" \
                      "<|start_header_id|>assistant<|end_header_id|>\n\n"

            inputs = processor(prompt, return_tensors='pt').to(0, torch.float16)

        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        response = processor.decode(output[0][2:], skip_special_tokens=True)
        
        print(f"AI: {response}")

        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{response}\n<|eot_id|>"


if __name__ == "__main__":
    image_path = input("Enter the local image file path (or press enter to chat without an image): ")
    if image_path.strip():
        chat_with_ai(image_path.strip())
    else:
        chat_with_ai()

