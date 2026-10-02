# pip install diffusers transformers torch pillow accelerate

import torch
from diffusers import StableDiffusionInpaintingPipeline
from PIL import Image, ImageDraw

MODEL_ID = "runwayml/stable-diffusion-inpainting"
pipe = StableDiffusionInpaintingPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")

IMAGE_PATH = "/mnt/data/wa_image_100575952248205841" 
init_image = Image.open(IMAGE_PATH).convert("RGB")
init_image = init_image.resize((512, 512)) #
mask = Image.new("L", (512, 512), 0) 
draw = ImageDraw.Draw(mask)
draw.rectangle([100, 100, 200, 250], fill=255)
draw.rectangle([300, 350, 380, 400], fill=255) 

mask.save("01_mask.png")
print("Mask saved as 01_mask.png - White area will be AI-healed")

prompt = "a vintage photograph restored, sharp, detailed face, clean skin, high quality, photorealistic, no scratches, no damage"

negative_prompt = "blurry, damaged, scratches, spots, torn, low quality, deformed"
result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    mask_image=mask,
    guidance_scale=7.5, 
    num_inference_steps=50, 
    strength=0.99 
).images[0]

result.save("02_healed_vintage.png")
result.show()
print("✅ Healed image saved as 02_healed_vintage.png")

from transformers import DetrImageProcessor, DetrForObjectDetection

print("\n--- Bonus: Object Detection Version ---")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
inputs = processor(images=init_image, return_tensors="pt")
outputs = model(**inputs)
target_sizes = torch.tensor([init_image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    print(f"Detected {model.config.id2label[label.item()]} with confidence {score:.2f} at {box}")