#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ví dụ về cách lắp ráp một pipeline suy luận Stable Diffusion từ các thành phần riêng biệt:
- VAE (Variational Autoencoder)
- UNet
- CLIP Text Encoder
- Scheduler
"""

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import argparse
from tqdm.auto import tqdm
import logging
import os
from datetime import datetime

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("custom_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("StableDiffusion")

def create_custom_pipeline(device="cuda"):
    """
    Lắp ráp các thành phần của Stable Diffusion thành một pipeline tùy chỉnh
    """
    logger.info("Đang tải các thành phần của Stable Diffusion...")
    
    # 1. Tải mô hình VAE để mã hóa/giải mã biểu diễn tiềm ẩn
    logger.info("Tải VAE model...")
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    
    # 2. Tải tokenizer và text encoder để mã hóa văn bản
    logger.info("Tải CLIP tokenizer và text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    
    # 3. Tải mô hình UNet để tạo biểu diễn tiềm ẩn của ảnh
    logger.info("Tải UNet model...")
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
    
    # 4. Tải scheduler - ở đây chúng ta sử dụng LMS thay vì PNDM mặc định
    logger.info("Tải LMS scheduler...")
    scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    
    # Chuyển các mô hình đến thiết bị
    logger.info(f"Chuyển các mô hình đến thiết bị: {device}")
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    
    return vae, text_encoder, tokenizer, unet, scheduler

def create_latents(prompt, tokenizer, text_encoder, unet, height, width, batch_size, device, seed=None):
    """
    Tạo text embeddings và latents ban đầu
    """
    logger.info(f"Tạo latents với seed: {seed if seed is not None else 'random'}")
    
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        logger.info(f"Đã thiết lập seed: {seed}")
    else:
        generator = torch.Generator(device=device)
        logger.info("Sử dụng seed ngẫu nhiên")
    
    # Tạo text embeddings
    logger.info("Tokenizing prompt...")
    text_input = tokenizer(
        prompt, 
        padding="max_length", 
        max_length=tokenizer.model_max_length, 
        truncation=True, 
        return_tensors="pt"
    )
    
    logger.info("Tạo text embeddings...")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    
    # Tạo unconditional embeddings cho classifier-free guidance
    logger.info("Tạo unconditional embeddings...")
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, 
        padding="max_length", 
        max_length=max_length, 
        return_tensors="pt"
    )
    
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    
    # Ghép text embeddings
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    # Tạo latents ngẫu nhiên ban đầu
    logger.info(f"Tạo latents với kích thước: {batch_size}x{unet.in_channels}x{height // 8}x{width // 8}")
    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
        device=device  # Specify the device when creating the tensor
    )
    
    return latents, text_embeddings, generator

def generate_image(
    prompt,
    vae, 
    text_encoder, 
    tokenizer, 
    unet, 
    scheduler,
    height=512,
    width=512,
    guidance_scale=7.5,
    num_inference_steps=50,
    seed=None,
    device="cuda",
    batch_size=1
):
    """
    Tạo ảnh từ prompt sử dụng pipeline tùy chỉnh
    """
    logger.info(f"Bắt đầu quá trình tạo ảnh với tham số:")
    logger.info(f"- Prompt: '{prompt}'")
    logger.info(f"- Kích thước: {width}x{height}")
    logger.info(f"- Guidance scale: {guidance_scale}")
    logger.info(f"- Số bước: {num_inference_steps}")
    logger.info(f"- Device: {device}")
    
    # 1. Tạo text embeddings và latents ban đầu
    latents, text_embeddings, generator = create_latents(
        prompt, tokenizer, text_encoder, unet, 
        height, width, batch_size, device, seed
    )
    
    # 2. Cài đặt scheduler với số bước suy luận
    logger.info(f"Cài đặt scheduler với {num_inference_steps} bước")
    scheduler.set_timesteps(num_inference_steps)
    
    # 3. Chuẩn bị latents cho quá trình khử nhiễu
    latents = latents * scheduler.init_noise_sigma
    
    # 4. Vòng lặp khử nhiễu
    logger.info(f"Bắt đầu quá trình khử nhiễu...")
    for t in tqdm(scheduler.timesteps):
        # Nhân đôi latents cho classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        
        # Điều chỉnh scale input theo timestep
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        # Dự đoán noise residual
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input, 
                t, 
                encoder_hidden_states=text_embeddings
            ).sample
        
        # Thực hiện classifier-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Khử nhiễu: bước t -> t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # 5. Scale và giải mã biểu diễn tiềm ẩn
    logger.info("Giải mã latents thành ảnh...")
    latents = 1 / 0.18215 * latents
    
    with torch.no_grad():
        image = vae.decode(latents).sample
    
    # 6. Chuyển đổi từ tensor sang định dạng ảnh PIL
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    logger.info("Đã chuyển đổi thành công tensor thành ảnh PIL")
    
    return pil_images

def main():
    # Phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Custom Stable Diffusion Pipeline")
    parser.add_argument("--prompt", type=str, default="a photo of an astronaut riding a horse on mars",
                       help="Mô tả văn bản để tạo ảnh")
    parser.add_argument("--height", type=int, default=512,
                       help="Chiều cao ảnh")
    parser.add_argument("--width", type=int, default=512,
                       help="Chiều rộng ảnh")
    parser.add_argument("--steps", type=int, default=50,
                       help="Số bước suy luận")
    parser.add_argument("--guidance", type=float, default=7.5,
                       help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None,
                       help="Seed ngẫu nhiên để tái tạo ảnh")
    parser.add_argument("--output", type=str, default="outputs/custom_output.png",
                       help="Tên file kết quả")
    
    args = parser.parse_args()
    
    # Xác định thiết bị
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("Cảnh báo: Bạn đang sử dụng CPU, quá trình sinh ảnh sẽ rất chậm!")
    
    # Tạo pipeline tùy chỉnh
    vae, text_encoder, tokenizer, unet, scheduler = create_custom_pipeline(device)
    
    # Sinh ảnh
    images = generate_image(
        args.prompt,
        vae, 
        text_encoder, 
        tokenizer, 
        unet, 
        scheduler,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        seed=args.seed,
        device=device
    )
    
    # Tạo tên file kết quả với guidance và steps
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Đã tạo thư mục đầu ra: {output_dir}")
    
    filename = os.path.basename(args.output)
    name, ext = os.path.splitext(filename)
    
    # Tạo tên file mới với guidance và steps
    new_filename = f"{name}_g{args.guidance}_s{args.steps}{ext}"
    output_path = os.path.join(output_dir, new_filename)
    
    # Lưu ảnh
    images[0].save(output_path)
    logger.info(f"Đã lưu ảnh tại: {output_path}")

if __name__ == "__main__":
    logger.info("Khởi động pipeline...")
    main()
    logger.info("Hoàn thành!")
