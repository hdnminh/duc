#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sinh ảnh từ văn bản với Stable Diffusion sử dụng thư viện diffusers của Hugging Face
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import argparse
import os

def image_grid(imgs, rows, cols):
    """Tạo lưới ảnh từ danh sách các ảnh"""
    assert len(imgs) == rows*cols, "Số lượng ảnh phải bằng rows*cols"
    
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def setup_pipeline():
    """Khởi tạo và cài đặt Stable Diffusion pipeline"""
    print("Đang tải mô hình Stable Diffusion...")
    
    # Xác định thiết bị phù hợp (GPU hoặc CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch_dtype = torch.float16  # Sử dụng float16 để tiết kiệm bộ nhớ GPU
    elif torch.backends.mps.is_available():  # Hỗ trợ Apple Silicon
        device = torch.device("mps")
        torch_dtype = torch.float32
    else:
        device = torch.device("cpu")
        torch_dtype = torch.float32
        print("Cảnh báo: Đang sử dụng CPU, quá trình sinh ảnh sẽ rất chậm!")
    
    # Tải pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        torch_dtype=torch_dtype
    )
    pipe = pipe.to(device)
    
    return pipe, device

def create_image_from_text(pipe, prompt, guidance_scale=7.5, steps=50, height=512, width=512, seed=None):
    """Sinh ảnh từ mô tả văn bản"""
    # Thiết lập generator nếu có seed
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    else:
        generator = None
    
    # Sinh ảnh
    image = pipe(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        height=height,
        width=width,
        generator=generator
    ).images[0]
    
    return image

def create_multiple_images(pipe, prompt, count=4, guidance_scale=7.5, steps=30, height=512, width=512):
    """Sinh nhiều ảnh từ cùng một mô tả văn bản"""
    # Lặp lại prompt thành list với số lượng bằng count
    prompts = [prompt] * count
    
    # Sinh nhiều ảnh cùng lúc
    images = pipe(
        prompts,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        height=height,
        width=width
    ).images
    
    return images

def save_image(image, filename="output.png", output_dir="outputs"):
    """Lưu ảnh vào thư mục"""
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Đường dẫn đầy đủ
    filepath = os.path.join(output_dir, filename)
    
    # Lưu ảnh
    image.save(filepath)
    print(f"Đã lưu ảnh tại: {filepath}")
    return filepath

def main():
    """Hàm chính"""
    # Phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Sinh ảnh từ văn bản bằng Stable Diffusion")
    parser.add_argument("--prompt", type=str, default="a beautiful landscape with mountains and a lake",
                        help="Mô tả văn bản để tạo ảnh")
    parser.add_argument("--num", type=int, default=1, 
                        help="Số lượng ảnh cần tạo")
    parser.add_argument("--guidance", type=float, default=7.5,
                        help="Guidance scale (1-20)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Số bước khử nhiễu")
    parser.add_argument("--height", type=int, default=512,
                        help="Chiều cao ảnh (nên là bội số của 8)")
    parser.add_argument("--width", type=int, default=512,
                        help="Chiều rộng ảnh (nên là bội số của 8)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed ngẫu nhiên để tái tạo ảnh")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Thư mục lưu ảnh kết quả")
    
    args = parser.parse_args()
    
    # Thiết lập pipeline
    pipe, device = setup_pipeline()
    
    # Sinh ảnh
    if args.num == 1:
        # Tạo một ảnh
        image = create_image_from_text(
            pipe, 
            args.prompt, 
            guidance_scale=args.guidance,
            steps=args.steps,
            height=args.height,
            width=args.width,
            seed=args.seed
        )
        save_image(image, "result.png", args.output)
    else:
        # Tạo nhiều ảnh
        images = create_multiple_images(
            pipe,
            args.prompt,
            count=args.num,
            guidance_scale=args.guidance,
            steps=args.steps,
            height=args.height,
            width=args.width
        )
        
        # Lưu từng ảnh riêng
        for i, img in enumerate(images):
            save_image(img, f"result_{i+1}.png", args.output)
        
        # Nếu có từ 2-4 ảnh, tạo lưới ảnh
        if 2 <= args.num <= 4:
            rows = 1 if args.num <= 2 else 2
            cols = 2
            grid = image_grid(images[:rows*cols], rows, cols)
            save_image(grid, "result_grid.png", args.output)

if __name__ == "__main__":
    main()
    print("Hoàn thành!") 