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
import logging
from datetime import datetime
import re

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stable_diffusion_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("StableDiffusionPipeline")

def image_grid(imgs, rows, cols):
    """Tạo lưới ảnh từ danh sách các ảnh"""
    assert len(imgs) == rows*cols, "Số lượng ảnh phải bằng rows*cols"
    
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    logger.info(f"Đã tạo lưới ảnh với {rows} hàng và {cols} cột")
    return grid

def setup_pipeline():
    """Khởi tạo và cài đặt Stable Diffusion pipeline"""
    logger.info("Đang tải mô hình Stable Diffusion...")
    
    # Xác định thiết bị phù hợp (GPU hoặc CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch_dtype = torch.float16  # Sử dụng float16 để tiết kiệm bộ nhớ GPU
        logger.info("Sử dụng CUDA với dtype float16")
    elif torch.backends.mps.is_available():  # Hỗ trợ Apple Silicon
        device = torch.device("mps")
        torch_dtype = torch.float32
        logger.info("Sử dụng Apple Silicon (MPS) với dtype float32")
    else:
        device = torch.device("cpu")
        torch_dtype = torch.float32
        logger.warning("Cảnh báo: Đang sử dụng CPU, quá trình sinh ảnh sẽ rất chậm!")
    
    # Tải pipeline
    logger.info("Tải StableDiffusionPipeline từ runwayml/stable-diffusion-v1-5...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch_dtype
    )
    pipe = pipe.to(device)
    logger.info(f"Đã tải pipeline và chuyển đến thiết bị {device}")
    
    return pipe, device

def create_image_from_text(pipe, prompt, guidance_scale=7.5, steps=50, height=512, width=512, seed=None):
    """Sinh ảnh từ mô tả văn bản"""
    logger.info(f"Bắt đầu sinh ảnh với prompt: '{prompt}'")
    logger.info(f"Tham số: guidance_scale={guidance_scale}, steps={steps}, kích thước={width}x{height}")
    
    # Thiết lập generator nếu có seed
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        logger.info(f"Sử dụng seed: {seed}")
    else:
        generator = None
        logger.info("Sử dụng seed ngẫu nhiên")
    
    # Sinh ảnh
    logger.info("Đang thực hiện quá trình sinh ảnh...")
    image = pipe(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        height=height,
        width=width,
        generator=generator
    ).images[0]
    logger.info("Đã sinh ảnh thành công")
    
    return image

def create_multiple_images(pipe, prompt, count=4, guidance_scale=7.5, steps=30, height=512, width=512):
    """Sinh nhiều ảnh từ cùng một mô tả văn bản"""
    logger.info(f"Bắt đầu sinh {count} ảnh với prompt: '{prompt}'")
    logger.info(f"Tham số: guidance_scale={guidance_scale}, steps={steps}, kích thước={width}x{height}")
    
    # Lặp lại prompt thành list với số lượng bằng count
    prompts = [prompt] * count
    
    # Sinh nhiều ảnh cùng lúc
    logger.info(f"Đang sinh {count} ảnh cùng lúc...")
    images = pipe(
        prompts,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        height=height,
        width=width
    ).images
    logger.info(f"Đã sinh {len(images)} ảnh thành công")
    
    return images

def save_image(image, filename="output.png", output_dir="outputs", guidance=7.5, steps=50, prompt=""):
    """Lưu ảnh vào thư mục với tên file bao gồm thông tin prompt, guidance và steps"""
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Đã tạo thư mục đầu ra: {output_dir}")
    
    # Xử lý prompt để làm tên file
    # Loại bỏ các ký tự không hợp lệ và giới hạn độ dài
    safe_prompt = re.sub(r'[^\w\s-]', '', prompt)  # Loại bỏ ký tự đặc biệt
    safe_prompt = re.sub(r'\s+', '_', safe_prompt.strip())  # Thay khoảng trắng bằng gạch dưới
    safe_prompt = safe_prompt[:30]  # Giới hạn độ dài
    
    # Tách tên file và phần mở rộng
    name, ext = os.path.splitext(filename)
    
    # Tạo tên file mới bao gồm prompt, guidance và steps
    if safe_prompt:
        new_filename = f"{name}_{safe_prompt}_g{guidance}_s{steps}{ext}"
    else:
        new_filename = f"{name}_g{guidance}_s{steps}{ext}"
    
    # Đường dẫn đầy đủ
    filepath = os.path.join(output_dir, new_filename)
    
    # Lưu ảnh
    image.save(filepath)
    logger.info(f"Đã lưu ảnh tại: {filepath}")
    return filepath

def main():
    """Hàm chính"""
    logger.info("Bắt đầu chương trình Stable Diffusion")
    
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
    logger.info(f"Tham số: {args}")
    
    # Thiết lập pipeline
    pipe, device = setup_pipeline()
    
    # Sinh ảnh
    if args.num == 1:
        # Tạo một ảnh
        logger.info("Chế độ: Tạo một ảnh duy nhất")
        image = create_image_from_text(
            pipe, 
            args.prompt, 
            guidance_scale=args.guidance,
            steps=args.steps,
            height=args.height,
            width=args.width,
            seed=args.seed
        )
        save_image(
            image, 
            "result.png", 
            args.output, 
            guidance=args.guidance, 
            steps=args.steps,
            prompt=args.prompt
        )
    else:
        # Tạo nhiều ảnh
        logger.info(f"Chế độ: Tạo {args.num} ảnh")
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
        logger.info("Lưu từng ảnh riêng lẻ")
        for i, img in enumerate(images):
            save_image(
                img, 
                f"result_{i+1}.png", 
                args.output, 
                guidance=args.guidance, 
                steps=args.steps,
                prompt=args.prompt
            )
        
        # Nếu có từ 2-4 ảnh, tạo lưới ảnh
        if 2 <= args.num <= 4:
            logger.info(f"Tạo lưới ảnh từ {args.num} ảnh")
            rows = 1 if args.num <= 2 else 2
            cols = 2
            grid = image_grid(images[:rows*cols], rows, cols)
            save_image(
                grid, 
                "result_grid.png", 
                args.output, 
                guidance=args.guidance, 
                steps=args.steps,
                prompt=args.prompt
            )

if __name__ == "__main__":
    try:
        main()
        logger.info("Chương trình hoàn thành thành công!")
    except Exception as e:
        logger.exception("Có lỗi xảy ra trong quá trình thực thi:")
    print("Hoàn thành!")
    