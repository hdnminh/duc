# Sinh ảnh từ văn bản với Stable Diffusion

Dự án này cung cấp các công cụ để sinh ảnh từ văn bản sử dụng mô hình Stable Diffusion với thư viện diffusers của Hugging Face.

## Yêu cầu

Trước khi bắt đầu, bạn cần cài đặt các thư viện sau:

```bash
pip install diffusers==0.31.0
pip install transformers==4.46.1 scipy ftfy accelerate
```

Để có hiệu suất tốt nhất, bạn nên sử dụng GPU với ít nhất 8GB VRAM. Nếu sử dụng Google Colab, hãy đảm bảo chọn runtime GPU.

## Các tệp tin trong dự án

Dự án bao gồm hai tệp tin chính:

1. **stable_diffusion_text2img.py**: Script cơ bản để sinh ảnh từ văn bản sử dụng StableDiffusionPipeline.
2. **custom_pipeline_example.py**: Ví dụ về cách lắp ráp một pipeline tùy chỉnh từ các thành phần riêng biệt (VAE, UNet, CLIP text encoder, Scheduler).

## Cách sử dụng

### 1. Sử dụng StableDiffusionPipeline

```bash
python stable_diffusion_text2img.py --prompt "cảnh biển với hoàng hôn đẹp" --guidance 7.5 --steps 50
```

Các tham số:
- `--prompt`: Mô tả văn bản (mặc định: "a beautiful landscape with mountains and a lake")
- `--num`: Số lượng ảnh cần tạo (mặc định: 1)
- `--guidance`: Guidance scale (mặc định: 7.5, phạm vi tốt: 1-20)
- `--steps`: Số bước khử nhiễu (mặc định: 50)
- `--height`: Chiều cao ảnh (mặc định: 512, nên là bội số của 8)
- `--width`: Chiều rộng ảnh (mặc định: 512, nên là bội số của 8)
- `--seed`: Seed ngẫu nhiên để tái tạo kết quả (mặc định: None)
- `--output`: Thư mục đầu ra (mặc định: "outputs")

### 2. Sử dụng pipeline tùy chỉnh

```bash
python custom_pipeline_example.py --prompt "thành phố tương lai với xe bay" --guidance 8.0 --steps 50
```

Các tham số:
- `--prompt`: Mô tả văn bản (mặc định: "a photo of an astronaut riding a horse on mars")
- `--height`: Chiều cao ảnh (mặc định: 512)
- `--width`: Chiều rộng ảnh (mặc định: 512)
- `--steps`: Số bước suy luận (mặc định: 50)
- `--guidance`: Guidance scale (mặc định: 7.5)
- `--seed`: Seed ngẫu nhiên (mặc định: None)
- `--output`: Tên file kết quả (mặc định: "custom_output.png")

## Hiểu về các tham số

### Guidance Scale
- Điều chỉnh mức độ tuân thủ theo mô tả văn bản
- Giá trị cao hơn (7-10) tạo ảnh khớp hơn với prompt nhưng ít đa dạng
- Giá trị thấp hơn (1-5) tạo ảnh đa dạng hơn nhưng có thể ít khớp với prompt

### Số bước suy luận (Steps)
- Điều chỉnh số bước khử nhiễu
- Nhiều bước hơn (50-100) tạo ảnh chi tiết hơn nhưng tốn thời gian hơn
- Ít bước hơn (20-30) nhanh hơn nhưng có thể kém chi tiết

### Kích thước ảnh
- Stable Diffusion được đào tạo với kích thước 512x512
- Các kích thước khác nhau có thể tạo ra kết quả tốt, nhưng nên giữ một chiều ở 512
- Chiều cao và chiều rộng phải là bội số của 8

## Lưu ý

- Sinh ảnh bằng Stable Diffusion đòi hỏi tài nguyên GPU lớn.
- Thời gian sinh mỗi ảnh phụ thuộc vào phần cứng và số bước suy luận.
- Để có kết quả tốt nhất, nên dùng mô tả chi tiết bằng tiếng Anh.
- Nếu muốn tái tạo chính xác một ảnh, hãy sử dụng cùng seed và tham số. 