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

# Hướng Dẫn Sử Dụng Pipeline Stable Diffusion Tùy Chỉnh

## Giới thiệu

Repository này chứa mã nguồn tùy chỉnh cho việc lắp ráp pipeline Stable Diffusion từ các thành phần riêng biệt. Thay vì sử dụng pipeline đóng gói sẵn, chúng ta xây dựng từng bước của quy trình khử nhiễu để tạo ra hình ảnh từ văn bản.

## Hai Cách Lắp Ráp Pipeline Stable Diffusion

### 1. Sử Dụng Pipeline Đóng Gói (stable_diffusion_text2img.py)

Cách này sử dụng `StableDiffusionPipeline` từ thư viện diffusers, đơn giản và nhanh chóng:

```python
# Tải pipeline đóng gói
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch_dtype
)
pipe = pipe.to(device)

# Sinh ảnh chỉ với một lệnh gọi
image = pipe(
    prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=steps,
    height=height,
    width=width,
    generator=generator
).images[0]
```

**Ưu điểm**: Đơn giản, dễ sử dụng, ít code.  
**Nhược điểm**: Hạn chế trong việc tùy chỉnh các bước chi tiết.

### 2. Lắp Ráp Thủ Công Từ Các Thành Phần (custom_pipeline_example.py)

Cách này tự lắp ráp pipeline từ các thành phần riêng biệt, cho phép kiểm soát chi tiết:

```python
# Tải từng thành phần riêng lẻ
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
scheduler = LMSDiscreteScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

# Chuyển các mô hình đến thiết bị
vae = vae.to(device)
text_encoder = text_encoder.to(device)
unet = unet.to(device)
```

Quá trình tạo ảnh được triển khai thủ công:

```python
# 1. Tạo text embeddings
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, 
                      truncation=True, return_tensors="pt")
with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

# Tạo unconditional embeddings cho classifier-free guidance
uncond_input = tokenizer([""] * batch_size, padding="max_length", 
                         max_length=max_length, return_tensors="pt")
with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

# Ghép text embeddings
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

# 2. Tạo latents ngẫu nhiên ban đầu
latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
    device=device
)
latents = latents * scheduler.init_noise_sigma

# 3. Quá trình khử nhiễu
scheduler.set_timesteps(num_inference_steps)
for t in scheduler.timesteps:
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

# 4. Giải mã latents thành ảnh
latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample

# Chuyển đổi sang ảnh PIL
image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
```

**Ưu điểm**:
- Kiểm soát từng bước của quá trình khử nhiễu
- Truy cập trực tiếp vào các giá trị trung gian (latents, noise predictions)
- Tùy chỉnh cách áp dụng classifier-free guidance
- Có thể theo dõi và gỡ lỗi từng bước

**Nhược điểm**:
- Phức tạp hơn, yêu cầu hiểu biết sâu về mô hình
- Cần code dài hơn và chi tiết hơn

## Các Thành Phần Chính

### 1. VAE (Variational Autoencoder)
- **Chức năng**: Mã hóa hình ảnh thành không gian tiềm ẩn (latent space) và giải mã ngược lại từ biểu diễn tiềm ẩn thành hình ảnh pixel.
- **Triển khai**: Sử dụng mô hình `AutoencoderKL` từ RunwayML Stable Diffusion v1.5.
- **Vai trò**: Chuyển đổi biểu diễn tiềm ẩn cuối cùng thành hình ảnh pixel trong bước cuối của quá trình tạo ảnh.

### 2. Mô hình UNet
- **Chức năng**: Mạng nơ-ron cốt lõi cho quá trình khử nhiễu.
- **Triển khai**: `UNet2DConditionModel` với khả năng xử lý đầu vào có điều kiện.
- **Vai trò**: Dự đoán phần dư của nhiễu (noise residual) tại mỗi bước khử nhiễu, được hướng dẫn bởi biểu diễn văn bản.

### 3. CLIP Text Encoder
- **Chức năng**: Chuyển đổi mô tả văn bản thành biểu diễn vector (embeddings) để hướng dẫn quá trình tạo hình ảnh.
- **Triển khai**: Sử dụng mô hình CLIP (Contrastive Language-Image Pre-training) của OpenAI.
- **Vai trò**: Xử lý cả prompt có điều kiện và prompt trống không điều kiện để thực hiện classifier-free guidance.

### 4. LMS Scheduler
- **Chức năng**: Điều khiển lịch trình nhiễu trong quá trình khử nhiễu.
- **Triển khai**: Linear Multistep Scheduler.
- **Vai trò**: Quyết định tốc độ loại bỏ nhiễu qua các bước.

## Quy Trình Tạo Ảnh

1. **Khởi tạo Pipeline**:
   - Tải các thành phần riêng biệt: VAE, CLIP text encoder, UNet, và LMS scheduler.
   - Chuyển tất cả các mô hình đến thiết bị phù hợp (CUDA hoặc CPU).

2. **Tạo Biểu Diễn Vector Văn Bản**:
   - Chuyển đổi prompt văn bản thành tokens bằng CLIP tokenizer.
   - Mã hóa tokens thành vector biểu diễn văn bản.
   - Tạo biểu diễn không điều kiện từ prompt trống cho classifier-free guidance.

3. **Khởi tạo Latents**:
   - Tạo nhiễu ngẫu nhiên trong không gian tiềm ẩn làm điểm khởi đầu.
   - Kích thước latents là [batch_size, 4, height/8, width/8].

4. **Quá Trình Khử Nhiễu**:
   - Thiết lập lịch trình với số bước suy luận.
   - Lặp qua các bước thời gian từ nhiễu cao đến thấp.
   - Tại mỗi bước:
     - Nhân đôi latents cho classifier-free guidance.
     - Dự đoán nhiễu dư bằng UNet.
     - Áp dụng classifier-free guidance để kết hợp dự đoán có điều kiện và không điều kiện.
     - Thực hiện một bước khử nhiễu với LMS scheduler.

5. **Giải Mã Latent thành Hình Ảnh**:
   - Chuẩn hóa latents cuối cùng.
   - Giải mã thành hình ảnh pixel bằng VAE decoder.
   - Chuyển đổi tensor thành hình ảnh PIL.

6. **Lưu Kết Quả**:
   - Lưu hình ảnh với tên file bao gồm thông tin prompt, guidance scale và số bước.
   - Tạo file log chi tiết với tất cả các tham số được sử dụng.

## Ưu Điểm của Pipeline Tùy Chỉnh

1. **Kiểm Soát Chi Tiết**:
   - Cho phép điều chỉnh từng bước trong quá trình khử nhiễu.
   - Có thể thay đổi hoặc tùy chỉnh các thành phần riêng lẻ.

2. **Hiểu Biết Sâu Sắc**:
   - Hiểu rõ từng bước trong quá trình tạo ảnh.
   - Dễ dàng theo dõi và gỡ lỗi.

3. **Khả Năng Mở Rộng**:
   - Có thể tích hợp các kỹ thuật tối ưu hoặc thành phần mới.
   - Linh hoạt để thực hiện các phương pháp tạo ảnh nâng cao.

## Cách Sử Dụng

Chạy pipeline với lệnh sau:

```bash
python custom_pipeline_example.py --prompt "Mô tả hình ảnh bạn muốn tạo" --guidance 7.5 --steps 50
```

Các tham số:
- `--prompt`: Mô tả văn bản cho hình ảnh bạn muốn tạo
- `--guidance`: Mức độ ảnh hưởng của prompt (thường là 7.0-8.0)
- `--steps`: Số bước khử nhiễu (nhiều hơn cho chất lượng cao hơn)
- `--height`, `--width`: Kích thước hình ảnh (bội số của 8)
- `--seed`: Giá trị seed để tái tạo hình ảnh (tùy chọn)
- `--output`: Đường dẫn file đầu ra 