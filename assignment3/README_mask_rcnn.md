# Bài tập 3: Phân đoạn đối tượng với Mask R-CNN (MMDetection)

Tập lệnh này thực hiện các tác vụ phân đoạn đối tượng sử dụng Mask R-CNN với MMDetection, một thư viện phát hiện đối tượng mã nguồn mở.

## Yêu cầu cài đặt

```bash
# Cài đặt MMDetection và các thư viện phụ thuộc
pip install -U openmim
mim install mmcv-full
pip install mmdet
pip install pycocotools
pip install tensorboard

# Các thư viện khác
pip install numpy opencv-python matplotlib tqdm
```

## Tính năng chính

1. **Suy luận (Inference) với mô hình Mask R-CNN đã huấn luyện sẵn**
   - Sử dụng mô hình đã huấn luyện sẵn để phát hiện và phân đoạn đối tượng
   - Trực quan hóa kết quả với bounding box và mask

2. **Chuyển đổi bộ dữ liệu tùy chỉnh sang định dạng COCO**
   - Hỗ trợ chuyển đổi dữ liệu tùy chỉnh (ví dụ: Balloon dataset)
   - Tự động chia tập dữ liệu thành tập huấn luyện và tập xác thực (80/20)

3. **Fine-tune mô hình Mask R-CNN**
   - Tùy chỉnh cấu hình mô hình cho dữ liệu mới
   - Huấn luyện với ghi log và giám sát quá trình

4. **Đánh giá mô hình**
   - Tính toán các số liệu đánh giá (mean Average Precision)
   - Hiển thị kết quả đánh giá chi tiết

## Cách sử dụng

### 1. Suy luận (Inference) với mô hình đã huấn luyện sẵn

```bash
python mask_rcnn_segmentation.py --task inference \
    --config configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py \
    --checkpoint mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
    --input test_image.jpg \
    --output output_folder \
    --score_threshold 0.3
```

### 2. Chuyển đổi bộ dữ liệu tùy chỉnh sang định dạng COCO

```bash
python mask_rcnn_segmentation.py --task convert \
    --custom_dataset path/to/custom_dataset \
    --custom_dataset_output path/to/coco_format_dataset
```

Cấu trúc thư mục dữ liệu đầu vào:
```
custom_dataset/
  ├─ images/
  │   ├─ image1.jpg
  │   ├─ image2.jpg
  │   └─ ...
  ├─ masks/
  │   ├─ image1.png
  │   ├─ image2.png
  │   └─ ...
  └─ annotations.json (tùy chọn)
```

### 3. Fine-tune mô hình trên dữ liệu mới

```bash
python mask_rcnn_segmentation.py --task train \
    --config configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py \
    --custom_dataset_output path/to/coco_format_dataset \
    --output train_results \
    --epochs 12
```

### 4. Đánh giá mô hình

```bash
python mask_rcnn_segmentation.py --task evaluate \
    --custom_dataset_output path/to/coco_format_dataset \
    --checkpoint train_results/latest.pth \
    --output eval_results
```

## Trực quan hóa với TensorBoard

Trong quá trình huấn luyện, các số liệu được ghi vào log TensorBoard. Để xem:

```bash
tensorboard --logdir train_results
```

## Ví dụ với Balloon Dataset

### 1. Tải và chuẩn bị Balloon Dataset

```bash
# Tải dataset
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
unzip balloon_dataset.zip

# Tạo cấu trúc thư mục phù hợp
mkdir -p balloon_dataset/images
mkdir -p balloon_dataset/masks
cp balloon/train/* balloon_dataset/images/
cp balloon/val/* balloon_dataset/images/

# Tạo mask từ annotations
# (Cần viết script riêng để chuyển đổi từ VIA format sang masks)
```

### 2. Chuyển đổi sang định dạng COCO

```bash
python mask_rcnn_segmentation.py --task convert \
    --custom_dataset balloon_dataset \
    --custom_dataset_output balloon_coco
```

### 3. Fine-tune mô hình

```bash
python mask_rcnn_segmentation.py --task train \
    --config configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py \
    --custom_dataset_output balloon_coco \
    --output balloon_results \
    --epochs 12
```

### 4. Đánh giá kết quả

```bash
python mask_rcnn_segmentation.py --task evaluate \
    --custom_dataset_output balloon_coco \
    --checkpoint balloon_results/latest.pth \
    --output balloon_eval
```

## Lưu ý

- Để có kết quả tốt nhất, nên sử dụng GPU với ít nhất 8GB VRAM
- Việc fine-tune cần điều chỉnh các tham số như learning rate, batch size phù hợp với kích thước dữ liệu
- Khi sử dụng với dữ liệu mới, cần cập nhật tên lớp trong tệp cấu hình 