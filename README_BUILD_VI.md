# eBay Image Tool - Build Standalone

## Flow v3

```text
Image input
→ Gemini generateContent phân tích ảnh và trả JSON
→ OpenCV chỉ sửa alpha mask: xoá nền, viền listing, chữ nền/disclaimer
→ OpenCV giữ RGB ảnh gốc, giữ phụ kiện nhỏ
→ Auto-fit sản phẩm vào frame
→ Collision check tránh frame đè lên sản phẩm
→ Gemini QA final nếu có API key
→ Output final + debug files
```

## Chạy source local

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app_auto.py
```

## API keys

- `CONFIG/gemini_api_key.txt`: Gemini API key, dùng cho phân tích ảnh và QA.
- `CONFIG/api_key.txt`: remove.bg API key, chỉ dùng fallback.

Tool cũng đọc biến môi trường `GEMINI_API_KEY` hoặc `GOOGLE_API_KEY` nếu file Gemini key trống.

## Build bằng GitHub Actions

Push lên `main` hoặc `master`, Action sẽ build:

- `eBayImageTool_Windows_standalone.zip`
- `eBayImageTool_macOS_standalone.zip`

## File chính

```text
app_auto.py
core/config.py
core/gemini_client.py
core/gemini_prompts.py
core/mask_cleaner.py
core/frame_composer.py
core/image_pipeline.py
core/api_bg_remove.py
```


## Ghi chú v3

- Không crop cứng theo Gemini `product_bbox`; bbox chỉ dùng để ưu tiên giữ/xoá vùng mềm.
- Không inpaint/chỉnh RGB ảnh gốc trong output; OpenCV chỉ tạo và clean alpha mask.
- Resize khi add frame luôn giữ nguyên tỷ lệ sản phẩm.
- Source trim mặc định tắt (`source_trim_uniform_border=false`) để tránh cắt mất sản phẩm.
- Nếu không fit được tuyệt đối trong frame, tool sẽ đặt frame dưới sản phẩm ở bước safety để frame không đè lên sản phẩm.
