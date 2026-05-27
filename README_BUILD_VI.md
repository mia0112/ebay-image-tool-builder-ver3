# eBay Image Tool - Build Standalone

## Flow moi

```text
Image input
→ Gemini generateContent phân tích ảnh và trả JSON
→ OpenCV xoá chữ nền / watermark theo JSON
→ OpenCV tạo mask sản phẩm, giữ phụ kiện nhỏ
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
