eBay Image Tool - Gemini Guided OpenCV
======================================

Muc tieu:
- Gemini nhin anh truoc de xac dinh san pham, phu kien can giu, chu/watermark can xoa.
- OpenCV xu ly pixel that: xoa nen, clean mask, giu phu kien nho.
- Tool tu can san pham vao frame sao cho to hon nhung khong bi frame de.

Cach dung nhanh:
1. Mo CONFIG/gemini_api_key.txt
2. Dan Gemini API key vao file nay va luu lai.
3. Tuy chon: mo CONFIG/api_key.txt va dan remove.bg API key de lam fallback khi Gemini/OpenCV fail.
4. Bo frame PNG vao folder FRAME. Neu folder co nhieu anh, tool uu tien file PNG dau tien.
5. Bo folder san pham vao INPUT, vi du:
   INPUT/A12345__51360-SDA-A01/main.jpg
6. Chay:
   - Windows: START_WINDOWS.bat
   - Mac: START_MAC.command
7. Lay ket qua trong folder output cua tung san pham.

Ket qua debug:
Neu save_debug_files=true, moi anh se co debug trong:
output/debug/<ten_anh>/
- preview gui Gemini
- JSON Gemini analysis
- anh preclean
- mask PNG
- cutout PNG
- QA JSON
- final JPG
- fit JSON

Config quan trong:
- gemini_model: mac dinh gemini-3.1-flash-lite
- remove_background_flow: gemini_guided_opencv
- fallback_mode: removebg_api_then_opencv
- component_mode: auto
- keep_small_accessories: true
- auto_fit_to_frame: true
- frame_collision_check: true

Luu y:
- Khong dung Gemini de ve lai san pham. Gemini chi phan tich va tra JSON.
- Neu anh co nhieu oc, long den, lo xo, phe cai, tool se uu tien giu chung thay vi xoa nhu rac.
- Neu san pham bi nho trong frame, tang nhe max_product_width_ratio / max_product_height_ratio.
- Neu san pham bi frame de, giam max_product_width_ratio / max_product_height_ratio hoac tang frame_clearance_px.
