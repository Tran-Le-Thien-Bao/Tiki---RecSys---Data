# Hệ khuyến nghị sản phẩm Tiki

Đồ án chung cho IE212 (Big Data), DS300 (Hệ khuyến nghị), IE104 (Web).
Đặc tả và quy tắc bắt buộc: `docs/DAC_TA.md`. Phân công: `docs/PHAN_CONG.md`. Quy trình nhóm: `CONTRIBUTING.md`.

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # Linux/macOS: cp .env.example .env
```

Sửa `.env`:

- `CRAWLER_CONTACT`: email sinh viên thật (ghi vào User-Agent).
- `HMAC_KEY_PATH`: đường dẫn khóa băm, **ngoài repo** (vd `~/.secrets/tiki_hmac.key`). Lần chạy đầu sẽ tự tạo.
- `DATA_DIR`: mặc định `./data`.

**Giữ kỹ file khóa**: mất khóa thì mã người dùng ở lần cào sau sẽ không khớp với lần trước.

## Chạy

```bash
pytest -q                                   # test, không gọi mạng
python -m crawler.run_crawl --smoke         # kiểm tra API: 1 danh mục, 1 sản phẩm, 1 trang review
python -m crawler.run_crawl --categories https://tiki.vn/dien-thoai-may-tinh-bang/c1789 \
    --max-product-pages 3 --min-reviews 5 --limit-products 100
python -m analysis.report data/raw/reviews.jsonl
```

Dừng giữa chừng (Ctrl+C) rồi chạy lại cùng lệnh sẽ tiếp tục từ sản phẩm chưa xong.
Nếu log báo `BỊ CHẶN` (exit code 2): dừng lại, không thử vượt chặn.
