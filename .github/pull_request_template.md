## Nội dung

Closes #

## Kiểm tra

- [ ] `pytest -q` pass (dán dòng cuối output):
- [ ] Test mới dùng dữ liệu giả, không đọc `data/raw/`
- [ ] Không có `user_id` kèm nội dung review / nội dung review cá nhân trong code, log, notebook output, mô tả PR
- [ ] Không commit `data/`, `.env`, khóa, `per_user.parquet`; notebook đã clear output
- [ ] Hàm ngẫu nhiên nhận `seed`; không chọn tham số theo test
- [ ] Có sửa quy ước chung (`config.py`, `schema.py`, `models/base.py`)? Nếu có: đã báo nhóm

## Output chạy thật (nếu có số liệu)

```
```
