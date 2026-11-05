# Quy trình làm việc nhóm

Đọc `docs/DAC_TA.md` (đặc tả, quy tắc bắt buộc) trước. Bảng phân công: `docs/PHAN_CONG.md`.

## Cài đặt

```powershell
git clone <repo>; cd tiki-recsys
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-recsys.txt
pytest -q
```

Dữ liệu **không có trên GitHub**. Xin file `reviews_anonymized.jsonl`, `products.jsonl` qua kênh riêng của nhóm, đặt vào `data/raw/`.
Không chia sẻ dữ liệu ra ngoài nhóm.

## Nhánh và PR

- `main` luôn chạy được `pytest -q`. Không push thẳng lên `main`.
- Mỗi issue một nhánh: `feat/<mã-issue>-<tên-ngắn>`, vd `feat/A1-build-interactions`.
- PR nhỏ, một việc. Ghi `Closes #<số>` trong mô tả. Cần ít nhất 1 người review.
- Sửa `recsys/config.py`, `recsys/schema.py`, `recsys/models/base.py` (quy ước chung) thì mở PR riêng và báo cả nhóm.

## Bắt buộc trước khi mở PR

1. `pytest -q` pass. Code mới có test trong `tests/recsys/` dùng dữ liệu giả, **không đọc `data/raw/`**.
2. Không có `user_id` kèm nội dung review, không có nội dung review của từng người trong code, log, notebook output, mô tả PR.
   Notebook: **Clear All Outputs** trước khi commit.
3. Không commit gì trong `data/`, `.env`, khóa, file `per_user.parquet`.
4. Mọi con số trong PR/README dán từ output chạy thật.
5. Mọi hàm ngẫu nhiên nhận `seed` tường minh. Tham số chọn trên validation, không chọn trên test.
6. Ngưỡng gắn cờ và λ chỉ sửa trước khi bắt đầu giai đoạn 2 (issue D2), không sửa theo metric gợi ý.

## Quy ước code

Python 3.11, type hint, docstring tiếng Việt ngắn, `pathlib`, `logging` (không `print` dữ liệu).
Mô hình mới: kế thừa `recsys.models.base.Recommender`. Kết quả: ghi bằng `recsys.evaluation.results.save_run`.
