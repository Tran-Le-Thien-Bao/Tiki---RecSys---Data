# recsys: phần Hệ khuyến nghị (DS300)

Đo độ nhạy của các mô hình gợi ý trước review rác và độ lệch 5 sao trên dữ liệu Tiki,
và so sánh các biện pháp giảm nhẹ. Đặc tả đầy đủ: `docs/DAC_TA.md`. Phân công: `docs/PHAN_CONG.md`.

## Cấu trúc

| Thư mục / file | Nội dung | Đặc tả | Trạng thái |
|---|---|---|---|
| `config.py` | mọi tham số: đường dẫn, mốc cắt, K, seed, ngưỡng gắn cờ, λ, ε | mục 3 | xong (λ, từ khóa promo, burst còn TẠM) |
| `schema.py` | quy ước cột/kiểu của bảng tương tác và kết quả; `RunConfig` | mục 5 | xong |
| `prep/` | lọc trùng, nhị phân hóa, chia theo thời gian, k-core trên train, warm/cold | mục 5.1 | xong |
| `models/` | `base.py` + Popular, ItemKNN, ALS, BPR, MF-BCE, LightFM | mục 5.2 | mới có `base.py` |
| `evaluation/` | `results.py` + metric, bootstrap | mục 5.3 | mới có `results.py` |
| `mitigation/` | gắn cờ, trọng số tin cậy, T-CE/R-CE | mục 5.4–5.6 | chưa |
| `attacks/` | sinh hồ sơ giả: random, average, bandwagon, observed | mục 5.7 | chưa |
| `experiments/` | `pilot.py`, `stage1.py`, `stage2.py` | mục 6 | chưa |
| `notebooks/` | khám phá, vẽ hình (không chứa logic chính) | | |

Dữ liệu đầu vào: `data/raw/` (không commit). Kết quả: `results/<giai đoạn>/`.

## Cài đặt và chạy (Windows, PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-recsys.txt
pytest -q
python -m recsys.prep.build_interactions --star all pos45 --k 0 2 3 5
```
