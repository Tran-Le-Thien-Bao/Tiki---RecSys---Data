# recsys: phần Hệ khuyến nghị (DS300)

Đo độ bền vững của các mô hình gợi ý trước review rác và độ lệch 5 sao trên dữ liệu Tiki,
và so sánh các biện pháp giảm nhẹ. Tình trạng, phân công và đặc tả: `docs/DO_AN.md`.

## Cấu trúc

| Thư mục / file | Nội dung | Đặc tả (`docs/DO_AN.md`) | Trạng thái |
|---|---|---|---|
| `config.py` | mọi tham số: đường dẫn, mốc cắt, K, seed, ngưỡng gắn cờ, λ, ε | mục 4.3 | xong (λ, từ khóa promo, burst còn tạm) |
| `schema.py` | quy ước bảng tương tác và kết quả; `RunConfig` | mục 4.5 | xong |
| `prep/` | lọc trùng, nhị phân hóa, chia theo thời gian, k-core trên train, warm/cold | mục 4.5.1 | xong |
| `models/` | `base.py`, Popular, ItemKNN, ALS (BPR, MF-BCE, LightFM sau) | mục 4.5.2 | 3/6 |
| `evaluation/` | metric, bootstrap, lưu kết quả | mục 4.5.3 | xong |
| `mitigation/` | `flags.py` (gắn cờ), `label_sample.py` (gán nhãn tay → λ); trust, T-CE/R-CE sau | mục 4.5.4–4.5.6 | gắn cờ xong |
| `attacks/` | sinh hồ sơ giả | mục 4.5.7 | chưa |
| `experiments/` | `pilot.py` (xong); `stage1.py`, `stage2.py` sau | mục 4.6 | pilot xong |
| `notebooks/` | `figures.py`: hình cho báo cáo | | xong |

## Lệnh

```powershell
python -m recsys.prep.build_interactions --star all pos45 --k 0 2 3 5
python -m recsys.experiments.pilot --k 0 2
python -m recsys.notebooks.figures
python -m recsys.mitigation.label_sample lambda
pytest -q
```
