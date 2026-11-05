# Phân công phần `recsys/` (DS300)

Mỗi mục `### [X#]` dưới đây = 1 GitHub issue. Cột **Chờ** = issue phải xong trước.
Đặc tả chi tiết: `docs/DAC_TA.md`.

## Đã xong (ngày 0, quy ước chung)

Các file sau là quy ước chung, các luồng dựa vào đó để làm song song. **Muốn sửa phải mở PR riêng và báo cả nhóm.**

| File | Nội dung |
|---|---|
| `recsys/config.py` | đường dẫn, mốc cắt, K, seed, ngưỡng gắn cờ, λ (tạm), ε, tham số tấn công, số liệu đối chiếu |
| `recsys/schema.py` | cột/kiểu của bảng tương tác và `per_user.parquet`; `RunConfig` (mỗi cấu hình thí nghiệm = 1 đối tượng) |
| `recsys/models/base.py` | lớp `Recommender`: mô hình con chỉ cài `_fit(R, ...)` và `_score(user_idx, item_idx)` |
| `recsys/evaluation/results.py` | `hash_user`, `save_run` (ghi `per_user.parquet` + `summary.json`, chặn cột định danh) |
| `tests/recsys/conftest.py` | fixture `toy_interactions` dùng chung |

## Giai đoạn A: chạy song song ngay

| Luồng | Issue | Chờ | Đầu ra |
|---|---|---|---|
| A. Prep | A1, A2 | — | `data/processed/interactions.parquet`, `data/splits/*`, `results/prep_summary.json` |
| B. Đánh giá | B1, B2 | — | `evaluation/metrics.py`, `evaluation/bootstrap.py` |
| C. Mô hình nền | C1, C2, C3 | — | `models/popular.py`, `itemknn.py`, `als.py` |
| D. Gắn cờ | D1, D2 | A1 (D2) | `mitigation/flags.py`, `data/labels/`, λ chốt trong config |

> **A1 + A2 đã xong** (2026-11-05): chạy `python -m recsys.prep.build_interactions --star all pos45 --k 0 2 3 5`
> để tạo mọi bản chia (khoảng 20 giây). Số liệu ở `results/prep_summary.json`. Phần còn lại của luồng A thuộc về N1: P1, S4.

### [A1] Prep: đọc, lọc trùng, scrub, nhị phân hóa
- `prep/build_interactions.py`: lọc trùng `review_id`/`product_id`; scrub title/content (dùng `crawler.privacy.scrub`); mỗi cặp giữ review sớm nhất → `ts`, `rating`, `n_reviews`; `is_fake=False`.
- Đối chiếu `config.EXPECTED*`, lệch thì báo lỗi rõ ràng (không tự sửa số).
- Ghi `interactions.parquet` (gọi `validate_interactions`) và `reviews_clean.parquet` (cho flags; **không commit**).
- Test trên dữ liệu giả: chọn review sớm nhất, đếm n_reviews, scrub.

### [A2] Prep: chia thời gian, k-core trên train, warm/cold
- `prep/split.py`: `t_val`, `t_test` = phân vị `VAL_QUANTILE`/`TEST_QUANTILE` của `ts`; train < t_val ≤ val < t_test ≤ test.
- `prep/kcore.py`: k-core lặp đến hội tụ, **chỉ trên train**; loại khỏi val/test sp không có trong train.
- Biến thể `all` và `pos45`; CLI `--star all --k 0 2 3 5`.
- `prep_summary.json`: số user, tương tác, tỉ lệ warm/cold của val và test, cho mỗi (star, k).
- Test: k-core trên đồ thị nhỏ tính tay; không rò rỉ thời gian (max ts train < min ts val).

### [B1] Metric
- `evaluation/metrics.py`: Recall@K, NDCG@K theo **từng user**; coverage; Jaccard@K giữa hai dict gợi ý; hit ratio@K và prediction shift (Δ điểm, Δ thứ hạng) cho sp mục tiêu; Kendall tau giữa thứ hạng mô hình.
- Trả `DataFrame` theo `schema.PER_USER_REQUIRED` + cột `<metric>@<K>`.
- Test: ví dụ tính tay (vd 1 user, 3 gợi ý, 2 đúng → NDCG cụ thể).

### [B2] Bootstrap
- `evaluation/bootstrap.py`: 1.000 lần lấy mẫu lại vectơ metric theo user, CI 95% percentile; bootstrap **cặp** cho hiệu hai phiên bản trên cùng tập user; trả cờ `distinguishable = not (lo <= 0 <= hi)`.
- Nhận `seed` tường minh. Test: hằng số → CI độ rộng 0; cùng seed → cùng kết quả.

### [C1] Popular · [C2] ItemKNN · [C3] ALS
- Kế thừa `Recommender`; dùng `self.R` (đã chứa w_ui) theo bảng ở `docs/DAC_TA.md` mục 5.2.
- Popular: `supports_cold = True`. ItemKNN: cosine trên cột R, giữ top-k láng giềng. ALS: `implicit`, confidence `1 + α·R`.
- Test nhỏ trên `toy_interactions`: chạy được, cùng seed ra cùng kết quả, `sample_weight` có tác dụng.

### [D1] Gắn cờ
- `mitigation/flags.py`: 3 dấu hiệu `dup_content`, `promo`, `burst` theo `config` (`docs/DAC_TA.md` mục 5.4), ra cờ theo **review** rồi gộp theo cặp.
- `promo` cần SĐT **trước** khi scrub: dùng cột `had_phone` có sẵn trong `data/processed/reviews_clean.parquet` (prep đã tính trước khi scrub).
- Chỉ in số đếm tổng hợp. Test trên dữ liệu giả.

### [D2] Gán nhãn tay → λ
- Lấy mẫu (seed cố định) ~100 review/dấu hiệu bị gắn cờ, gán nhãn "rác/không" **trên máy**, lưu `data/labels/` (không commit).
- λ_k = precision; cập nhật `TRUST_LAMBDA`, đặt `TRUST_LAMBDA_IS_PROVISIONAL = False`, chốt `PROMO_KEYWORDS`, `BURST_*`. **Phải xong trước E1.**
- Không dán nội dung review lên issue/chat.

## Mốc kiểm tra: Pilot

### [P1] `experiments/pilot.py`
Chờ: A2, B1, B2, C1–C3, D1. Popular/ItemKNN/ALS trên train gốc vs train bỏ `dup_content ∪ promo`; báo Jaccard@10, ΔNDCG@10 + CI; quyết định RQ3 theo quy tắc định trước. **Họp nhóm sau mốc này.**

## Giai đoạn 1

| Issue | Việc | Chờ |
|---|---|---|
| S1 | `models/bpr.py` (implicit hoặc torch nếu implicit không nhận trọng số) | C3 |
| S2 | `models/mf_bce.py` (torch, BCE, 4 âm/1 dương, w nhân vào số hạng dương) | — |
| S3 | `models/lightfm_model.py` (danh mục, thương hiệu, khoảng giá, từ khóa tên). **Thử cài lightfm sớm**, lỗi thì báo nhóm, không tự đổi thư viện | — |
| S4 | `experiments/stage1.py`: 144 lần chạy, chọn 1 k + 1 biến thể sao theo validation | P1, S1–S3 |

## Giai đoạn 2

| Issue | Việc | Chờ |
|---|---|---|
| E1 | `attacks/inject.py`: random/average/bandwagon/observed, mức 1/3/5%, 10+10 mục tiêu/seed | D2, S4 |
| E2 | `mitigation/trust.py`: `w = max(ε, 1 − min(1, Σ λ_k s_k))` | D2 |
| E3 | `mitigation/denoise.py`: T-CE, R-CE trên MF-BCE (đối chiếu DenoisingRec) | S2 |
| E4 | `experiments/stage2.py`: 2a (390), 2b (240), 2c (180) | E1–E3 |

## Phân công 4 người

| Người | Vai trò | Trước pilot | Giai đoạn 1 | Giai đoạn 2 |
|---|---|---|---|---|
| **N1** | Dữ liệu + chạy thí nghiệm (đường găng, ghép nối) | A1, A2 | P1, S4 | E4 |
| **N2** | Đánh giá + tấn công | B1, B2 | S3 (LightFM) | E1 |
| **N3** | Mô hình cổ điển + trọng số | C1, C2, C3 | S1 (BPR) | E2 + chạy 2c |
| **N4** | Gắn cờ + khử nhiễu | D1, D2 | S2 (MF-BCE) | E3 + chạy 2b |

- **N1 giữ đường găng:** A1/A2 xong sớm nhất có thể; trong lúc chờ, các người khác test trên `toy_interactions`.
  N1 cũng là người review PR sửa quy ước chung (`config.py`, `schema.py`, `models/base.py`).
- **D2 (gán nhãn):** N4 chủ trì; N1 gán nhãn chéo ~30 mẫu/dấu hiệu để đo độ đồng thuận (Cohen's kappa) → λ đáng tin hơn.
  Hạn chót: trước khi N2 bắt đầu E1.
- **Mốc họp:** (1) xong A2 → cả nhóm chạy thử trên dữ liệu thật; (2) sau P1 → quyết RQ3; (3) sau S4 → chốt k + biến thể sao;
  (4) sau D2 → khóa `config.py` phần gắn cờ/λ (từ đây không sửa).
- N2 đổi bớt sang N3 nếu B1/B2 xong sớm: Kendall tau và vẽ hình cho báo cáo.
