# Đặc tả đồ án: Độ nhạy của hệ gợi ý trước review rác và độ lệch đánh giá (Tiki)

Đây là tài liệu đặc tả chính của phần `recsys/`. Phân công: `docs/PHAN_CONG.md`. Quy trình nhóm: `CONTRIBUTING.md`.

## 1. Đề tài

Đây là nghiên cứu **đo lường**. Đo xem tương tác giả (review rác, tài khoản thuê) và độ lệch 5 sao bẻ cong các mô hình gợi ý
phổ biến đến đâu trên dữ liệu Tiki, và 3 biện pháp có sẵn giảm nhẹ được bao nhiêu.
Đồ án không đề xuất thuật toán mới. **Kết quả âm** (không có hiệu ứng, biện pháp không giúp) là kết luận hợp lệ và phải báo cáo đúng.

Một sản phẩm dùng chung cho 3 môn: IE212 (dữ liệu lớn, PySpark, `bigdata/`), DS300 (hệ khuyến nghị, `recsys/`), IE104 (web, `web/`).
**Giai đoạn hiện tại làm `recsys/` trước.** Mỗi cấu hình thí nghiệm là một hàm thuần `run_config(cfg) -> kết quả`
(`cfg` là `recsys.schema.RunConfig`), để sau này PySpark gọi phân tán được.

### Câu hỏi nghiên cứu
- **RQ1 (phụ):** tấn công có làm giảm chất lượng gợi ý chung (NDCG@K trên user warm) không, và có làm đổi thứ hạng giữa các mô hình (Kendall tau) không. Dự kiến hiệu ứng nhỏ; Kendall tau = 1 là kết quả hợp lệ.
- **RQ2 (lõi, thước đo chính):** tấn công đẩy được sản phẩm mục tiêu vào top-K đến mức nào. Đo bằng hit ratio@K và prediction shift.
- **RQ3 (chỉ giữ nếu pilot đạt):** bỏ các tương tác bị nghi ngờ có thật sự làm top-K và metric đổi đáng kể không.
- **RQ4:** lọc cứng, trọng số tin cậy và T-CE/R-CE phục hồi được bao nhiêu. So sánh chính trên cùng mô hình nền MF-BCE.

## 2. Dữ liệu

`data/raw/` không có trên GitHub và **không được sửa**. Lấy qua kênh riêng của nhóm.
- `reviews_anonymized.jsonl`: `review_id, user_id, product_id, rating, title, content, thank_count, created_at (unix giây), is_verified_purchase`.
  `user_id` đã được thay mã (giả danh hóa, **không phải** ẩn danh).
- `products.jsonl`: `product_id, sku, name, category, brand_name, price, original_price, discount_rate, rating_average, review_count, all_time_quantity_sold, short_description, specifications, thumbnail_url, crawled_at`.

Số liệu đã kiểm. Prep tự đối chiếu với `config.EXPECTED*` và báo lỗi nếu lệch:
- reviews: 408.160 dòng → **353.035** `review_id` duy nhất; 227.557 user; 347.901 cặp (user, product); 7.360 sản phẩm có review.
- products: 15.338 dòng → 15.336 `product_id` duy nhất; 9 danh mục.
- rating: 1★ 5.016 · 2★ 2.376 · 3★ 5.366 · 4★ 29.656 · 5★ 310.621.
- 60,5% review không có nội dung; 99,9% có `is_verified_purchase = true`.
- k-core trên toàn bộ cặp (user / sp / cặp): 2-core 54.656 / 4.500 / 173.263; 3-core 21.943 / 2.931 / 105.637; 5-core 6.235 / 1.537 / 50.140.
- Nội dung (dài > 20 ký tự sau strip) lặp ≥ 5 lần: 202 nội dung, 1.823 review.

## 3. Quy tắc bắt buộc

1. **Quyền riêng tư:** không in ra màn hình, log, notebook output hay file kết quả (a) `user_id` kèm nội dung review, (b) nội dung review của từng người.
   Kết quả chỉ ở dạng tổng hợp. Số điện thoại được che thành `[sdt]` và email thành `[email]` ngay ở bước prep.
2. **Không công bố dữ liệu thô** và không đưa `data/` lên Git.
3. **Không cào thêm dữ liệu** trong giai đoạn này. Giữ nguyên `crawler/`.
4. **Không bịa số.** Mọi con số trong báo cáo, README hay PR phải đến từ lần chạy thật, kèm output.
5. **Chống đánh giá vòng tròn:**
   - Không bao giờ lọc tập test bằng bộ gắn cờ của nhóm.
   - Ngưỡng và λ của bộ gắn cờ phải được cố định trong `recsys/config.py` **trước khi** sinh dữ liệu tấn công.
   - Hồ sơ giả chỉ nằm trong train. Metric chỉ tính trên user thật.
6. **Không chọn tham số theo test.** Mọi lựa chọn (α, số chiều, k-core, biến thể sao) đều dựa trên validation.
7. Mọi hàm có tính ngẫu nhiên nhận `seed` tường minh. Chạy lại cùng seed phải ra cùng kết quả.

## 4. Cấu trúc thư mục

```
recsys/
├── config.py          # đường dẫn, mốc cắt, K, seed, ngưỡng gắn cờ, λ, ε (nơi duy nhất)
├── schema.py          # quy ước cột/kiểu dữ liệu, RunConfig
├── prep/              # build_interactions.py, split.py, kcore.py
├── models/            # base.py + popular, itemknn, als, bpr, mf_bce, lightfm_model
├── attacks/           # inject.py
├── mitigation/        # flags.py, trust.py, denoise.py (T-CE, R-CE)
├── evaluation/        # metrics.py, bootstrap.py, results.py
├── experiments/       # pilot.py, stage1.py, stage2.py
└── notebooks/         # chỉ khám phá + vẽ hình
tests/recsys/          # test dùng dữ liệu giả nhỏ, không đọc data/raw
results/{pilot,stage1,stage2}/
```

## 5. Đặc tả từng phần

### 5.1 Prep (`recsys/prep/`). Đã xong.
1. Đọc reviews và lọc trùng theo `review_id`. Đọc products và lọc trùng theo `product_id`.
2. Scrub `title`/`content` (SĐT, email). Cột `had_phone`/`had_email` được tính **trước** khi scrub và lưu trong `reviews_clean.parquet` để dùng cho cờ `promo`.
3. **Tương tác nhị phân:** mỗi cặp (user, product) giữ **review sớm nhất** (`created_at` nhỏ nhất; nếu hòa thì lấy `review_id` nhỏ hơn).
   Cột `ts` và `rating` lấy từ review đó; `n_reviews` là số review của cặp.
4. **Biến thể sao:** `all` (mọi cặp là dương) và `pos45` (chỉ rating ≥ 4 là dương; áp cho cả train, val và test).
5. **Chia theo mốc thời gian toàn cục:** `t_val` và `t_test` là phân vị 80% và 90% của `ts` trên toàn bộ tương tác (biến thể `all`),
   dùng chung cho mọi biến thể. Quy ước: train < t_val ≤ val < t_test ≤ test.
6. **k-core chỉ áp trên train** (để không dùng thông tin tương lai), với k ∈ {0 (không lọc), 2, 3, 5}.
   Sau đó loại khỏi val/test các sản phẩm không có trong train.
7. **Warm/cold:** user ở val/test có ≥ 1 tương tác trong train (sau k-core) là warm, còn lại là cold.
8. Đầu ra: `data/processed/{interactions,reviews_clean,products}.parquet`, `data/splits/{star}_{k}core/{train,val,test}.parquet`, `results/prep_summary.json`.

Lệnh: `python -m recsys.prep.build_interactions --star all pos45 --k 0 2 3 5`

### 5.2 Mô hình (`recsys/models/`)
Mọi mô hình kế thừa `recsys.models.base.Recommender` và chỉ cài `_fit(R, train, item_features, seed)` và `_score(user_idx, item_idx)`.
`R` là ma trận user × item, ô (u,i) = w_ui (không có trọng số thì w_ui = 1).

| Mô hình | Cài đặt | `sample_weight` (w_ui) dùng thế nào |
|---|---|---|
| Popular | đếm tương tác trong train | tổng w thay cho số đếm |
| ItemKNN | cosine giữa các cột ma trận tương tác, giữ top-k láng giềng | dùng R có trọng số TRƯỚC khi tính cosine |
| ALS | thư viện `implicit` | confidence c_ui = 1 + α·w_ui |
| BPR | `implicit`, hoặc tự viết bằng torch nếu `implicit` không nhận trọng số | nhân w_ui vào loss/gradient của bộ ba (u,i,j) với i dương; giữ nguyên cách lấy mẫu |
| MF-BCE | torch, MF + binary cross-entropy, lấy mẫu âm (mặc định 4 âm/1 dương) | nhân w_ui vào số hạng BCE của tương tác dương; mẫu âm có trọng số 1 |
| LightFM | thư viện `lightfm`, đặc trưng sản phẩm: danh mục, thương hiệu, khoảng giá, từ khóa tên | không dùng w |

Popular và LightFM có `supports_cold = True`.

### 5.3 Đánh giá (`recsys/evaluation/`)
- Ground truth của một user là tập sản phẩm user đó tương tác trong val (hoặc test). Gợi ý loại trừ sản phẩm đã có trong train.
- Metric tính theo **từng user** rồi mới lấy trung bình: Recall@K, NDCG@K (K = 10, 20), coverage, Jaccard@10 giữa hai mô hình hoặc hai phiên bản.
- Với tấn công:
  - **hit ratio@K:** tỉ lệ user thật có ít nhất một sản phẩm mục tiêu trong top-K.
  - **prediction shift:** thay đổi điểm và thứ hạng trung bình của sản phẩm mục tiêu, so với cùng mô hình, cùng seed trên dữ liệu sạch.
- RQ1–RQ4 tính trên **user warm**. Popular và LightFM báo thêm kết quả trên user cold.
- **Bootstrap:** lấy mẫu lại 1.000 lần **vectơ metric theo user** (không huấn luyện lại), khoảng tin cậy 95% theo percentile.
  So hai phiên bản thì dùng bootstrap cặp trên cùng tập user. Khoảng chứa 0 thì ghi "không phân biệt được".
- Mỗi lần chạy lưu `results/<stage>/<run_id>/per_user.parquet` (user đã băm, không kèm nội dung) và `summary.json`, qua `evaluation.results.save_run`.

### 5.4 Gắn cờ (`recsys/mitigation/flags.py`): chỉ dùng 3 dấu hiệu
| Dấu hiệu | Định nghĩa (tham số cố định trong `config.py`) |
|---|---|
| `dup_content` | nội dung chuẩn hóa (lowercase, strip, gộp khoảng trắng) dài > 20 ký tự và xuất hiện ≥ 5 lần trên toàn bộ dữ liệu |
| `promo` | chứa từ khóa trong `PROMO_KEYWORDS` (vd "săn sale", "trả công", "inbox", "zalo") hoặc có SĐT trước khi scrub (`had_phone`) |
| `burst` | cùng user có ≥ 2 review cho cùng sản phẩm, hoặc review ≥ N sản phẩm trong cửa sổ T phút (N, T trong config) |

**Không dùng:** "chưa xác thực mua", "luôn 5 sao không nội dung", "lệch cảm xúc".

### 5.5 Trọng số tin cậy (`mitigation/trust.py`)
`w_ui = max(ε, 1 − min(1, Σ_k λ_k · s_k(u,i)))`, với ε = 0,05.
λ_k là precision của dấu hiệu k trên mẫu gán nhãn tay (`data/labels/`), được cố định trước khi chạy thí nghiệm.
Khi chưa có nhãn thì dùng giá trị tạm trong config và ghi rõ là tạm. **Không chỉnh λ theo metric gợi ý.**

### 5.6 Khử nhiễu (`mitigation/denoise.py`)
Cài T-CE (truncated loss) và R-CE (reweighted loss) theo Wang và cộng sự (WSDM 2021) trên MF-BCE.
Siêu tham số đối chiếu với mã gốc DenoisingRec (github.com/WenjieWWJ/DenoisingRec).

### 5.7 Tấn công (`recsys/attacks/inject.py`): đẩy sản phẩm mục tiêu
- **Kiểu tấn công:**
  - `random`: filler chọn đều.
  - `average`: filler theo phân phối độ phổ biến.
  - `bandwagon`: filler là các sản phẩm phổ biến nhất.
  - `observed`: bắt chước nhóm "săn sale" (cùng nội dung, đăng dồn dập), nên bộ gắn cờ nhìn thấy được.
- **Mức tấn công:** số hồ sơ giả bằng 1%, 3% hoặc 5% số user train.
  Mỗi hồ sơ gồm các sản phẩm mục tiêu cộng filler; số filler bằng trung vị số tương tác/user trong train.
- **Mục tiêu:** mỗi seed chọn 10 sản phẩm ít phổ biến và 10 sản phẩm phổ biến trung bình (lấy từ train).
- **Thời điểm** của tương tác giả nằm trong khoảng thời gian train.
- **Định danh:** hồ sơ giả có `user_id` âm (−1, −2, ...) và `is_fake = True`.
- Chạy 5 seed cho mỗi cấu hình.

## 6. Thí nghiệm

| Bước | Nội dung | Số lần huấn luyện |
|---|---|---|
| Pilot (`pilot.py`) | Popular, ItemKNN, ALS trên train gốc và trên train đã bỏ tương tác bị gắn cờ (`dup_content` ∪ `promo`); val/test giữ nguyên; báo Jaccard@10 và ΔNDCG@10 kèm CI. **Quy tắc định trước:** giữ RQ3 nếu CI của ΔNDCG không chứa 0 HOẶC Jaccard@10 trung bình < 0,9 | ~6–18 |
| Giai đoạn 1 (`stage1.py`) | 6 mô hình × k ∈ {0,2,3,5} × sao ∈ {all, pos45} × 3 seed, không tấn công; chọn 1 mức k và 1 biến thể sao theo cỡ nhóm warm và độ rộng CI trên validation | 144 |
| Giai đoạn 2a | 6 mô hình × (1 sạch + 4 kiểu × 3 mức) × 5 seed | 390 |
| Giai đoạn 2b | MF-BCE × {lọc cứng, trọng số, T-CE, R-CE} × 12 × 5 | 240 |
| Giai đoạn 2c | trọng số tin cậy trên ALS, BPR, ItemKNN × 12 × 5 | 180 |

Nếu thiếu thời gian, giảm số seed của giai đoạn 2 từ 5 xuống 3.

## 7. Quy ước code

- Python có type hint, docstring tiếng Việt ngắn, đường dẫn dùng `pathlib` (không hard-code dấu `\`).
- Không `print` dữ liệu cá nhân; dùng `logging`.
- Hàm thuần, không phụ thuộc biến toàn cục ngoài `config.py`.
- Test đặt ở `tests/recsys/`, chạy bằng `pytest -q`, không đọc `data/raw/`.
- Mô hình mới kế thừa `Recommender`. Bảng tương tác và kết quả theo đúng `recsys/schema.py`.
