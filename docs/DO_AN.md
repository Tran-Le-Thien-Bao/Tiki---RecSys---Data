# Đồ án: Đánh giá độ bền vững của hệ khuyến nghị trước review rác và độ lệch đánh giá trên dữ liệu Tiki

**Môn:** DS300 · **GVHD:** ThS. Huỳnh Văn Tín · **Nhóm 12:** Trần Lê Thiên Bảo (24520185), Huỳnh An Bình (24520193), Trần Nguyễn Nhật Bảo (24520186), Nguyễn Trần Phú Cường (24520241), Nguyễn Hồ Phúc Bảo (24520169)


---

## 1. Tình trạng

| Phần | Trạng thái | Ở đâu |
|---|---|---|
| Làm sạch, chia dữ liệu (8 biến thể) | Xong | `recsys/prep/`, `results/prep_summary.json` |
| Metric, bootstrap | Xong | `recsys/evaluation/` |
| Popular, ItemKNN, ALS | Xong | `recsys/models/` |
| Bộ gắn cờ 3 dấu hiệu | Xong (từ khóa, ngưỡng, λ còn **tạm**) | `recsys/mitigation/flags.py` |
| Pilot → quyết định **giữ RQ3** | Xong | `recsys/experiments/pilot.py`, `results/pilot/` |
| Hình cho báo cáo (h1–h5) | Xong | `results/figures/` |
| Gán nhãn tay → λ | **Đang làm** (An Bình, Phúc Bảo) | `data/labels/` (chỉ trên máy) |
| Báo cáo giữa kỳ | **Đang viết** | mục 2 bên dưới |
| BPR, MF-BCE, LightFM, tấn công, giai đoạn 1–2 | Chưa | mục 5 |

Code và kết quả nằm trên nhánh `feat/pilot`. `pytest -q` phải ra **68 passed**.

### Chuẩn bị máy (ai cần chạy code)
```powershell
git clone <repo>; cd <repo>; git checkout feat/pilot
py -3.9 -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-recsys.txt matplotlib
# lấy data/raw/*.jsonl từ Thiên Bảo qua kênh riêng (không qua GitHub)
python -m recsys.prep.build_interactions --star all pos45 --k 0 2 3 5   # ~20 giây
pytest -q
```

---

## 2. Phân công báo cáo giữa kỳ

**Thiên Bảo không viết, chỉ duyệt cuối và nộp.** Mỗi người viết đúng phần của mình, mở PR vào nhánh `feat/pilot`.

| Chương / mục | Người viết |
|---|---|
| **1. Giới thiệu** (bối cảnh, mục tiêu, 4 câu hỏi nghiên cứu, đóng góp, bố cục) | **Nhật Bảo** |
| **2.1–2.3** Hệ gợi ý và phản hồi ngầm · 6 mô hình · thước đo và bootstrap | **Phú Cường** |
| **2.4–2.6** Tấn công shilling · rò rỉ dữ liệu và khử nhiễu · nghiên cứu liên quan (Shrestha 2021) | **Nhật Bảo** |
| **3.1–3.3** Nguồn và định dạng dữ liệu · làm sạch và đặc điểm · chia dữ liệu | **An Bình** |
| **3.4–3.7** Pipeline · bộ gắn cờ và trọng số tin cậy · thiết kế tấn công và biện pháp · nguyên tắc chống đánh giá vòng tròn | **Phú Cường** |
| **4. Kết quả thực nghiệm** (pilot) | **Phúc Bảo** |
| **5. Kết luận** (đạt được, hạn chế, kế hoạch) | **Phúc Bảo** |
| **Tài liệu tham khảo** (gom và thống nhất kiểu trích dẫn) | **Nhật Bảo** |
| Gán nhãn 300 review | **An Bình** |
| Gán nhãn chéo 90 review | **Phúc Bảo** |
| Duyệt cuối, nộp | **Thiên Bảo** |

### Quy tắc chung
1. **Không tự sửa hay bịa số.** Số đã có sẵn ở mục 3 kèm nguồn. Cần số mới thì chạy lệnh và dán output vào PR.
2. **Không đưa nội dung review của từng người** vào báo cáo, slide, chat hay PR.
3. Mọi ý lấy từ bài báo phải có trích dẫn (danh sách ở mục 6).
4. Không có câu "đề xuất mô hình mới": đây là nghiên cứu **đo lường**.

### Nhật Bảo — Chương 1, mục 2.4–2.6, tài liệu tham khảo (~4 trang)
- **Chương 1:** bối cảnh (88,0% đánh giá là 5★: 310.621 / 353.035), mục tiêu đo lường, 4 RQ kèm thước đo (mục 4.1), đóng góp (thiết kế đo lường chặt trên dữ liệu TMĐT Việt Nam), bố cục 5 chương.
- **2.4 Tấn công shilling:** hồ sơ giả = sản phẩm mục tiêu + filler; random, average (Lam & Riedl 2004), bandwagon; hit ratio và prediction shift (Burke và cs. 2015).
- **2.5 Rò rỉ dữ liệu và khử nhiễu:** Ji và cs. 2023 (chia ngẫu nhiên cho mô hình học từ tương lai → chia theo mốc thời gian toàn cục); Gusak & Frolov 2026 (k-core và chia thời gian); Wang và cs. 2021 (mẫu nhiễu có loss lớn → T-CE, R-CE).
- **2.6 Nghiên cứu liên quan:** Shrestha, Spezzano, Pera 2021 — Yelp/Amazon, 5 mô hình, giữ hoặc bỏ spammer thật; prediction shift 0,087–0,160 so với 0,5–1,5 khi mô phỏng; user non-mainstream chịu ảnh hưởng nhiều hơn. Kết bằng bảng so sánh (có sẵn ở mục 3.4).
- **Tài liệu có sẵn:** mục 4.1, 6 của file này; slide 23–25 trong `docs/DS300_SEMINAR_NHÓM 12_BCTĐ.pptx`; PDF bài Shrestha (xin Thiên Bảo).
- **Xong khi:** viết thành văn, mỗi ý có trích dẫn, có bảng so sánh, danh mục tài liệu đủ.

### Phú Cường — mục 2.1–2.3 và 3.4–3.7 (~4–5 trang) + duyệt code
- **2.1** Phản hồi ngầm (r = 1 nếu đã review), ma trận thưa, lọc cộng tác lân cận và nhân tố ẩn.
- **2.2** 6 mô hình kèm công thức (mục 4.8), nhấn mạnh 3 mô hình đã cài và tham số pilot: ItemKNN k = 100; ALS 64 chiều, regularization 0,01, α = 1, 15 vòng, 1 luồng.
- **2.3** Recall@K, NDCG@K, hit ratio, prediction shift, Jaccard@10, Kendall tau; bootstrap 1.000 lần, CI 95%, bootstrap cặp.
- **3.4–3.7** Pipeline 6 bước; bộ gắn cờ và công thức trọng số tin cậy (nói rõ giá trị đang tạm); thiết kế tấn công 4 kiểu × 3 mức và 3 biện pháp; 3 nguyên tắc chống đánh giá vòng tròn (mục 4.3).
- **Duyệt code:** đọc `recsys/evaluation/metrics.py`, `bootstrap.py`, `recsys/models/popular.py`, `itemknn.py`, `als.py`; ví dụ tính tay ở `tests/recsys/test_eval_models.py`. Ghi mọi chỗ sai hoặc khó hiểu vào PR.
- **Tài liệu có sẵn:** mục 4.5, 4.8 của file này; slide 6–7, 9–21 trong file slide.
- **Xong khi:** viết thành văn có công thức; có ghi chú duyệt code.

### An Bình — mục 3.1–3.3 (~2–3 trang) + gán nhãn 300 review
- **3.1** JSON Lines, 2 file (review 9 trường, sản phẩm 15 trường); `user_id` giả danh hóa; che SĐT/email.
- **3.2** 408.160 → 353.035 review; giữ review sớm nhất mỗi cặp → 347.901 tương tác; đặc điểm (bảng mục 3.1) kèm **hệ quả cho thí nghiệm**; hình h1, h2, h3.
- **3.3** Mốc 06/11/2023 và 02/12/2024; k-core chỉ trên train; bảng warm/cold (mục 3.2); hình h4.
- **Gán nhãn:** xem hướng dẫn ngay dưới.
- **Xong khi:** viết thành văn có 4 hình; 300 dòng có nhãn; dán output lệnh `lambda`.

### Phúc Bảo — Chương 4, Chương 5 (~3 trang) + gán nhãn chéo 90 review
- **Chương 4:** thiết lập pilot; số đếm bộ gắn cờ; bảng kết quả và hình h5; 2 kiểm tra bổ sung; thảo luận và quyết định giữ RQ3. Mọi số và kết luận đã có ở mục 3.3; việc chính là **viết diễn giải**:
  - cách đọc khoảng tin cậy và Jaccard;
  - vì sao Jaccard của ALS thấp là do nhiễu seed (0,362 khi chỉ đổi seed);
  - bỏ tương tác bị gắn cờ làm NDCG **giảm nhẹ**: nêu giả thuyết, nói rõ **chưa kết luận** vì bộ gắn cờ chưa được kiểm bằng nhãn tay.
- **Chương 5:** đạt được, hạn chế (mục 3.5), kế hoạch (mục 5).
- **Xong khi:** viết thành văn có hình h5; 90 dòng có nhãn; dán output lệnh `kappa`.

### Hướng dẫn gán nhãn tay
**Mục đích:** bộ gắn cờ là luật tự động nên có thể bắt nhầm. Người đọc một mẫu để biết mỗi luật đúng bao nhiêu phần trăm (precision). Con số đó là hệ số λ trong trọng số tin cậy.

| File | Người | Số dòng |
|---|---|---|
| `data/labels/mau_gan_nhan.csv` | An Bình | 300 (100 mỗi dấu hiệu, đã trộn thứ tự) |
| `data/labels/mau_gan_nhan_cheo.csv` | Phúc Bảo, gán **độc lập** | 90 (30 dòng đầu mỗi dấu hiệu) |

Mở bằng Excel, điền cột `nhan` và `nguoi_gan`:
- **1 = rác / không phản ánh nhận xét thật về sản phẩm:** quảng cáo, kêu gọi liên hệ (zalo, inbox…), văn mẫu dán hàng loạt để nhận xu/quà, nội dung không liên quan sản phẩm.
- **0 = bình thường:** nhận xét thật về sản phẩm, kể cả khi ngắn hoặc giống người khác ("giao nhanh, hàng tốt").
- **Không chắc:** để trống, ghi lý do ở cột `ghi_chu`.

Xong thì chạy và dán kết quả vào PR:
```powershell
python -m recsys.mitigation.label_sample lambda   # An Bình: precision từng dấu hiệu
python -m recsys.mitigation.label_sample kappa    # Phúc Bảo, sau khi cả hai xong: độ đồng thuận
```
Kappa ≥ 0,6 là chấp nhận được; thấp hơn thì hai bạn thống nhất lại tiêu chí trước khi chốt λ.
**Hai file nhãn chứa nội dung review thật: không gửi ra ngoài nhóm, không chụp màn hình, không commit.**

### Checklist duyệt (Thiên Bảo)
- [ ] Số trong văn khớp mục 3; không có số "ước chừng".
- [ ] Không có nội dung review cá nhân, không có `user_id`.
- [ ] Mọi ý lấy từ bài báo có trích dẫn.
- [ ] Không có câu "đề xuất mô hình mới".
- [ ] Thuật ngữ thống nhất giữa các chương (user warm, phản hồi ngầm, bộ gắn cờ…).
- [ ] Đã có output `lambda` và `kappa`; cập nhật λ vào `recsys/config.py`, đặt `TRUST_LAMBDA_IS_PROVISIONAL = False`.


## 3. Số liệu cho báo cáo

Tất cả từ lần chạy thật. Hình ở `results/figures/` (vẽ lại: `python -m recsys.notebooks.figures`).

### 3.1 Tổng quan dữ liệu (nguồn: `results/prep_summary.json`)
| Chỉ số | Giá trị |
|---|---|
| Dòng review thô → sau lọc trùng | 408.160 → 353.035 |
| Người dùng / sản phẩm có review / danh mục | 227.557 / 7.360 / 9 |
| Cặp (user, sản phẩm) = tương tác | 347.901 (3.717 cặp có ≥ 2 review) |
| Phân bố sao 1–5★ | 5.016 · 2.376 · 5.366 · 29.656 · 310.621 (5★ = 88,0%) |
| Review không có nội dung / có nhãn đã mua | 60,5% / 99,9% |
| User chỉ có 1 tương tác | 75,7% |
| 10% sản phẩm phổ biến nhất chiếm | 77,3% tương tác |
| Danh mục sách chiếm | 63,4% tương tác |
| Mật độ ma trận user × sản phẩm | 0,02% |
| Review có SĐT / email trước khi che | 5 / 2 |
| Biến thể pos45 bỏ | 12.670 / 347.901 cặp 1–3★ (3,6%) |

### 3.2 Chia dữ liệu
t_val = 06/11/2023 (phân vị 80%), t_test = 02/12/2024 (90%); dữ liệu từ 09/2014 đến 10/2026.

| Bản chia | Train: tương tác / user / sản phẩm | Val: user warm / tổng | Test: user warm / tổng |
|---|---|---|---|
| all, 0-core | 278.320 / 198.339 / 3.915 | 6.447 / 20.152 (32,0%) | 3.329 / 12.672 (26,3%) |
| all, 2-core | 121.541 / 42.290 / 2.477 | 3.206 / 19.112 (16,8%) | 1.700 / 11.866 (14,3%) |
| all, 3-core | 66.474 / 15.255 / 1.693 | 1.769 / 17.352 (10,2%) | 980 / 10.807 (9,1%) |
| all, 5-core | 26.190 / 3.585 / 919 | 642 / 14.756 (4,4%) | 338 / 9.088 (3,7%) |

Ở 0-core, test mất 14.962 / 34.791 tương tác (43%) vì sản phẩm chưa có trong train.

### 3.3 Pilot (nguồn: `results/pilot/pilot_summary.json`; lệnh: `python -m recsys.experiments.pilot --k 0 2`)
**Thiết lập:** Popular, ItemKNN, ALS trên train gốc và train đã bỏ các cặp bị gắn cờ `dup_content` ∪ `promo`; đánh giá trên validation, user warm theo train gốc; seed 0.

**Bộ gắn cờ** (từ khóa promo là danh sách tạm):
| Dấu hiệu | Review | Cặp |
|---|---|---|
| dup_content | 2.200 | 2.138 |
| promo | 688 | 678 |
| burst | 14.594 | 9.460 |
| dup_content ∪ promo | — | 2.810 / 347.901 (0,8%) |

Train bị bỏ: 2.234 / 278.320 cặp (0-core); 1.321 / 121.541 (2-core).

**Kết quả** (ΔNDCG@10 = lọc − gốc, ×10⁻³, CI 95%):
| Bản chia | Mô hình | NDCG@10 gốc | NDCG@10 lọc | ΔNDCG@10 | Jaccard@10 |
|---|---|---|---|---|---|
| 0-core (6.447 user) | Popular | 0,0024 | 0,0016 | −0,87 [−1,26; −0,50] | 0,826 [0,825; 0,827] |
| | ItemKNN | 0,0247 | 0,0239 | −0,82 [−1,51; −0,23] | 0,892 [0,889; 0,896] |
| | ALS | 0,0176 | 0,0167 | −0,92 [−2,50; +0,58] | 0,343 [0,337; 0,348] |
| 2-core (3.206 user) | Popular | 0,0089 | 0,0084 | −0,53 [−1,16; +0,07] | 0,818 [0,817; 0,819] |
| | ItemKNN | 0,0328 | 0,0323 | −0,45 [−1,47; +0,58] | 0,874 [0,869; 0,880] |
| | ALS | 0,0290 | 0,0281 | −0,84 [−3,00; +1,36] | 0,451 [0,445; 0,457] |

**Kiểm tra 1 — nhiễu seed ALS:** cùng dữ liệu gốc, chỉ đổi seed 0 → 1: Jaccard@10 = 0,362 [0,357; 0,368] (0-core), 0,409 [0,403; 0,416] (2-core), ngang mức khi lọc → Jaccard thấp của ALS chủ yếu là nhiễu huấn luyện. Ở 2-core, đổi seed còn làm NDCG@10 đổi có ý nghĩa (−3,33×10⁻³ [−5,80; −1,01]) → giai đoạn 1 cần nhiều seed.

**Kiểm tra 2 — user mất hết lịch sử:** lọc làm 968 user warm (0-core) mất toàn bộ tương tác train, nhưng chỉ 25 người có trong validation; loại họ ra, ItemKNN 0-core vẫn có ΔNDCG@10 = −0,75 [−1,45; −0,12]×10⁻³, Jaccard 0,896.

**Quyết định (quy tắc định trước):** giữ RQ3. Căn cứ: Popular và ItemKNN (mô hình tất định, không nhiễu seed) có Jaccard@10 < 0,9 ở cả hai bản chia và ΔNDCG@10 có CI không chứa 0 ở 0-core.

**Diễn giải sơ bộ:** bỏ 0,8% tương tác làm **7–10% sản phẩm trong top-10** thay đổi ở hai mô hình tất định (trung bình giữ nguyên 9,04/10 sản phẩm với Popular và 9,34/10 với ItemKNN ở 0-core; 9,00 và 9,24 ở 2-core). Không quy đổi 1 − Jaccard thành tỉ lệ thay đổi: Jaccard 0,83 tương ứng khoảng 1 sản phẩm khác trên 10; bỏ tương tác bị gắn cờ làm NDCG giảm nhẹ (giả thuyết: các tương tác này tập trung vào sản phẩm vẫn được mua tiếp ở giai đoạn validation, ví dụ chiến dịch khuyến mãi) — **chưa kết luận** khi chưa có nhãn tay. NDCG tuyệt đối rất thấp (0,002–0,033) do dữ liệu cực thưa.

### 3.4 Bảng so sánh với Shrestha và cs. 2021
| Khía cạnh | Shrestha và cs. 2021 | Đồ án |
|---|---|---|
| Chia dữ liệu | 5-fold ngẫu nhiên | Mốc thời gian toàn cục (Ji và cs. 2023) |
| Tập test | Đổi theo khi bỏ spammer | Cố định, chỉ train thay đổi |
| Nhãn spam | Proxy: 4–5★ và helpfulness ≤ 0,4 | Gắn cờ + nhãn tay để đo precision |
| Kiểm định | Nhiều t-test, không hiệu chỉnh | Bootstrap cặp, CI 95% |
| Phạm vi | Giữ hoặc bỏ spammer | Thêm tấn công có kiểm soát + 3 biện pháp |
| Dữ liệu | Yelp, Amazon; dự đoán rating | Tiki; phản hồi ngầm, top-K |

### 3.5 Hạn chế cần nêu
- Nhóm user warm nhỏ (6.447 ở val, 0-core) → CI rộng.
- 43% tương tác test (0-core) bị loại vì sản phẩm mới.
- Bộ gắn cờ chưa được kiểm bằng nhãn tay; từ khóa promo là tạm.
- Pilot mới 1 seed; ALS dao động mạnh theo seed.

---

## 4. Đặc tả kỹ thuật

Các docstring trong code trỏ về mục này (ví dụ "docs/DO_AN.md mục 4.5.2").

### 4.1 Đề tài và câu hỏi nghiên cứu
Nghiên cứu **đo lường**: tương tác giả (review rác, tài khoản thuê) và độ lệch 5 sao bẻ cong các mô hình gợi ý phổ biến đến đâu trên dữ liệu Tiki, và 3 biện pháp có sẵn giảm nhẹ được bao nhiêu. Không đề xuất mô hình mới; kết quả âm là kết luận hợp lệ.
Sản phẩm chung cho 2 môn: IE212 (PySpark, `bigdata/`), DS300 (`recsys/`). Mỗi cấu hình thí nghiệm là hàm thuần `run_config(cfg)` (`cfg` = `recsys.schema.RunConfig`) để chạy phân tán sau này.
- **RQ1 (phụ):** tấn công có làm giảm chất lượng gợi ý chung (NDCG@K, user warm) và đổi thứ hạng mô hình (Kendall tau)?
- **RQ2 (lõi):** tấn công đẩy sản phẩm mục tiêu vào top-K được bao nhiêu (hit ratio@K, prediction shift)?
- **RQ3 (giữ theo pilot):** bỏ tương tác nghi ngờ thật có làm top-K đổi đáng kể (Jaccard@10, ΔNDCG@10)?
- **RQ4:** lọc cứng, trọng số tin cậy, T-CE/R-CE phục hồi được bao nhiêu (trên cùng nền MF-BCE)?

### 4.2 Dữ liệu
`data/raw/` không lên GitHub, không sửa. `reviews_anonymized.jsonl`: `review_id, user_id, product_id, rating, title, content, thank_count, created_at (unix giây), is_verified_purchase`. `products.jsonl`: `product_id, sku, name, category, brand_name, price, original_price, discount_rate, rating_average, review_count, all_time_quantity_sold, short_description, specifications, thumbnail_url, crawled_at`. Số liệu kiểm ở mục 3.1; prep tự đối chiếu với `config.EXPECTED*` và dừng nếu lệch.
Nội dung dài > 20 ký tự lặp ≥ 5 lần: 202 nội dung / 1.823 review khi chỉ strip; 228 / 2.200 theo chuẩn hóa của cờ `dup_content` (lowercase, gộp khoảng trắng).

### 4.3 Quy tắc bắt buộc
1. **Quyền riêng tư:** không in/ghi `user_id` kèm nội dung review hay nội dung review của từng người; kết quả chỉ ở dạng tổng hợp; SĐT → `[sdt]`, email → `[email]` ngay ở prep.
2. Không công bố dữ liệu thô; không đưa `data/` lên Git. 3. Không cào thêm; giữ nguyên `crawler/`.
4. Không bịa số: mọi số trong báo cáo/README/PR đến từ lần chạy thật.
5. **Chống đánh giá vòng tròn:** test không bao giờ bị lọc bằng bộ gắn cờ; ngưỡng và λ cố định trong `recsys/config.py` trước khi sinh dữ liệu tấn công; hồ sơ giả chỉ trong train, metric chỉ trên user thật.
6. Không chọn tham số theo test: mọi lựa chọn dựa trên validation. 7. Hàm ngẫu nhiên nhận `seed` tường minh.

### 4.4 Cấu trúc code
```
recsys/ config.py (mọi tham số) · schema.py (quy ước bảng, RunConfig)
        prep/ · models/ · attacks/ · mitigation/ · evaluation/ · experiments/ · notebooks/
tests/recsys/ (dữ liệu giả, không đọc data/raw) · results/{pilot,stage1,stage2,figures}/
```

### 4.5 Đặc tả từng phần
**4.5.1 Prep (xong).** Lọc trùng `review_id`/`product_id` → scrub (cột `had_phone`, `had_email` tính trước khi scrub) → mỗi cặp giữ review sớm nhất (hòa: `review_id` nhỏ hơn), `n_reviews` = số review của cặp → biến thể `all` / `pos45` (≥ 4★, áp cho cả train/val/test) → chia theo phân vị 80%/90% của thời gian trên biến thể `all`, dùng chung → k-core ∈ {0, 2, 3, 5} **chỉ trên train**, bỏ khỏi val/test sản phẩm không có trong train → warm = user val/test có ≥ 1 tương tác train. Lệnh: `python -m recsys.prep.build_interactions --star all pos45 --k 0 2 3 5`.

**4.5.2 Mô hình.** Kế thừa `recsys.models.base.Recommender`, chỉ cài `_fit(R, …)` và `_score(user_idx, item_idx)`; `R` = ma trận user × item, ô = w_ui (không có w thì 1).
| Mô hình | Cài đặt | Dùng w | Trạng thái |
|---|---|---|---|
| Popular | đếm tương tác | tổng w | xong |
| ItemKNN | cosine giữa cột, k láng giềng | R có trọng số trước khi tính cosine | xong |
| ALS | `implicit` | c = 1 + α·w | xong |
| BPR | `implicit` hoặc torch | nhân w vào loss cặp (u,i,j), i dương | chưa |
| MF-BCE | torch, BCE, 4 âm / 1 dương | nhân w vào số hạng dương | chưa |
| LightFM | `lightfm`, đặc trưng sản phẩm | không dùng w | chưa |

**4.5.3 Đánh giá.** Ground truth = sản phẩm user tương tác trong val/test; gợi ý loại sản phẩm đã có trong train. Metric theo từng user rồi lấy trung bình: Recall@K, NDCG@K (K = 10, 20), coverage, Jaccard@10; tấn công: hit ratio@K, prediction shift (so cùng mô hình, cùng seed trên dữ liệu sạch). RQ1–RQ4 trên user warm; Popular và LightFM báo thêm cold. Bootstrap 1.000 lần trên vectơ metric theo user, CI 95% percentile; bootstrap cặp khi so hai phiên bản; khoảng chứa 0 → "không phân biệt được". Mỗi lần chạy lưu `results/<stage>/<run_id>/per_user.parquet` (user đã băm) và `summary.json`.

**4.5.4 Bộ gắn cờ (chỉ 3 dấu hiệu).** `dup_content`: nội dung chuẩn hóa > 20 ký tự, xuất hiện ≥ 5 lần. `promo`: từ khóa trong `PROMO_KEYWORDS` hoặc có SĐT trước khi scrub. `burst`: cùng user ≥ 2 review cùng sản phẩm, hoặc ≥ N sản phẩm trong T phút (N = 5, T = 60, **tạm**). Không dùng "chưa xác thực mua", "luôn 5 sao không nội dung", "lệch cảm xúc".

**4.5.5 Trọng số tin cậy.** w_ui = max(ε, 1 − min(1, Σ_k λ_k · s_k(u,i))), ε = 0,05; λ_k = precision của dấu hiệu k trên nhãn tay, cố định trước thí nghiệm, không chỉnh theo metric gợi ý.

**4.5.6 Khử nhiễu.** T-CE (bỏ tỉ lệ ε(t) mẫu dương có loss lớn nhất, ε(t) tăng dần đến mức tối đa) và R-CE (trọng số loss mẫu dương = điểm dự đoán^β) theo Wang và cs. 2021, cài trên MF-BCE; siêu tham số đối chiếu mã gốc DenoisingRec.

**4.5.7 Tấn công (đẩy sản phẩm).** Kiểu: `random`, `average`, `bandwagon`, `observed` (bắt chước nhóm săn sale: cùng nội dung, đăng dồn dập). Mức: hồ sơ giả = 1%, 3%, 5% số user train; mỗi hồ sơ = sản phẩm mục tiêu + filler (số filler = trung vị tương tác/user train). Mục tiêu mỗi seed: 10 sản phẩm ít phổ biến + 10 phổ biến trung bình. Thời điểm giả trong khoảng train; `user_id` âm, `is_fake = True`; 5 seed.

### 4.6 Kế hoạch thí nghiệm
| Bước | Nội dung | Lần huấn luyện |
|---|---|---|
| Pilot | xong (mục 3.3) | 12 |
| Giai đoạn 1 | 6 mô hình × 4 k-core × 2 biến thể sao × 3 seed; chọn 1 k + 1 biến thể theo cỡ nhóm warm và độ rộng CI trên validation | 144 |
| 2a | 6 mô hình × (sạch + 4 kiểu × 3 mức) × 5 seed | 390 |
| 2b | MF-BCE × {lọc cứng, trọng số, T-CE, R-CE} × 12 × 5 | 240 |
| 2c | trọng số tin cậy trên ALS, BPR, ItemKNN × 12 × 5 | 180 |
Thiếu thời gian: giai đoạn 2 giảm từ 5 xuống 3 seed.

### 4.7 Quy ước code
Python có type hint, docstring tiếng Việt ngắn, `pathlib`, `logging` (không `print` dữ liệu cá nhân); hàm thuần, chỉ phụ thuộc `config.py`; test trong `tests/recsys/` chạy `pytest -q`; quy trình PR trong `CONTRIBUTING.md`.

### 4.8 Công thức
- ALS (Hu và cs. 2008): min Σ c_ui (p_ui − x_uᵀy_i)² + λ(‖x‖² + ‖y‖²), p_ui = 1 nếu có tương tác, c_ui = 1 + α·r_ui.
- BPR (Rendle và cs. 2009): max Σ ln σ(ŷ_ui − ŷ_uj) − λ‖Θ‖², i đã tương tác, j chưa.
- BCE: ℓ = −[y·log σ(ŷ) + (1 − y)·log(1 − σ(ŷ))].
- ItemKNN: sim(i, j) = cos(R·i, R·j); điểm(u, i) = Σ_j r_uj · sim_k(j, i).
- Recall@K = |top-K ∩ G_u| / |G_u|. NDCG@K = DCG@K / IDCG@K, DCG@K = Σ_{k=1..K} rel_k / log₂(k + 1).
- Jaccard = |A ∩ B| / |A ∪ B|. Hit ratio@K = tỉ lệ user thật có ≥ 1 sản phẩm mục tiêu trong top-K.



## 6. Tài liệu tham khảo

**Bài gốc**
- Ji, Sun, Zhang, Li. *A Critical Study on Data Leakage in Recommender System Offline Evaluation*. ACM TOIS 2023. [arXiv:2010.11060](https://arxiv.org/abs/2010.11060)
- Wang, Feng, He, Nie, Chua. *Denoising Implicit Feedback for Recommendation*. WSDM 2021. [arXiv:2006.04153](https://arxiv.org/abs/2006.04153) · mã: [DenoisingRec](https://github.com/WenjieWWJ/DenoisingRec)
- Shrestha, Spezzano, Pera. *An Empirical Analysis of Collaborative Recommender Systems Robustness to Shilling Attacks*. OHARS @ RecSys 2021, CEUR-WS Vol-3012. [PDF](https://ceur-ws.org/Vol-3012/OHARS2021-paper4.pdf)

**Tấn công và độ bền vững**
- Lam, Riedl. *Shilling recommender systems for fun and profit*. WWW 2004. [doi](https://doi.org/10.1145/988672.988726)
- Burke, O'Mahony, Hurley. *Robust collaborative recommendation*. Recommender Systems Handbook, 2015. *(chép từ danh mục của Shrestha 2021, chưa kiểm lại)*
- Zhang, Cao, Sun và cs. *Robust Recommender System: A Survey and Future Directions*. ACM Computing Surveys. [arXiv:2309.02057](https://arxiv.org/abs/2309.02057)
- Wang, Yu, Gao và cs. *Poisoning Attacks and Defenses in Recommender Systems: A Survey*. 2024. [arXiv:2406.01022](https://arxiv.org/abs/2406.01022)

**Mô hình**
- Hu, Koren, Volinsky. *Collaborative Filtering for Implicit Feedback Datasets*. ICDM 2008. [doi](https://doi.org/10.1109/ICDM.2008.22)
- Rendle và cs. *BPR: Bayesian Personalized Ranking from Implicit Feedback*. UAI 2009. [arXiv:1205.2618](https://arxiv.org/abs/1205.2618)
- Ferrari Dacrema, Cremonesi, Jannach. *Are we really making much progress?* RecSys 2019. [arXiv:1907.06902](https://arxiv.org/abs/1907.06902)

**Đánh giá**
- Gusak, Frolov. *Your 5-Core Train Split May Not Be 5-Core*. RecSys 2026. [doi](https://doi.org/10.1145/3773078.3841280)
- Sun. *Take a Fresh Look at Recommender Systems from an Evaluation Standpoint*. SIGIR 2023. [arXiv:2210.04149](https://arxiv.org/abs/2210.04149)
- Tamm, Damdinov, Vasilev. *Quality Metrics in Recommender Systems: Do We Calculate Metrics Consistently?* RecSys 2021. [arXiv:2206.12858](https://arxiv.org/abs/2206.12858)
- Meng, McCreadie, Macdonald, Ounis. *Exploring Data Splitting Strategies for the Evaluation of Recommendation Models*. RecSys 2020. [arXiv:2007.13237](https://arxiv.org/abs/2007.13237)

**Khử nhiễu (tiếp nối Wang 2021)**
- Gao và cs. *Self-Guided Learning to Denoise for Robust Recommendation*. SIGIR 2022. [arXiv:2204.06832](https://arxiv.org/abs/2204.06832)
- He và cs. *Double Correction Framework for Denoising Recommendation*. KDD 2024. [arXiv:2405.11272](https://arxiv.org/abs/2405.11272)

Trước khi đưa vào báo cáo, mở link kiểm lại thông tin xuất bản.
