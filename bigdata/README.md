# bigdata: Phần Dữ liệu lớn (IE212 — PySpark)

Triển khai pipeline dữ liệu lớn bằng **PySpark + Medallion Parquet (Bronze → Silver → Gold)** chạy trực tiếp trên **Google Colab / Kaggle** (không cần cài Docker/HDFS về máy), dùng chung quy ước với `recsys/` ([docs/DO_AN.md](../docs/DO_AN.md)).

## Kiến trúc & các module

| File | Vai trò |
|---|---|
| `bigdata/config.py` | Chuẩn hóa đường dẫn Medallion (`resolve_paths`) và khởi tạo `SparkSession` tối ưu (`build_spark`) |
| `bigdata/etl.py` | PySpark ETL (lọc trùng, scrub PII, tương tác, chia thời gian, k-core), bộ gắn 3 cờ bằng Window Functions (`dup_content`, `promo`, `burst`) và `compare_parity` đối chiếu 100% với Pandas |
| `bigdata/spark_als.py` | Huấn luyện `pyspark.ml.recommendation.ALS(implicitPrefs=True)` và đối chiếu với `implicit` ALS của DS300 |
| `bigdata/runner.py` | Chạy phân tán danh sách `recsys.schema.RunConfig` qua Spark RDD (`broadcast` bảng cờ, hỗ trợ resume/skip khi Colab ngắt) |
| `bigdata/benchmark.py` | Benchmark Pandas vs PySpark (1×, 5×, 10×), tối ưu Spark (`JSONL 200p` → `Parquet 200p` → `Parquet Tuned`), đo Strong Scaling (1–2–4 cores) và xuất 3 hình `bd1–bd3` |
| `bigdata/notebooks/ie212_colab.ipynb` | Notebook bấm chạy trọn gói trên Google Colab hoặc Kaggle |

## Chạy nhanh trên Google Colab / Kaggle

1. Upload 2 file `reviews_anonymized.jsonl` và `products.jsonl` lên Google Drive nhóm (ví dụ `MyDrive/tiki-recsys/data/raw/`).
2. Mở `bigdata/notebooks/ie212_colab.ipynb` trên Colab (hoặc Kaggle để có 4 vCPU).
3. Chạy lần lượt các cell: cài `requirements-bigdata.txt` → ETL + Parity Check → Spark MLlib ALS → Distributed Runner → Benchmark.

Hoặc chạy qua CLI (khi môi trường đã có `pyspark` + Java):

```bash
pip install -r requirements.txt -r requirements-recsys.txt -r requirements-bigdata.txt
python -m bigdata.etl --star all pos45 --k 0 2 3 5
python -m bigdata.spark_als --k 0 2 --seeds 0 1
python -m bigdata.runner --suite pilot --mode spark
python -m bigdata.benchmark --scales 1 5 10
```

## Nối với Giai đoạn 1–2 của DS300

Khi Giai đoạn 1–2 bên `recsys/` bổ sung các mô hình `bpr`, `mf_bce`, `lightfm`, `bigdata/runner.py` sẽ tự động nhận diện qua `available_models()` để chạy phân tán danh sách `RunConfig` qua Spark.

## Kiểm thử

```bash
pytest -q
```

Các unit test không phụ thuộc Spark trong `tests/bigdata/test_bigdata.py` luôn chạy trên mọi máy; các test PySpark dùng `pytest.importorskip("pyspark")` nên tự bỏ qua an toàn ở máy local chưa cài Spark/Java.
