# Schema dữ liệu

## `data/raw/reviews.jsonl`

Mỗi dòng một review (JSON, UTF-8). Khóa: `review_id`.

| Trường | Kiểu | Ghi chú |
|---|---|---|
| review_id | int | khóa, dùng để lọc trùng |
| product_id | int | |
| user | str(16) | HMAC-SHA256 của customer_id, 16 ký tự hex |
| rating | int 1–5 | |
| created_at | int | unix timestamp (giây) |
| title | str | đã scrub (SĐT → `[sdt]`, email → `[email]`) |
| content | str | đã scrub |
| thank_count | int | |
| purchased | bool | đã mua hàng xác thực |
| user_total_review | int | tổng review toàn sàn của người dùng (công khai) |

Không lưu: `customer_id` gốc, tên, `full_name`, ảnh đại diện, `region`, `purchased_at`, ảnh review.

## `data/raw/products.csv`

| Cột | Kiểu | Nguồn API |
|---|---|---|
| id | int | `id` |
| name | str | `name` |
| brand | str | `brand_name` |
| price | int | `price` |
| rating_average | float | `rating_average` |
| review_count | int | `review_count` |
| category | int | ID danh mục lấy từ URL |

## `data/raw/products.done`

Mỗi dòng một `product_id` đã cào xong review; chạy lại sẽ bỏ qua.

## `data/raw/crawl.log`

Log của lần cào (không chứa thông tin cá nhân).
