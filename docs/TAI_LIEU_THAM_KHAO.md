# Tài liệu tham khảo

Danh sách lấy từ Semantic Scholar (tra ngày 2026-11-09). Gồm các bài mà hai bài gốc **tham chiếu** (nền tảng)
và các bài **trích dẫn** hai bài gốc (tiếp nối), đã lọc theo mức liên quan tới đồ án.
"TD" là số lượt trích dẫn theo Semantic Scholar vào ngày tra. Trước khi đưa vào báo cáo, mở link để kiểm lại thông tin xuất bản.

## A. Đánh giá offline và chia dữ liệu theo thời gian

**Bài gốc:** Yitong Ji, Aixin Sun, Jie Zhang, Chenliang Li. *A Critical Study on Data Leakage in Recommender System Offline Evaluation*.
ACM TOIS 41(3), 2023 (arXiv 2020). [arXiv:2010.11060](https://arxiv.org/abs/2010.11060).
Dùng cho prep: chia theo mốc thời gian toàn cục (`docs/DAC_TA.md` mục 5.1).

### A1. Nền tảng (bài gốc tham chiếu)
| Bài | Liên quan tới đồ án |
|---|---|
| Kaufman, Rosset, Perlich và cs. *Leakage in data mining: formulation, detection, and avoidance*. TKDD 2012. [doi:10.1145/2020408.2020496](https://doi.org/10.1145/2020408.2020496) | Định nghĩa rò rỉ dữ liệu nói chung, làm cơ sở lý luận cho việc k-core chỉ áp trên train |
| Campos, Díez, Cantador. *Time-aware recommender systems: a comprehensive survey and analysis of existing evaluation protocols*. UMUAI 2014. [doi:10.1007/s11257-012-9136-x](https://doi.org/10.1007/s11257-012-9136-x) | Tổng quan các giao thức đánh giá có yếu tố thời gian |
| Meng, McCreadie, Macdonald, Ounis. *Exploring Data Splitting Strategies for the Evaluation of Recommendation Models*. RecSys 2020. [arXiv:2007.13237](https://arxiv.org/abs/2007.13237) | Cách chia dữ liệu làm đổi thứ hạng mô hình, gần với RQ1 (Kendall tau) |
| Ferrari Dacrema, Cremonesi, Jannach. *Are we really making much progress?* RecSys 2019. [arXiv:1907.06902](https://arxiv.org/abs/1907.06902) | Baseline đơn giản (ItemKNN, Popular) thường không thua mô hình phức tạp, giải thích vì sao đồ án giữ các baseline này |
| Krichene, Rendle. *On Sampled Metrics for Item Recommendation*. KDD 2020. [doi:10.1145/3394486.3403226](https://doi.org/10.1145/3394486.3403226) | Lý do tính metric trên toàn bộ item, không lấy mẫu |
| Sun, Yu, Fang và cs. *Are We Evaluating Rigorously? Benchmarking Recommendation for Reproducible Evaluation and Fair Comparison*. RecSys 2020. [doi:10.1145/3383313.3412489](https://doi.org/10.1145/3383313.3412489) | Quy trình đánh giá chặt chẽ, có thể tái lập |

### A2. Tiếp nối (trích dẫn bài gốc)
| Bài | Liên quan tới đồ án |
|---|---|
| Aixin Sun. *Take a Fresh Look at Recommender Systems from an Evaluation Standpoint*. SIGIR 2023 (Perspectives). [arXiv:2210.04149](https://arxiv.org/abs/2210.04149) | Mở rộng lập luận về mốc thời gian toàn cục; định nghĩa lại baseline Popular |
| Ji, Sun, Zhang, Li. *Do Loyal Users Enjoy Better Recommendations? Understanding Recommender Accuracy from a Time Perspective*. ICTIR 2022. [arXiv:2204.05927](https://arxiv.org/abs/2204.05927) | Độ chính xác theo nhóm user. Liên quan việc tách warm/cold và nhóm warm nhỏ của Tiki |
| Gusak, Frolov. *Your 5-Core Train Split May Not Be 5-Core: Core Filtering and Temporal Splitting Do Not Commute*. RecSys 2026. [doi:10.1145/3773078.3841280](https://doi.org/10.1145/3773078.3841280) | **Rất sát:** thứ tự k-core và chia thời gian, đúng lựa chọn "k-core chỉ trên train" của đồ án |
| Pechatov. *Split Decisions: How Evaluation Protocols Change Recommender Model Selection*. CIKM 2026. [doi:10.1145/3799682.3839942](https://doi.org/10.1145/3799682.3839942) | Giao thức đánh giá làm đổi lựa chọn mô hình, gần với RQ1 |
| Zhao, Lin, Feng và cs. *A Revisiting Study of Appropriate Offline Evaluation for Top-N Recommendation Algorithms*. TOIS 2022. [doi:10.1145/3545796](https://doi.org/10.1145/3545796) | So sánh có hệ thống các thiết lập đánh giá top-N |
| Tamm, Damdinov, Vasilev. *Quality Metrics in Recommender Systems: Do We Calculate Metrics Consistently?* RecSys 2021. [arXiv:2206.12858](https://arxiv.org/abs/2206.12858) | Các thư viện tính NDCG/Recall khác nhau, cần cho B1 (metric) |
| Shevchenko, Belousov, Vasilev và cs. *From Variability to Stability: Advancing RecSys Benchmarking Practices*. KDD 2024. [arXiv:2402.09766](https://arxiv.org/abs/2402.09766) | Độ biến động theo seed và dataset; liên quan bootstrap và nhiều seed |
| Gusak, Volodkevich, Klenitskiy và cs. *Time to Split: Exploring Data Splitting Strategies for Offline Evaluation of Sequential Recommenders*. RecSys 2025. [arXiv:2507.16289](https://arxiv.org/abs/2507.16289) | So sánh các kiểu chia theo thời gian |
| Fan, Ji, Zhang, Sun. *Our Model Achieves Excellent Performance on MovieLens: What Does It Mean?* TOIS 2023. [arXiv:2307.09985](https://arxiv.org/abs/2307.09985) | Đặc điểm dataset ảnh hưởng tới kết luận, lý do đồ án dùng dữ liệu Tiki thật |

## B. Khử nhiễu implicit feedback, tương tác giả và độ bền vững

**Bài gốc:** Wenjie Wang, Fuli Feng, Xiangnan He, Liqiang Nie, Tat-Seng Chua. *Denoising Implicit Feedback for Recommendation*.
WSDM 2021. [arXiv:2006.04153](https://arxiv.org/abs/2006.04153), [doi:10.1145/3437963.3441800](https://doi.org/10.1145/3437963.3441800), mã: [DenoisingRec](https://github.com/WenjieWWJ/DenoisingRec).
Dùng cho T-CE/R-CE (`docs/DAC_TA.md` mục 5.6).

### B1. Nền tảng (bài gốc tham chiếu)
| Bài | Liên quan tới đồ án |
|---|---|
| Hu, Koren, Volinsky. *Collaborative Filtering for Implicit Feedback Datasets*. ICDM 2008. [doi:10.1109/ICDM.2008.22](https://doi.org/10.1109/ICDM.2008.22) | Gốc của ALS với confidence `1 + α·w`, mô hình C3 |
| Rendle, Freudenthaler, Gantner, Schmidt-Thieme. *BPR: Bayesian Personalized Ranking from Implicit Feedback*. UAI 2009. [arXiv:1205.2618](https://arxiv.org/abs/1205.2618) | Mô hình BPR (S1) |
| Han, Yao, Yu và cs. *Co-teaching: Robust training of deep neural networks with extremely noisy labels*. NeurIPS 2018. [arXiv:1804.06872](https://arxiv.org/abs/1804.06872) | Ý tưởng "mẫu loss lớn là nhiễu" mà T-CE kế thừa |
| Lam, Riedl. *Shilling recommender systems for fun and profit*. WWW 2004. [doi:10.1145/988672.988726](https://doi.org/10.1145/988672.988726) | **Gốc của tấn công random/average** (E1) |
| Chen, Dong, Wang và cs. *Bias and Debias in Recommender System: A Survey and Future Directions*. TOIS 2023. [arXiv:2010.03240](https://arxiv.org/abs/2010.03240) | Tổng quan về thiên lệch, liên quan độ lệch 5 sao và biến thể `pos45` |

### B2. Tiếp nối (trích dẫn bài gốc)
| Bài | Liên quan tới đồ án |
|---|---|
| Zhang, Cao, Sun và cs. *Robust Recommender System: A Survey and Future Directions*. ACM Computing Surveys (arXiv 2023). [arXiv:2309.02057](https://arxiv.org/abs/2309.02057) | **Nên đọc đầu tiên.** Tổng quan cả tấn công lẫn nhiễu, giúp đặt đồ án vào bối cảnh |
| Wang, Yu, Gao và cs. *Poisoning Attacks and Defenses in Recommender Systems: A Survey*. arXiv 2024. [arXiv:2406.01022](https://arxiv.org/abs/2406.01022) | Tổng quan tấn công đầu độc, gồm các kiểu random/average/bandwagon của E1 |
| Wang, Xin, Meng và cs. *Learning Robust Recommenders through Cross-Model Agreement*. WWW 2022. [arXiv:2105.09605](https://arxiv.org/abs/2105.09605) | Khử nhiễu bằng sự đồng thuận giữa các mô hình; có thể dùng làm baseline so với T-CE/R-CE |
| Gao, Du, Hu và cs. *Self-Guided Learning to Denoise for Robust Recommendation*. SIGIR 2022. [arXiv:2204.06832](https://arxiv.org/abs/2204.06832) | Cải tiến trực tiếp ADT (T-CE/R-CE) |
| Tian, Xie, Li và cs. *Learning to Denoise Unreliable Interactions for Graph Collaborative Filtering*. SIGIR 2022. [doi:10.1145/3477495.3531889](https://doi.org/10.1145/3477495.3531889) | Khử nhiễu tương tác không đáng tin trên mô hình đồ thị |
| Wang, Gao, Li và cs. *Efficient Bi-Level Optimization for Recommendation Denoising*. KDD 2023. [arXiv:2210.10321](https://arxiv.org/abs/2210.10321) | Học trọng số cho từng tương tác, gần với "trọng số tin cậy" (E2) nhưng học tự động |
| Lin, Zhao, Wang và cs. *AutoDenoise: Automatic Data Instance Denoising for Recommendations*. WWW 2023. [arXiv:2303.06611](https://arxiv.org/abs/2303.06611) | Chọn tương tác sạch tự động |
| He, Wang, Yang và cs. *Double Correction Framework for Denoising Recommendation*. KDD 2024. [arXiv:2405.11272](https://arxiv.org/abs/2405.11272) | Hướng mới sau T-CE/R-CE, có so sánh với ADT |
| Zhang, Cao, Wu và cs. *Personalized Denoising Implicit Feedback for Robust Recommender System*. WWW 2025. [arXiv:2502.00348](https://arxiv.org/abs/2502.00348) | Chỉ ra hạn chế của ngưỡng loss toàn cục kiểu T-CE |
| Wang, Xu, Yang và cs. *Implicit Feedbacks are Not Always Favorable: Iterative Relabeled One-Class Collaborative Filtering against Noisy Interactions*. ACM MM 2021. [doi:10.1145/3474085.3475446](https://doi.org/10.1145/3474085.3475446) | Gán nhãn lại tương tác nhiễu, so sánh được với lọc cứng |
| Li, Zheng, Wang và cs. *Debiased Recommendation with Noisy Feedback*. KDD 2024. [arXiv:2406.17182](https://arxiv.org/abs/2406.17182) | Kết hợp nhiễu và thiên lệch, gần với độ lệch 5 sao |

## Gợi ý đọc theo người

| Người | Nên đọc trước |
|---|---|
| N1 (prep, thí nghiệm) | Ji 2023; Sun 2023; Gusak & Frolov 2026; Meng 2020 |
| N2 (metric, tấn công) | Tamm 2021; Krichene & Rendle 2020; Lam & Riedl 2004; khảo sát poisoning (Wang 2024) |
| N3 (ALS, BPR, trọng số) | Hu 2008; Rendle 2009; Ferrari Dacrema 2019; bi-level denoising (Wang 2023) |
| N4 (gắn cờ, khử nhiễu) | Wang 2021 (bài gốc); Han 2018; Gao 2022; khảo sát robust (Zhang 2025) |

## Còn thiếu
Hai bài gốc không bao phủ hết các chủ đề sau, nên cần tìm riêng:
- tấn công **bandwagon** (bài gốc thường được trích là Mobasher và cs. 2007);
- **phát hiện review rác** dựa trên nội dung trùng và đăng dồn dập (dùng cho D1);
- **độ lệch rating J-shape** trên các sàn thương mại điện tử.
