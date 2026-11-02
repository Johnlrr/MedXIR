# Tiến trình MedXIR

**Cập nhật:** 30/10/2026 (UTC+07:00). Bản bàn giao: phần "Bàn giao" bên dưới là trạng thái hiện tại. Các mục từ 1 trở đi là nhật ký theo thời gian; một số khẳng định cũ (ví dụ "đang chạy", "chưa có điểm" ở mục 16) đã lỗi thời và được ghi đè bởi phần bàn giao.

> **Tốt nhất 31/10: FINAL 0,2984** (`v2ens-bgefull-drr-dp03bge-llm10-d200-c131-x1-lz`): công thức 0,2755 (ensemble dense 4B + Qwen3-8B + Octen-8B với query gốc, PRF, biến thể LLM + BM25 v2, gộp min-max trong `fuse_scores.py`, bge doc rerank, cửa sổ 3 chunk) + bge-reranker chấm toàn pool của ensemble + ZIP LZMA (131 cửa sổ) + doc-prior bge 0,3 + Qwen3-8B rerank 120 doc đầu (RRF với bge ở vế doc). Chi tiết: mục "Lên 0,2984: bge toàn pool, nén, doc-prior, LLM rerank — 31/10/2026" cuối file. Các mục trước: "Ghép hai hướng trên corpus v2 — 30/10/2026", hướng A "Hybrid retrieval và ghép chunk", hướng B "Ensemble 8B, cửa sổ, PRF và biến thể query" và [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Bàn giao

### 1. Trạng thái

- **Điểm tốt nhất trên leaderboard: FINAL 0,2984** (`v2ens-bgefull-drr-dp03bge-llm10-d200-c131-x1-lz`, corpus v2; Doc F2 0,3235, Chunk F2 0,2734). Trước đó 0,2755 (`v2ens-bge25-drr-d200-c76-x1`, hai hướng ghép). Hướng A riêng: 0,2539 (`v2 fusion 4B 0,35 · Qwen3-8B 0,35 · Octen-8B 0,30 nb60-d200-drr-bge-only`). Hướng B riêng (corpus v1): 0,2444 (`var-all-b1-d200-c70-x1`).
- Lịch sử điểm hướng A: baseline 0,0818 → d100 0,1453 → d200 0,1517 → d200-c100 0,1783 → d200-c150 0,1847 → hybrid 0,1965 → expand nb30 0,2184 → expand nb60 0,2235 → nb60-drr-bge 0,2253 → v2 nb60-d200 0,2437 → v2 nb60-d200-drr-bge-only 0,2464 → ghigh 0,2477.
- Lịch sử điểm hướng B: d200-c150 0,1847 → ens4-b1 0,1975 → ens4-b1-x1 (cửa sổ 3 chunk) 0,2410 → Octen-PRF 0,2429 → biến thể LLM 0,2444 → **ghép hai hướng trên v2 0,2755**.
- **Nút thắt là doc recall (~0,47)**, bằng nhau ở mọi bài hybrid. Mục tiêu tiếp theo: doc recall ~0,56 (ensemble embedding 4B + Qwen3-8B + Octen-8B).
- Hạn public 30–31/10, nộp hệ thống 04/11. Quota 10 lượt/ngày; hỏi trước khi nộp.
- File mới từ 27/10 chưa commit (xem mục 5).

### 2. Metric (quan trọng cho mọi quyết định)

- FINAL = trung bình (Doc F2 macro, Chunk F2 macro). Ví dụ d200-c150: (0,2612 + 0,1081) / 2 = 0,1847.
- Chunk đúng nếu LCS token ≥ 40% đoạn tham chiếu, cùng `doc_id`. Chia cho độ dài đoạn tham chiếu, không phải đoạn nộp.
- **Điểm chỉ phụ thuộc tập hợp, không phụ thuộc thứ tự.** Sắp lại chunk hay doc không làm đổi điểm. Chỉ thay đổi tập chunk/doc mới có tác động.
- Doc F2 đạt đỉnh quanh 200 doc. Chunk F2 tăng theo số chunk nhưng giảm dần: 20 → 0,042; 100 → 0,095; 150 → 0,108.
- Giới hạn dashboard: file nộp tối đa 100 MB. Với chunk text nguyên văn, khoảng 150 chunk/query là tối đa. c200 nặng 121 MB, không nộp được.

### 3. Dữ liệu

- **Corpus chính thức:** `dataset/links_corpus_vi_en_zh.parquet`, 1.220.598 ID, id và url khớp kế hoạch recrawl 100%.
- **Corpus v1** (`artifacts/corpus_v1/{en,vi,zh}.parquet`): 1.027.516 ID, đều nằm trong parquet, url khớp 100%. Đây là nền cho chunk và embedding hiện tại. Phủ 97,2% ID không thuộc nguồn bị chặn (chặn: 120ask, familydoctor.com.cn, zysjonline, iiyi, laodong; BTC xác nhận không tính).
- Pending 26.179 ID (suckhoedoisong, longchau theo quota), lỗi 3.628 ID. Không tái crawl.
- **Chưa nhập vào corpus:** `artifacts/corpus_v1_patch/crawl_gap_v1_1.jsonl` (140 tài liệu, đã kiểm tra url) và 4.753 tài liệu VI từ 7 batch sau khi build v1 (cần lọc nguồn bị chặn). Nhập sẽ kéo theo chunk và embed mới.
- **ZH parser v2.3:** khoảng 72k tài liệu cần reparse lên v2.4+. Chưa đo tác động. Hoãn.
- Đã xóa: `artifacts/crawl/`, `artifacts/crawl_full/`, `artifacts/readiness_check/`, `.work/` (debug). Giữ: `artifacts/recrawl_20261015/` (state.sqlite là checkpoint resume; cc_v1 4,7 GB và pharmacity_v1 772 MB chỉ có ở local, chưa upload).
- Drive: `gdrive:` dùng `--drive-root-folder-id 1tfvL8OJIRejT_c-4wfaQnTZecT67qJld`. Có batch recrawl vi/en/zh và wayback_v1 (đã xác minh). Các batch crawl pilot và crawl_full không có trên Drive.

### 4. Vector, chunk, BM25

- Chunk: `artifacts/kaggle/chunk-corpus-v1/chunks/{en,vi,zh}/part-*.parquet`, `hash_groups.parquet`. Chunk 350/50/40, chunk ID = `text_sha256[:16]-order`.
- Vector 4B: 9 part trên Kaggle `boinhbo/medxir-vectors-qwen3-4b-s0..s2` (13 GB). Bản `.npy` local đã xóa; còn `ids.parquet` và `artifacts/submission/candidates-4b-k500.npz` (top-500 dense cho 1.200 query, cache để chạy lại không cần GPU).
- Query vector 4B: `artifacts/kaggle/query-vectors-4b/`.
- BM25 tiếng Việt: `artifacts/bm25/vi-candidates-k500.npz` (top-500 cho 1.200 query, kết quả Kaggle `boinhbo/medxir-bm25-vi`, `scripts/bm25_vi.py`). BM25 đứng riêng đạt 0,098; RRF 1:1 làm giảm điểm (0,1433 so với 0,1517).

### 5. Script chính

- `scripts/retrieve_local_4b.py`: tạo bài nộp dense. Tham số `--doc-counts`, `--chunk-counts`; nén zip mức 9, JSON gọn.
- `scripts/fuse_dense_bm25.py`: RRF dense + BM25 (`--w-dense`, `--w-bm25`, `--doc-counts`, `--chunk-counts`).
- `scripts/build_rerank_submission.py`, `scripts/build_rerank_candidates.py`, `scripts/build_rerank_notebooks.py`: nhánh reranker, đã dừng.
- `scripts/bm25_vi.py`: BM25 tiếng Việt (chạy trên Kaggle CPU, không chạy local vì OOM).
- `scripts/hybrid_retrieval.py` (logic thuần, có test `tests/test_hybrid_retrieval.py`) và `scripts/build_hybrid_notebook.py` → `src/notebooks/kaggle_hybrid_retrieval.ipynb`, config `configs/kaggle/hybrid_retrieval_rtx6000.yaml`. Dense 4B + BM25 + rerank bge-v2-m3 trên pool, toggle ở cell đầu.
- `scripts/expand_chunks.py` (Kaggle CPU, 3–5 phút): dựng lại phần chunk của bài nộp từ `pool.parquet` (biến thể `nbN`, `wholedoc`), giữ nguyên danh sách doc, giữ ZIP < 99 MB.

### 6. Kaggle

- Mọi job nặng chạy trên Kaggle qua `uv run python -m tools.kaggle kernel run --config configs/kaggle/<job>.yaml --source . --account boinhbo`. Internet tắt cho RTX Pro 6000.
- **Để chọn RTX Pro 6000:** `competition_sources: [arc-prize-2026-arc-agi-3]` và Internet tắt. Thiếu `competition_sources` thì CLI có thể cấp Tesla T4 mà không báo lỗi. Notebook nên kiểm tra tên GPU (đã có ở `kaggle_rerank_benchmark`).
- `kaggle kernels output/status/logs` chỉ đọc version mới nhất. Khi có nhiều version, dùng giao diện web để lấy output của version cụ thể.
- Notebook đã dùng: `kaggle_chunk_corpus`, `kaggle_encode_queries_4b`, `kaggle_search_4b` (không chạy được do lỗi đọc vector, đã thay bằng tìm kiếm local), `kaggle_bm25_vi`, `kaggle_rerank_benchmark`, `kaggle_package_rerankers`.

### 7. Quy tắc làm việc

- Việc nặng (lập chỉ mục, BM25, tìm kiếm toàn bộ vector, rerank) chạy trên Kaggle, không chạy local. Lý do: 25/10 BM25 local bị OOM hai lần.
- Không nộp leaderboard khi chưa có bằng chứng bài mới khác tập chunk/doc. Hỏi bạn trước khi dùng lượt nộp.
- Không rerank hay chạy GPU nếu chưa có kế hoạch cụ thể.
- Không dùng VPN, xoay IP, hay vượt chặn nguồn (AGENT.md, memory `respect-site-blocks`).

### 8. Hướng tiếp theo (cập nhật 29/10)

Mục cũ (chọn chunk từ pool top-500, gộp điểm doc, trọng số BM25) đã làm trong notebook hybrid. Thứ tự đề xuất mới:

1. **Kiểm tra output các kernel 8B** (`medxir-embed-corpus-{qwen8b,octen8b}-s0..s2`, `medxir-encode-queries-{qwen8b,octen8b}`, `medxir-embed-icd-octen8b`): đã xong chưa, tên file vector, số chiều (dự kiến 4096), có file ids đi kèm không.
2. **Notebook ensemble:** search 4B + Qwen3-8B + Octen-8B, gộp RRF, rerank pool, ghép chunk kiểu nb60 trong notebook. Ba bộ vector ~55 GB fp16, vừa GPU 97 GB.
3. **expand_chunks kết hợp** (chỉ CPU): nguyên văn top-3..5 doc, phần dung lượng còn lại theo nb60.
4. **d300/d400:** danh sách doc gần như không tốn dung lượng.
5. **Ghép chunk có chọn lọc:** rerank cả chunk liền kề, chỉ ghép chunk điểm cao.
6. **Probe tỷ lệ ZH/EN:** ép ≥ ~20% chunk ZH/EN.
7. **Embed tiêu đề doc VI/ZH** (bỏ hậu tố tên trang như "| Sức Khỏe Cộng Đồng"; EN đã có tiêu đề ở đầu văn bản). Cột `title` trong `Kaggle-Group/archive.zip` → `docs/<lang>/part-v1-*.parquet`.
8. ICD (Octen): để sau. Cân nhắc tập dev ~100 query nhãn LLM để thử offline.

**Phân tích chunk recall (30/10 tối).** (1) Bản wholedoc nộp trọn ~34 doc đầu bảng: Chunk P 0,21 ≈ tỉ lệ doc đúng trong top 34 (BM25 Baseline của BTC: ~35 doc, P 0,88). (2) Mỗi doc đúng chứa ~1 đoạn đáp án (~58 doc đúng, ~50 đoạn/query), nên Chunk R tăng theo số doc đúng được chạm tới; nb60 chạm 83 doc/query, định vị đúng đoạn ở ~75% doc đúng chạm tới. (3) Dưới giới hạn 100 MB, Chunk R gần như bão hòa ~0,22–0,25 (nb30/nb60/x1-c70) → cần thêm byte. (4) Cùng JSON bản 0,2477: ZIP deflate 86,5 MB, **ZIP bzip2 56,5 MB**, ZIP LZMA 54,4 MB → nén bzip2 nhét được ~1,55× văn bản. Biến thể `nb60-drr-bge-only-ghigh-bz` (bzip2, budget_scale 1,6). Chưa biết dashboard có đọc ZIP bzip2 không: file thử `R2/submission/medxir-expand-nb60-drr-bge-only-ghigh-BZIP2TEST-submission.zip` (cùng nội dung 0,2477).

**Nghi vấn chưa tách (30/10), thử sau.** Bản gộp `ghigh1-min200-s20` (0,2450) thua `ghigh` (0,2477) vì Chunk R giảm 0,234 → 0,2195. Ba bước gộp chung nên chưa biết bước nào làm mất recall. Nghi theo thứ tự: (a) γ −1 lấy quá nhiều chỗ của query yếu; (b) `min_score` 0,2 bỏ cả chunk đúng; (c) `min_chars` 200 bỏ đoạn ZH ngắn mà đáp án ZH của BTC cũng ngắn. Cách tách: nộp riêng `R2/submission/medxir-expand-nb60-drr-bge-only-ghigh-min200-submission.zip` (chỉ min200) và `nball-drr-bge-only-ghigh-min200` (cụm 3 chunk, không cắt chunk). Ở P ≈ 0,17, R ≈ 0,22, một đơn vị R đáng ≈ 2,6 đơn vị P trong F2: chỉ giữ bước cắt nào không làm giảm R.

### 9. Bài nộp hiện có (`artifacts/submission/` và `R2/submission/`)

| File | Đã nộp | Điểm | Ghi chú |
|---|---|---:|---|
| **v2ens-bgefull-drr-dp03bge-llm10-d200-c131-x1-lz** (`artifacts/submission/`) | Có | **0,2984** | **Mốc tốt nhất** (31/10) |
| v2ens-bge35-drr-dp05bge-llm10-d200-c131-x1-lz | Có | 0,2981 | bge 0,35 + doc-prior 0,5 |
| v2ens-bgefull-drr-dp03-llm05-d200-c131-x1-lz | Có | 0,2976 | LLM 0,5, cả trong doc-prior |
| v2ens-bgefull-drr-dp03-d200-c131-x1-lz | Có | 0,2965 | ZIP LZMA |
| v2ens-bgefull-drr-d200-c120-x1-bz | Có | 0,2916 | ZIP bzip2 |
| v2ens-bgefull-drr-dp03bge-llmavg10-d200-c131-x1-lz | Không nộp | | Tập 200 doc trùng 100% với 0,2984 |
| v2ens-bge25-drr-d200-c76-x1 (`artifacts/submission/`) | Có | 0,2755 | Hai hướng ghép, corpus v2 |
| v2ens-bge25-d200-c76-x1 (`artifacts/submission/`) | Chưa nộp | | Như trên, chưa bge doc rerank |
| v2 fusion 4B · Qwen3-8B · Octen-8B nb60-drr-bge-only (`R2/submission/fusion-35-35-30/`) | Có | 0,2539 | Mốc tốt nhất hướng A |
| v2 Qwen3-8B hybrid nb60-drr-bge-only (`R2/submission/qwen8b/`) | Có | 0,2509 | |
| v2 nb60-d200-drr-bge-only | Có | 0,2464 | |
| **var-all-b1-d200-c70-x1** (`artifacts/submission/`) | Có | 0,2444 | Mốc tốt nhất hướng B |
| ens4-prfoct-b1-d200-c70-x1 | Có | 0,2429 | |
| ens4-b1-d200-c70-x1 | Có | 0,2410 | |
| only8b-var-b1-d200-c70-x1 | Chưa nộp | | Ablation: bỏ 4B và 0.6B |
| var-all-b1-d200-c50-x2 | Chưa nộp | | Ablation: cửa sổ 5 chunk, 50 cửa sổ |
| ens4-icd02-b1-d200-c70-x1, q8-oct-bm25-b1-d200-c70-x1 | Chưa nộp | | |
| Các bài hướng B khác từ 27/10 | Có, đã xóa file | | Điểm ở docs/EXPERIMENTS.md; dựng lại bằng `fuse_scores.py` |
| **v2 nb60-d200** (`R2/submission/v2/`) | Có | 0,2437 | |
| expand nb60 (`R2/submission/chunk concat/medxir-expand-nb60-submission.zip`) | Có | 0,2235 | |
| expand nb30 / wholedoc | Có | 0,2184 / 0,2065 | Cùng thư mục |
| expand nb0 / nb10 | Chưa nộp | | |
| hybrid bật hết (`R2/submission/full-config/`) | Có | 0,1965 | |
| hybrid tắt rerank + BM25 (`R2/submission/no-rerank-bm25/`) | Có | 0,1759 | |
| probe VI (`R2/submission/probe-vi-chunks/`) | Có | 0,1944 | |
| d200-c150 (`medxir-qwen3-4b-d200-c150-submission.zip`) | Có | 0,1847 | Mốc dense cũ |
| d200-c100 | Có | 0,1783 | |
| d200 / d100 / d50 / d300 / d150 (`d150` chưa nộp) | Có (trừ d150) | | Dense |
| rrf-d100, rrf-d200, bm25only-d200 | Có | 0,1377 / 0,1433 / 0,0982 | |
| rrf03-d200 | Chưa nộp | | Trọng số BM25 0,3 |
| bm25only-d100 | Chưa nộp | | |
| d200-c50 | Chưa nộp | | |
| rr06b / rr4b / rrbge / rr4b-f05 / rr4b-f10 (d200-c150) | Không nộp | | Trùng tập với dense, không có giá trị |

Ghi chú: c200 và c300 không nộp được vì vượt 100 MB.

---

## Nhật ký chi tiết

Tài liệu này ghi kết quả đã thu được. Các bước thiết kế chưa triển khai được đánh dấu riêng; hướng giải quyết tổng thể nằm trong [PROPOSED_SOLUTION.md](PROPOSED_SOLUTION.md).

## Rà soát bổ sung — 15/10/2026

Đã kiểm tra rclone trực tiếp: kho MedXIR chỉ còn ba thư mục rỗng `vi/`, `en/`, `zh/`; người dùng xác nhận teammate đã xóa dữ liệu để crawl lại. Các batch verified và kết quả crawl dưới đây là lịch sử, không phản ánh dữ liệu còn trên Drive. Local full shard 0 đã xử lý 321.391 URL ngày 12/10, nên mô tả “chưa chạy full” ở phần lịch sử đã lỗi thời.

Corpus hiện hành có 1.220.598 ID / 94 host, hash khớp Hugging Face, khác số 1.194.469 trong thông báo. Pilot mới: 266/275 URL đã xử lý, 240 có text, 9 pending do pause; chưa đủ căn cứ kết luận mọi nội dung đã đầy đủ. Đã sửa nhận diện challenge/blocked NCBI trong crawler, 8 test offline đạt. Chưa triển khai parser PubMed, chưa crawl full lại hoặc upload.

Báo cáo, danh sách host, giới hạn kiểm thử và đề xuất triển khai: [AUDIT_2026-10-15.md](docs/AUDIT_2026-10-15.md).

## Triển khai crawl lại và hàng đợi sửa lỗi — 15/10/2026

- `prepare_recrawl.py` tạo đủ ba plan, không loại ID theo lịch sử; hợp bằng corpus, ID/URL duy nhất, hai shard modulo 2 không trùng. Mapping host/ngôn ngữ và policy được lưu trong `configs/`.
- Tổng VI/EN/ZH: 740.598 / 240.000 / 240.000. Ban đầu mở 703.983 VI + 107.633 ZH = **811.616 ID**; **408.982 ID hoãn**, vẫn nằm trong plan và follow-up.
- Đã ghim ba storage config vào folder ID con hiện có, chuẩn bị cloud cho cả hai shard và chia sẻ manifest/plan/config trong `MedXIR/coordination/recrawl_20261015/`.
- `followup.py` xuất snapshot SQLite nhất quán: CSV tất cả ID cần xử lý, plan retry tải và plan ID reparse; có lý do, action, raw locator, checksum, thống kê theo host. Cập nhật mỗi 300 giây và khi kết thúc, upload rồi đọc lại xác minh. Có báo cáo ban đầu cho cả shard 1 dù chưa crawl.
- Parser v2.3 bổ sung review bài text ngắn và bài ít text/nhiều ảnh. Reparse chọn lọc ba ID 64406/89705/89714 từ raw cũ đã giữ nguyên text và chuyển đúng sang needs_review.
- `--reparse-plan` chọn chính xác ID/URL để reparse, không tải lại website; `--max-new` giới hạn đợt kiểm chứng; runner chia tổng 16 worker thành 8 VI + 8 ZH/máy.
- 12 kiểm tra offline đạt; đã thử thực tế prepare, upload/checksum và reparse chọn lọc. Đợt đầu: 600 URL đã xử lý, 593 có text (447 extracted_unreviewed, 146 needs_review), 7 không có text. Bốn batch đã upload/đọc lại khớp SHA-256. Đã khôi phục raw từ một batch mỗi ngôn ngữ VI/ZH, kiểm hash nguồn và parse lại khớp text đã upload; bằng chứng ở `artifacts/recrawl_20261015/verification.json`.
- Shard 0 chạy dài với 8 worker VI + 8 worker ZH, 1/nhóm nguồn, delay hiệu dụng 4 giây/máy; batch 2.000 kết quả hoặc 128 MiB raw gzip, prune chỉ sau xác minh. Shard 1 đã khởi tạo báo cáo nhưng chưa crawl, dành cho teammate. PID/log hiện hành ở `artifacts/recrawl_20261015/active_run.json` và `processes-shard-0.json`.

Lệnh hiện hành và chi tiết xử lý lỗi: [RECRAWL_RUNBOOK.md](docs/RECRAWL_RUNBOOK.md). Checkpoint SQLite vẫn giữ local; giữ nguyên plan/shard/config khi resume. Độ đầy đủ nội dung và retrieval chưa được benchmark; extracted_unreviewed không đồng nghĩa đã duyệt sạch.

## 1. Đọc đề bài và khảo sát dữ liệu — đã thực hiện

Đã đọc tài liệu về nhiệm vụ, dữ liệu, đánh giá và định dạng nộp bài. Nhiệm vụ trả về tài liệu và chunk liên quan cho câu hỏi tiếng Việt; đánh giá riêng hai cấp bằng Precision, Recall và F2 theo macro-average. Chưa có nhãn liên quan cục bộ để benchmark retrieval.

Kết quả khảo sát hai file Parquet:

| Hạng mục | Kết quả |
|---|---:|
| Truy vấn | 1.200 |
| Bản ghi corpus / URL không trùng | 4.394.718 |
| Host | 97 |
| ID tài liệu nhỏ nhất / lớn nhất | 583 / 4.420.561 |

ID tài liệu có khoảng trống. Số bản ghi thực tế khác con số 4.420.561 trong tài liệu dữ liệu; pipeline phải dựa vào các ID có trong Parquet.

| Host lớn nhất | Số URL |
|---|---:|
| `www.cnkang.com` | 963.438 |
| `www.120ask.com` | 918.479 |
| `www.familydoctor.com.cn` | 447.453 |
| `ask.39.net` | 337.375 |

Bốn host chiếm khoảng 60,68% corpus. Vì các host còn lại vẫn chiếm gần 40%, pilot được mở rộng cho toàn bộ danh sách host.

## 2. Chốt hướng giải pháp V1 — đã viết thiết kế

Đã ghi giải pháp trong `PROPOSED_SOLUTION.md`: ưu tiên thu thập, làm sạch và chunking; baseline ban đầu dùng một sparse retriever và một dense retriever. Chưa chọn checkpoint cuối cùng, chưa triển khai hoặc đo chất lượng retrieval.

Đã thống nhất nguyên tắc lưu raw cùng ID/URL/trạng thái trước khi parse để có thể sửa parser mà không tải lại. Crawler hiện đã hỗ trợ chia theo `doc_id % shard_count` từ cùng kế hoạch. Chưa có bộ điều phối tốc độ tập trung giữa các máy; mỗi máy tự giảm tốc độ theo số shard.

## 3. Audit host và template — đã chạy pilot

Notebook `notebooks/host_template_audit.ipynb` lấy mẫu theo host/đường dẫn, với mục tiêu khoảng 20–50 URL tùy nguồn và cấu hình. Không phải host nào cũng có đủ mẫu hoặc truy cập được.

Manifest hiện tại có **1.820 bản ghi, bao phủ danh sách 97 host**:

| Trạng thái nguồn | Số bản ghi |
|---|---:|
| HTML response | 1.581 |
| Bỏ qua host (`skipped_host`) | 126 |
| HTTP error | 73 |
| Access challenge | 20 |
| Chuyển về trang chủ | 14 |
| Lỗi tải | 3 |
| Response rỗng | 2 |
| Không phải HTML | 1 |
| **Tổng** | **1.820** |

Các phát hiện chính đã được ghi trong [docs/HOST_TEMPLATES.md](docs/HOST_TEMPLATES.md):

- Có bài kiến thức, hỏi–đáp một/nhiều/không có answer, hồ sơ bệnh, chương sách, danh sách, trang ảnh/media/PDF và quiz.
- `www.familydoctor.com.cn` gặp HTTP 403 trong pilot. `www.familydoctor.cn` là host khác, chưa có cơ sở dùng làm nguồn thay thế.
- HTTP 200 có thể chỉ chứa challenge hoặc body rỗng; redirect có thể đưa về trang chủ.
- Một số trang dùng GBK/GB2312; có payload mã hóa, section trả phí hoặc nội dung chỉ xuất hiện qua tab/link khác.
- Box answer vẫn có quảng cáo, thông tin bác sĩ và UI. Nội dung hữu ích cũng có thể nằm ngoài đoạn tổng quan, trong bảng, list, caption hoặc text trực tiếp.

## 4. Dọn cấu trúc repo — đã thực hiện

Đã bỏ tiền tố số ở tên notebook, bỏ `.python-version` và quản lý thư viện bằng uv với `package = false`. Code audit nằm trong notebook. Artifacts audit được rút còn manifest, tổng hợp host và raw.

Parser hiện nằm ở `scripts/content_parser.py` để notebook review và crawler tương lai dùng chung. Notebook review tập trung chạy, kiểm tra và đối chiếu parser. Không cần một bản `content_parser.py` ở thư mục gốc.

## 5. Parser và review — đã triển khai, đã chạy

Đã triển khai parser theo selector của các host khảo sát, xử lý encoding và giữ cấu trúc nội dung. Parser lưu `doc_id`, URL gốc/cuối, dấu vết raw, title, block theo thứ tự, câu hỏi/answer, metadata, tài nguyên và phần bị loại kèm lý do.

Lần chạy notebook hiện tại dùng raw đã tải, không gửi request mới. Kết quả trong `artifacts/parser_review/documents.jsonl` và `review.html`:

| Kết quả | Số bản ghi |
|---|---:|
| `extracted_unreviewed` | 1.393 |
| `needs_review` | 168 |
| `encoded_content` | 19 |
| Nhận diện trang chủ riêng | 1 |
| Các trạng thái nguồn còn lại | 239 |
| **Tổng** | **1.820** |
| Có text trích xuất | **1.559, thuộc 83 host** |
| Lỗi parser / template chưa nhận diện trong pilot | **0 / 0** |
| Cảnh báo mất/lặp chữ khi đọc DOM vùng đã chọn | **0** |

Số mẫu có text khác tổng hai trạng thái trích xuất vì một số trang có media hoặc placeholder nhưng không có text hữu ích.

### Những trường hợp đã kiểm chứng

- Hỏi–đáp: giữ đúng question/answer, bỏ box quảng cáo và chuyên môn bác sĩ khỏi body answer; câu hỏi thiếu answer được ghi nhận riêng.
- Câu hỏi bổ sung: chỉ bỏ đoạn lặp hoàn toàn; giữ tuổi, thời gian và diễn tiến nếu có thêm thông tin.
- Bách khoa: giữ triệu chứng, xét nghiệm, biến chứng, section và tài liệu tham khảo.
- Chương sách: giữ nội dung thật, loại điều hướng và lời mời tải sách.
- Bài báo: loại tin liên quan; giữ bảng dữ liệu và caption.
- Media: giữ URL tài nguyên, bỏ text fallback trình duyệt, không khẳng định đã có transcript.
- Section trả phí: giữ phần truy cập được và ghi phần thiếu, không đưa thông báo trả phí vào text retrieval.

Các kiểm tra hồi quy trong notebook đã qua. Một số ID đối chiếu: `1298551`, `2291691` (Q&A), `1029368` (audio), `2516803` (JBK), `3160915` (Baidu), `893827` (chương sách), `764166` (YouMed), `2547549` (trả phí).

### Ý nghĩa và giới hạn

`extracted_unreviewed` chỉ nghĩa là chưa có cảnh báo tự động, không phải đã duyệt sạch. `needs_review` có thể là nội dung trích đúng nhưng chưa đầy đủ, không nhất thiết parser sai.

Kiểm tra bảo toàn chữ chỉ xác nhận các vùng đã chọn sau khi cắt UI được đọc đúng; không chứng minh mọi nội dung hữu ích ngoài selector đã được lấy. Báo cáo hiển thị heading và đoạn dài ngoài vùng chọn, notebook cho xem text toàn trang để hỗ trợ đối chiếu.

Kết quả pilot chưa thể suy rộng thành tỷ lệ thành công cho 4,4 triệu URL. Không có số đo Precision/Recall/F2 của retrieval ở thời điểm này.

## 6. Các nhóm còn tồn đọng

| Nhóm | Số mẫu | Phần cần xử lý |
|---|---:|---|
| Wiki | 40 | Duyệt phần đuôi theo loại trang, kiểm tra nhiễu trong vùng nội dung |
| JB39 | 20 | Section hiện tại là bản xem trước; chưa tải phần chi tiết |
| JBK | 17 | Chưa tải tab/link chi tiết |
| Section trả phí | 19 | Thiếu phần không truy cập được |
| Payload mã hóa | 19 | Chưa có decoder tạo được text bài |
| Quiz | 5 | Chưa có câu hỏi trong HTML server |
| Media chưa xác minh transcript | 20 | Text có thể chưa phản ánh nội dung audio/video |
| Ảnh chưa chuyển thành text | 9 | Có thể thiếu thông tin trong ảnh |
| PDF chưa parse | 5 | Chưa lấy text tài liệu đính kèm |
| Nội dung nhúng chưa parse | 10 | Cần kiểm tra tài nguyên nhúng |

Các cảnh báo có thể chồng lấp trên cùng bản ghi. Ngoài bảng còn có trang danh sách, chỉ mục chương, placeholder, câu hỏi không có answer và marker chưa render; chi tiết nằm trong báo cáo.

## 7. Crawler và đợt mở rộng — đã triển khai, đã hoàn tất

Đã thêm `scripts/crawl.py`, dùng lại parser hiện có, không thêm thư viện ngoài `pyproject.toml`.

- Lập kế hoạch chung trước khi chia shard; giữ nguyên ID và loại URL đã có trong pilot.
- Lưu response nén gzip trước khi parse; checkpoint từng URL bằng SQLite.
- Chạy tiếp các URL chưa hoàn thành; khóa shard để tránh hai tiến trình cục bộ cùng ghi.
- Tối đa 12 request đồng thời, tuần tự theo hostname với khoảng cách mặc định 1,5 giây. Redirect/retry cũng qua giới hạn.
- Timeout, giới hạn kích thước response, retry có giới hạn, xử lý `Retry-After` và tạm dừng host sau ba lỗi truy cập liên tiếp.
- Xuất manifest, tài liệu, summary và báo cáo review mẫu; có chế độ chỉ export hoặc parse lại raw mà không tải mới.
- Giảm tốc độ mỗi máy theo số shard; chưa bảo đảm giới hạn tổng nghiêm ngặt giữa các máy.

Đã tạo **10.000 URL mới trên 80 host** tại `artifacts/crawl/plan.jsonl`. Ba host `fk.99.com.cn`, `mega.vietnamplus.vn`, `special.vietnamplus.vn` có HTML pilot nhưng không còn URL mới trong corpus. Các host không có HTML pilot chưa được đưa vào đợt này; danh sách loại được in khi lập kế hoạch.

Lấy tối đa 20 URL ban đầu mỗi host, phần còn lại phân bổ theo căn bậc hai số URL, xen kẽ nhóm thành phần đầu đường dẫn. Phân bổ này phục vụ kiểm chứng độ đa dạng, không đại diện tỷ trọng corpus và chưa xác nhận mọi template.

Snapshot đã xuất trước khi chạy tiếp:

| Kết quả đợt mở rộng | Số bản ghi |
|---|---:|
| Đã lưu checkpoint | 1.187 |
| Có text | 1.152 |
| `extracted_unreviewed` | 1.019 |
| `needs_review` | 133 |
| `encoded_content` | 21 |
| `unknown_template` | 3 |
| HTTP error | 4 |
| Chuyển về trang chủ | 5 |
| Không phải HTML | 2 |
| Chưa hoàn thành tại snapshot | 8.813 |

Chưa ghi nhận `parser_error` trong snapshot. Ba trang chưa khớp template là `713530` (infographic VOV2), `856747` (People Health), `2052399` (chỉ mục y án ZYDCD). Được giữ nguyên raw và trạng thái để duyệt; chưa kết luận nguyên nhân hoặc độ đầy đủ.

Đã xuất kết quả checkpoint rồi khởi chạy tiếp cùng kế hoạch. Sau đó đã dừng tiến trình nền theo yêu cầu người dùng, xuất đầy đủ kết quả đã lưu. Người dùng tiếp tục chạy trực tiếp và hoàn tất 10.000 URL. Các file JSONL/review là snapshot lần xuất gần nhất, không cập nhật theo từng URL; summary được cập nhật theo `--progress-every` (mặc định 25 URL) và khi kết thúc.

Checkpoint khi dừng: **6.891/10.000 URL (68,9%)**, còn **3.109 URL**; **6.757 bản ghi có text**, gồm 5.871 `extracted_unreviewed`, 887 `needs_review`; có 24 `unknown_template` và không ghi nhận `parser_error`. Raw, SQLite, JSONL và báo cáo mẫu đã được giữ. Không còn tiến trình crawler nền do agent khởi chạy.

Đã bổ sung `--resume` (yêu cầu checkpoint có sẵn), `--status` (đọc tiến độ trực tiếp, không tải), `--progress-every` (tần suất in tiến độ). Chế độ shard dùng `--shard-count`/`--shard-id`; `--status` và `--resume` áp dụng theo shard được chọn. Đợt hiện tại giữ cấu hình một shard để tiếp tục đúng checkpoint.

## 8. Kiểm tra chất lượng đợt 10.000 — đã thực hiện

Kết quả cuối cùng được xuất lúc **21:49 ngày 10/10/2026 (UTC+07:00)**. Nguồn kiểm tra: `artifacts/crawl/shard-000-of-001/{pages.jsonl,documents.jsonl,state.sqlite,raw/}`. Không tải thêm URL và không sửa/parse lại toàn bộ kết quả trong vòng review này.

### Kết quả định lượng

| Kết quả | Số bản ghi | Tỷ lệ trên 10.000 |
|---|---:|---:|
| Đã hoàn thành checkpoint | 10.000 | 100% |
| HTML response | 9.840 | 98,40% |
| Có text | 9.737 | 97,37% |
| `extracted_unreviewed` | 8.316 | 83,16% |
| `needs_review` | 1.423 | 14,23% |
| `unknown_template` | 55 | 0,55% |
| `encoded_content` | 46 | 0,46% |
| HTTP error | 65 | 0,65% |
| Chuyển về trang chủ | 55 | 0,55% |
| Response rỗng | 32 | 0,32% |
| Không phải HTML | 5 | 0,05% |
| Access challenge | 2 | 0,02% |
| Fetch error | 1 | 0,01% |
| `parser_error` | 0 | 0% |

Hai bản ghi có media nhưng không có text khiến tổng hai trạng thái trích xuất là 9.739, khác 9.737 bản ghi có text. Các nhóm cảnh báo có thể chồng lấp.

### Phạm vi kiểm tra và kết quả đạt được

- Đối chiếu toàn bộ 10.000 ID/URL/host với plan, và ID/URL với Parquet chính thức: không có sai lệch; manifest và tài liệu đều đủ 10.000 bản ghi.
- Giải nén và kiểm tra SHA-256, độ dài của toàn bộ **9.999 raw response** có đường dẫn: không có lỗi file hoặc sai hash. Một fetch error không có raw.
- Đường dẫn raw và hash nguồn trong tài liệu đều khớp manifest.
- Đọc thống kê/cảnh báo trên toàn bộ tài liệu; parse lại **160 mẫu, hai mẫu mỗi host**, từ raw với parser hiện tại: title, text, blocks, answers, warnings và status khớp kết quả lưu. Điều này xác nhận khả năng tái lập, không chứng minh mọi mẫu sạch.
- Đối chiếu raw các nhóm bất thường: title rỗng, unknown template, bảng bị lệch chữ, phân trang, HTML còn trong text và khối gợi ý/quảng cáo.
- Cập nhật file `artifacts/crawl/shard-000-of-001/review.html` hiện có với tập mẫu theo host và các dấu hiệu lỗi: tất cả unknown/title rỗng/phân trang đã phát hiện, ca đã xác nhận và hai mẫu mỗi host. Chỉ cập nhật báo cáo hiển thị, giữ nguyên raw, checkpoint và `documents.jsonl`.
- Các phần `answers[].text` đều khớp các block tương ứng. Có 1.286 trang Q&A; 53 mẫu không có answer được cảnh báo riêng. Nội dung ngắn không tự động bị loại.
- Có 366 tài liệu với text dưới 120 ký tự, nhiều mẫu là câu trả lời/chương sách/ngắn hoặc caption hợp lệ; không coi toàn bộ là lỗi.

### Vấn đề đã xác nhận

| Vấn đề | Bằng chứng / phạm vi | Ý nghĩa |
|---|---|---|
| Title rỗng nhưng nguồn có title | **86 mẫu**: 61 Baidu, 25 People Health; cả 86 đều có title hoặc heading không rỗng trong raw | `title_of` dừng ở node rỗng, bỏ qua fallback. Các mẫu hiện không được cảnh báo lỗi này. |
| Bảng bị đọc lặp | **1 mẫu**, SKĐS `493463`; khi đối chiếu vùng DOM đã làm sạch có thêm 416 ký tự sau khi bỏ khoảng trắng/separator | Lỗi đọc cấu trúc bảng lồng/malformed; cảnh báo `block_text_mismatch` đã bắt được. Đây là lặp chữ, không phải mất 416 ký tự. |
| Bài phân trang chưa lấy đủ | **32 mẫu** có text “下一页” và link trang tiếp theo trong raw: Wujue 26, CNKang 3, ba host 39.net mỗi host 1 | Crawler chỉ tải URL được cấp, chưa nối các trang tiếp theo. Text còn pager, chưa có cảnh báo thiếu phân trang. |
| Nhiễu publisher lặp lại | **1.563 tài liệu** còn câu bản quyền 39.net trong text | Rule hiện chưa phủ các vị trí DOM chứa boilerplate; cần tách khỏi text retrieval, giữ provenance. |
| Gợi ý bài còn trong body | `911500` (wei.39.net: “相关推荐”), `834086` (fk.39.net: “编辑推荐”) | Cả hai đang là `extracted_unreviewed`; wrapper đúng vẫn có thể chứa khối gợi ý. |
| Khối quảng bá còn trong bài | `174333`, `174369` (MEDLATEC Store); `746491` (Vinmec hotline); `760445` (YouMed chat/app) | Cần rule theo cấu trúc/section. Không xóa toàn bộ đoạn chỉ vì có từ app/hotline. |
| HTML còn ở dạng text | **11 ứng viên** có dấu hiệu tag trong text: 9 Q&A 120ask, 1 Woman39, 1 Bệnh viện Nhi; đã xác nhận ví dụ `1212271`, `923749` | Có escaped markup từ nguồn chưa được xử lý; ảnh trong markup có thể chưa thành resource. Cần phân biệt với văn bản thật. |
| Wiki còn điều hướng | **402 tài liệu** chứa nhãn điều hướng wiki; toàn bộ 681 tài liệu wiki đã có `wiki_tail_requires_review` | Khối đuôi chưa sạch, cần phân loại cả trang file/attachment và sách/bệnh/thuốc. |

Quét từ khóa còn phát hiện 62 ứng viên có câu đặt lịch/hotline (44 ở MEDLATEC), nhưng đây chỉ là danh sách cần đối chiếu, không phải 62 lỗi đã xác nhận. Mẫu `1313` của Bạch Mai thực sự hướng dẫn dùng ứng dụng khám bệnh; loại theo từ khóa sẽ làm mất nội dung hữu ích. Tương tự, “Đăng nhập” trong `760515` là chữ nằm ở phần trích nguồn, không phải menu đăng nhập.

### Phân biệt template mới với vấn đề của nguồn

55 `unknown_template` thuộc 14 host; CNKang chiếm 30. Đối chiếu raw xác nhận ít nhất **10 mẫu** là lỗi/thiếu nội dung nguồn, không đơn thuần thiếu selector:

- Năm CNKang trả `Fatal error` dù HTTP 200: `1728056`, `1728067`, `1728069`, `3485054`, `3485056`.
- Một 39.net chuyển sang `/web/404.html`: `812601`.
- Một Vinmec báo không tìm thấy dữ liệu: `742721`.
- Ba trang có vùng thân bài rỗng: Quảng Ninh `82817`, PMPHAI `2541307`, `2541601`.

**45 mẫu còn lại cần phân loại/template bổ sung**, có cả bài cũ, longform, media, infographic và trang danh sách; không khẳng định cả 45 là bài text lấy được đầy đủ. Ví dụ Dân trí có `#desktop-cate-in-article`, VietNamNet longform có `.inner-contetn`; ZYDCD `2052399` là chỉ mục y án, không phải một y án đơn.

CNKang có 59 HTTP 404 và 32 response rỗng trong 813 URL mẫu. Toàn bộ 65 HTTP error của đợt đều là 404, không phải 403/429. Vì vậy không nên tăng retry cho nhóm này để mong cải thiện độ phủ.

### Nội dung chưa đầy đủ đã được cảnh báo

- Wiki: 681; JB39 preview: 90; JBK chưa tải tab chi tiết: 132.
- Section trả phí: 101; nội dung mã hóa: 46; quiz chưa có câu hỏi trong HTML: 3.
- Ảnh chưa chuyển thành text: 63; media chưa xác minh transcript: 83; PDF chưa parse: 17; nội dung nhúng chưa parse: 120.

52 ứng viên có đoạn dài ngoài selector đều thuộc ask.39.net, food.39.net, gc.39.net và wei.39.net. Nội dung quan sát được là hồ sơ bệnh/triệu chứng hoặc chuyên môn bác sĩ ở sidebar; đây là bằng chứng việc bỏ ngoài selector có thể đúng, không nên gom lại chỉ vì đoạn dài. Heuristic này không bao phủ mọi cách bỏ sót nội dung.

Có ba nhóm hai ID có text trùng hoàn toàn, gồm alias thuốc hoặc hai URL của cùng nội dung. Chưa loại ID nào; deduplicate nội dung cần giữ mapping tới toàn bộ ID chính thức. Hai mẫu có ký tự replacement trong text cũng cần đối chiếu dữ liệu nguồn, dù decoder không ghi cảnh báo fallback/replacement.

### Kết luận chất lượng và thứ tự sửa theo ROI

Lớp tải/lưu dữ liệu hoạt động ổn trên tập đã chọn; raw đủ để sửa parser mà không crawl lại. **Chưa đủ cơ sở gọi 9.737 bản ghi có text là corpus sạch**, và `extracted_unreviewed` vẫn có false negative của hệ thống cảnh báo.

1. Sửa fallback title, đọc bảng, boilerplate và khối gợi ý/quảng bá đã xác nhận; thêm cảnh báo title rỗng, escaped markup và phân trang. Chi phí thấp, có bằng chứng cụ thể và phạm vi ảnh hưởng lớn.
2. Phân loại 55 unknown; bổ sung template cho các nhóm có nội dung, tách lỗi nguồn/empty body khỏi unknown. Giữ raw, không cần tải lại để sửa selector.
3. Thiết kế theo link phân trang có quan hệ rõ với cùng bài, lưu phần bổ sung gắn `doc_id` gốc; chưa triển khai trong vòng review này.
4. Parse lại raw sau khi sửa, duyệt lại mẫu lỗi và mẫu bình thường; sau đó mới quyết định mở rộng crawl/chunk.

Chưa sửa parser, chưa thay đổi kết quả crawl và chưa chạy thêm crawl trong vòng kiểm tra chất lượng này.

## 9. Hạ tầng lưu trữ V1 — thiết kế ban đầu OneDrive, 11/10/2026

Đã cài rclone v1.75.2 tại `~/.local/bin/rclone` trên máy hiện tại, thêm `scripts/storage.py` và tích hợp upload batch vào crawler. Teammate cần tự cài rclone; Python vẫn dùng các dependency hiện có.

### Cơ chế đã cài đặt

- Cấu hình chung `storage.json` chứa remote/thư mục và `drive_id`. Mỗi máy đăng nhập rclone riêng; drive sai bị từ chối trước upload. Chưa tạo file này vì chưa đăng nhập OneDrive.
- Đường dẫn cloud theo SHA-256 plan, shard và UUID batch; không ghi đè batch giữa các máy. Phân công vẫn dùng ID chính thức modulo số shard.
- Đóng batch theo 10.000 document hoặc 256 MiB raw gzip. Giữ nguyên `.gz` trong `raw.tar`; lưu đủ pages/document/audit dưới JSONL gzip và manifest checksum.
- Upload payload, đọc lại byte để kiểm SHA-256, rồi xuất manifest làm dấu hoàn tất. Batch chưa xác minh được giữ local và thử lại khi resume.
- `--prune-local` chỉ dọn batch đã xác minh; SQLite giữ trạng thái/locator và document rút gọn. SQLite tái sử dụng phần trống; muốn giảm kích thước file ngay cần VACUUM khi đã dừng crawler. Các JSONL xuất cũ vẫn giữ nguyên.
- `--cache-limit-gb` đặt mốc dừng cho raw/batch local; có thể vượt tạm thời trong khi đóng batch/request đang chạy. Không áp dụng cho SQLite/file export và không phải hạn ngạch đĩa cứng.
- Có lệnh đọc document theo shard, khôi phục một raw và xuất document của tất cả shard từ OneDrive mà không cần SQLite trên máy indexing. Xuất remote chọn batch mới nhất cho từng ID, cần đồng bộ đồng hồ các máy.
- Crawler thêm `--host-workers`, cho phép tải nhiều response trong cùng hostname nhưng vẫn giới hạn khoảng cách bắt đầu request, retry và redirect. Mặc định cũ 12 tổng/1 host/delay 1,5 giây được giữ.

### Kết quả thao tác thực tế

Đã chạy đóng batch trên checkpoint 10.000 mẫu. Trước khi đóng, kiểm SHA-256 raw giải nén với digest nguồn; không có lỗi khiến thao tác dừng.

| Thành phần | Dung lượng byte |
|---|---:|
| `raw.tar` | 265.584.640 |
| `pages.jsonl.gz` | 891.887 |
| `documents.jsonl.gz` | 24.127.288 |
| Tổng payload | 290.603.815 (~277,1 MiB) |

Batch `d41e657764064425a66a2cbc5fed88cf`, plan SHA-256 `f3db74ca32b0e3e32c059d70accaedca5ccabb01b223024ad17e6692d3fa1464`, shard `000-of-001`. Trạng thái: **1 batch pending, 0 document chưa đóng**. Checkpoint vẫn đủ 10.000 kết quả; raw gốc chưa xóa, chưa compact SQLite và chưa đổi nội dung parser.

Đã kiểm tra cú pháp Python và đọc trạng thái checkpoint/batch. **Chưa xác minh upload/download OneDrive thực tế**, vì cấu hình rclone hiện chưa có remote; cần người dùng đăng nhập Microsoft. Chưa đo tốc độ crawler sau thay đổi song song; chưa chạy thêm crawl hoặc khởi động crawler nền.

Cấu hình khởi điểm đề xuất: 32 worker tổng, 2/host, delay 1,5 giây. Tăng tiếp 48–64 tổng/2–4 host tùy timeout, 429/5xx và tài nguyên máy; chưa có số worker tối đa đã đo. Giữ nguyên delay 1,5 giây thì riêng 963.438 URL CNKang có cận dưới khoảng 16,7 ngày, bất kể worker tổng. Các host bị chặn vẫn cần xử lý riêng.

Hướng dẫn thao tác trong [README.md](README.md#google-drive-lưu-cuốn-chiếu-và-dùng-nhiều-máy). Chưa tự backup/khôi phục SQLite/plan từ cloud; giữ các file checkpoint local.

## 10. Chuyển sang Google Drive — 11/10/2026

Người dùng đổi đích lưu trữ sang Google Drive và đã đăng nhập rclone remote `gdrive` (backend `drive`). Đã cập nhật `scripts/storage.py`, CLI crawler và README để dùng đích này; dữ liệu/batch đã đóng được tái sử dụng.

- Đã tạo thư mục `MedXIR`, ID `1tfvL8OJIRejT_c-4wfaQnTZecT67qJld`, và `storage.json` chứa remote, backend, folder ID và Shared Drive ID nếu có. File không chứa token/mật khẩu; có thể chia sẻ nguyên file cho các máy.
- Mọi thao tác cloud cố định `root_folder_id` theo ID thư mục trên. Hai tài khoản có thư mục trùng tên không dẫn upload sang hai nơi. Tài khoản teammate khác phải có quyền sửa vào cùng thư mục.
- Thêm `storage.py prepare --plan PLAN --shard-count N` để một máy tạo các thư mục plan/shard chung trước khi các máy bắt đầu. Google Drive cho phép tên thư mục trùng; bước chuẩn bị tuần tự tránh việc hai máy tạo các thư mục cha trùng nhau. Crawler yêu cầu thư mục shard đã được chuẩn bị.
- Đã chuẩn bị layout cho plan hiện tại, một shard; chưa thay đổi số shard/checkpoint và chưa khởi động crawl mới.

### Upload thực tế đã hoàn tất

Batch `d41e657764064425a66a2cbc5fed88cf` chứa đủ 10.000 bản ghi đã upload vào:

```text
MedXIR/runs/f3db74ca32b0e3e32c059d70accaedca5ccabb01b223024ad17e6692d3fa1464/
  shard-000-of-001/batches/d41e657764064425a66a2cbc5fed88cf/
```

Ba payload tổng 290.603.815 byte (~277,1 MiB) và manifest đều được đọc lại qua rclone và **khớp SHA-256** với bản local. Checkpoint ghi batch **verified**. Lần này chạy `upload` không có `--prune-local`: raw, batch local, full document trong SQLite và các bản xuất cũ vẫn giữ nguyên. Chưa chạy VACUUM.

Remote báo quota **429.496.729.600 byte (400 GiB)** và **398.653.978.817 byte (~371,3 GiB) trống** trước upload; đây là quota Google Drive thực tế, không phải 5 TB OneDrive đã đề cập trước đó. Kích thước toàn corpus cần tiếp tục đo khi phân bố host thay đổi.

Đã kiểm cú pháp Python và đọc trạng thái local. Xác minh lần này là upload/download byte thực tế trong cơ chế lưu trữ; chưa đo crawler với số worker mới, chưa chạy crawl nền và chưa sửa parser. Hướng dẫn hiện hành ở [README.md](README.md#google-drive-lưu-cuốn-chiếu-và-dùng-nhiều-máy); thư mục cloud [MedXIR](https://drive.google.com/drive/folders/1tfvL8OJIRejT_c-4wfaQnTZecT67qJld).

## 11. Parser v2.1 và hướng dẫn CLI — 11/10/2026

Đã sửa `scripts/content_parser.py` dựa trên các trường hợp trong raw của đợt 10.000 URL, giữ ID và URL chính thức. Không tải thêm URL nguồn.

### Các thay đổi đã cài đặt

- Bổ sung fallback cho Dân trí, VietNamNet longform, CNKang bài cũ/recipe/listing, VOV/VOV2, Care39, chuyên đề Man39, ZYDCD và video SKĐS. Giữ sapo/heading/body theo thứ tự DOM.
- Title fallback tiếp tục tìm khi node hoặc meta rỗng, thay vì kết thúc với title rỗng.
- Đọc bảng theo DOM để tránh lặp text ở hàng/cell lồng hoặc thiếu thẻ đóng; giữ cấu trúc hàng cho bảng dữ liệu. Vẫn đối chiếu text vùng đã chọn với các block và cảnh báo nếu không khớp.
- Cleanup thông báo publisher/editor, các khối link gợi ý, store card MEDLATEC, booking CTA Vinmec và app promotion YouMed theo cấu trúc đã xác nhận. Không áp dụng xóa hàng loạt bằng từ khóa; hướng dẫn ứng dụng trong bài Bach Mai được giữ. `removed` lưu text/link/lý do của phần loại.
- Chuyển escaped HTML có tag được nhận diện thành DOM, giữ caption/heading/tài nguyên; lưu bản trước chuyển trong `normalizations` và cảnh báo.
- Tách `source_error` cho fatal error, soft 404 và Vinmec báo không tìm thấy dữ liệu; `empty_content` cho vùng bài thật sự rỗng; `unknown_template` dành cho trang chưa tìm được vùng nội dung.
- Ghi `pagination_links` và cảnh báo phần tiếp theo chưa tải. Loại nút điều hướng khỏi text; không nhận link chuyển chương sách/tab chi tiết khác ID là phân trang cùng bài. Chưa tải trang tiếp theo.
- Giữ URL video từ widget/data attribute, URL ảnh/audio/PDF cùng caption nếu có. Gallery/infographic được ghi thiếu text trong ảnh, không coi mô tả ngắn là đã lấy đủ nội dung.

### Thao tác và tài liệu

Crawler có `--parse-workers` để reparse bằng process CPU, `--force-reparse` để áp dụng lại cùng version, và tiến độ reparse trực tiếp. Mặc định reparse bỏ qua document đã cùng version để tiếp tục sau khi dừng. `--status` và summary ghi phân bố parser version. Worker reparse tách riêng `--workers`/`--host-workers` dùng cho HTTP.

README đã ghi lệnh parser độc lập, crawl, prepare, resume, status, reparse thường/ép, export, chia shard, thay plan/output, cloud và toàn bộ tùy chọn crawler/parser. Danh mục template được cập nhật các selector/cleanup đã cài đặt.

### Kết quả parse lại toàn bộ 10.000 raw

Lượt cuối dùng `--reparse --force-reparse --parse-workers 4`, hoàn tất khoảng **300,8 giây** trên máy hiện tại. Sau đó xuất lại JSONL/summary và báo cáo mẫu 250; số mẫu HTML thực tế bị giới hạn bởi các nhóm host/trạng thái. Toàn bộ 10.000 document có `parser_version = parser-v2.1`, không gửi request mới tới website nguồn. Đã kiểm cú pháp Python; không thêm thư viện hoặc bộ test mới.

| Chỉ tiêu | Trước sửa (`pilot-v1`) | Sau sửa (`parser-v2.1`) |
|---|---:|---:|
| Bản ghi có text | 9.737 | **9.775** |
| `extracted_unreviewed` | 8.316 | 8.173 |
| `needs_review` | 1.423 | 1.611 |
| `unknown_template` | 55 | **0** |
| `source_error` | Chưa tách | 7 |
| `empty_content` | Chưa tách | 3 |
| `encoded_content` | 46 | 46 |
| `parser_error` | 0 | 0 |
| `block_text_mismatch` | 1 | **0** |
| Title rỗng ở bản trích/needs_review | 86 | **0** |

55 unknown cũ được phân thành **45 bản trích** (31 không có cảnh báo, 14 cần duyệt), **7 lỗi nguồn** và **3 vùng nội dung rỗng**. Có thêm 38 bản ghi chứa text; phần còn lại gồm trang media chỉ có tài nguyên. Không tính trang lỗi thành tài liệu đã lấy nội dung.

- Bảng SKĐS `493463` giữ 6 hàng, không còn lặp chữ; 6 recipe CNKang giữ hàng/cell và không có mismatch. Hai bài CNKang chỉ có text/br `3506847`, `3507150` được khôi phục.
- Mẫu Dân trí, VietNamNet longform, Care39, Man39, VOV/VOV2 và ZYDCD mới đã có vùng trích phù hợp. Infographic `713530` chỉ có mô tả và ảnh nên vẫn `needs_review`; gallery `923749` giữ caption/ảnh và dấu vết chuyển escaped markup.
- 1.563 tài liệu trước đó chứa cụm publisher “未经书面授权”; còn **1** (`855409`) do chung đoạn với lời nhắc phương thuốc lấy từ mạng và cần thận trọng. Rule xóa nguyên thông báo không xóa đoạn có thêm thông tin này. `834086`, `911500` đã loại khối related/publisher; `174333`, `174369`, `746491`, `760445` đã xử lý các CTA/store/app đã xác nhận.
- 5 title còn thiếu thuộc các trang PHP fatal error, không thuộc bản trích nội dung.

`needs_review` tăng vì nhận diện thêm media/gallery/infographic, pagination và chuẩn hóa escaped HTML, cùng các template mới. Số bản ghi không cảnh báo giảm không đồng nghĩa mất dữ liệu; số liệu này cũng không chứng minh mọi nội dung đã sạch.

**Còn thiếu:** 31 tài liệu có phân trang chưa lấy phần tiếp; 46 payload mã hóa; cảnh báo 73 trang có text trong ảnh chưa chuyển, 195 trang media chưa xác minh transcript, 17 PDF và 148 nội dung nhúng chưa parse. Ngoài ra còn 101 trang trả phí, 90 preview JB39, 132 tổng quan JBK thiếu tab chi tiết, 681 wiki cần duyệt đuôi và 53 QA không có answer trong HTML. Các nhóm chồng lấp. Không còn unknown trong 10.000 mẫu không bảo đảm mọi template của 4,4 triệu URL đã được bao phủ.

Kết quả hiện tại ở `artifacts/crawl/shard-000-of-001/documents.jsonl`, `summary.json` và `review.html`. Pilot/báo cáo cũ vẫn là kết quả lịch sử cho tới khi tự chạy lại notebook; raw gốc được giữ để xử lý tiếp.

### Xuất bản kết quả mới lên Google Drive

Đã chạy `storage.py upload` sau reparse, tạo batch **`4ea9a3ff320e4132bb26600c88640726`** chứa 10.000 bản ghi v2.1, cùng plan/shard và thư mục Drive ở mục 10. Ba payload và manifest đã được đọc lại, khớp SHA-256; checkpoint ghi `verified = 1` và mọi document trỏ tới batch mới.

| Thành phần | Dung lượng byte |
|---|---:|
| `raw.tar` | 265.584.640 |
| `pages.jsonl.gz` | 891.887 |
| `documents.jsonl.gz` | 25.430.030 |
| Tổng payload | 291.906.557 (~278,4 MiB) |

Batch v1 cũ vẫn giữ trên cloud/local để đối chiếu, cùng các ID với batch mới; không phải 10.000 tài liệu mới. `export-remote` chọn bản trong batch mới hơn. Raw vẫn được đóng lại trong batch mới theo cơ chế V1, nên cloud tăng thêm khoảng 278,4 MiB; chưa tách riêng kho raw để dùng chung qua nhiều phiên bản parser. Không dùng `--prune-local`, chưa compact document hoặc VACUUM, không khởi động crawler tải URL nền.

## 12. Parser v2.2 và khả năng chạy full — 11/10/2026

### Parser và nguồn mới

- Giải mã payload CMS Phú Thọ/Thanh Hóa theo JavaScript công khai của CMS (alphabet/rotate/Base64); không chạy JavaScript tùy ý. Kiểm tra cấu trúc HTML sau decode, giữ payload gốc và đường dẫn nguồn. **46/46 mẫu cũ và 19/19 mẫu pilot đã giải mã**; không coi chuỗi mã hóa là text.
- Long Châu đã truy cập lại được: lấy trường đúng bài từ `__NEXT_DATA__`, kiểm tra slug, giữ mô tả, triệu chứng, nguyên nhân, nguy cơ, chẩn đoán/điều trị, sinh hoạt/phòng ngừa, FAQ và tham khảo. Loại danh sách sản phẩm/bài/bệnh gợi ý. Sáu hồ sơ bệnh có phần hữu ích nằm ngoài nội dung HTML ban đầu; đã giữ đủ các trường hiện có và 5 FAQ mỗi hồ sơ.
- Việt Đức: hai bố cục `.news-primary-content .edittor-content` và `.news-primary-content`; giữ nội dung và tác giả trong bài, loại chia sẻ/tag/gợi ý. Selector đầu tiên quá hẹp gây 3 unknown, gate đã tạm dừng đúng; bổ sung fallback rồi parse lại và hoàn tất phần còn lại.
- Reparse toàn bộ pilot và cập nhật báo cáo HTML local bằng v2.2. Không còn unknown, encoded hay mismatch trong pilot; số có text là 1.578/1.820 (gồm trạng thái fetch/skip cũ).

### Kết quả đo được

| Đợt | Hoàn thành | Có text | Unknown / encoded / parser error / mismatch |
|---|---:|---:|---:|
| 10.000 mẫu cũ, reparse raw | 10.000 | 9.821 | 0 / 0 / 0 / 0 |
| 1.000 URL mới, không trùng mẫu cũ | 1.000 | 964 | 0 / 0 / 0 / 0 |
| Long Châu + Việt Đức, 20 URL mỗi host | 40 | 40 | 0 / 0 / 0 / 0 |

Reparse 10.000 mẫu mất khoảng 226,3 giây với 4 process. 1.611 bản cần duyệt; không đồng nghĩa 9.821 bản đều đủ nội dung hoặc hoàn toàn sạch.

**179 mẫu cũ không có text:** 65 HTTP 404, 55 redirect về trang chủ, 32 response rỗng, 7 lỗi nội dung nguồn, 5 non-HTML, 3 vùng body rỗng, 2 challenge, 1 lỗi fetch và 9 trang media chỉ có tài nguyên. Không có trường hợp còn mất text do decoder trong 46 mẫu đã sửa.

**36 mẫu mới không có text:** 28 CAPTCHA/challenge, 3 HTTP 404, 3 redirect trang chủ, 1 non-HTML và 1 media. Các challenge đã hoàn thành với trạng thái lỗi được giữ lại; không báo thành công hoặc tự retry chúng. Sau cooldown, 5 URL pending thuộc nhóm nguồn tạm dừng đã tải được với giới hạn mới.

### Crawler và lưu trữ

- Thêm giới hạn chung theo nhóm subdomain 39.net/familydoctor/iiyi, áp dụng cả retry/redirect. Chờ semaphore/delay không bị tính là timeout HTTP.
- Pause nguồn khi lỗi truy cập liên tiếp hoặc parser unknown/error/encoded/mismatch vượt gate; giữ pause trong SQLite qua resume. `--retry-paused` chỉ mở lại URL pending; còn pending thì kết thúc exit code 2.
- Thêm `--all-hosts`, `--include-host`, `--exclude-completed`; skipped pilot chưa fetch không bị loại khỏi full plan. Checkpoint nhập theo batch và cờ `plan_loaded` tránh nhập lại hàng triệu dòng khi resume.
- Cache checksum/tar theo batch khi đọc raw cloud; kiểm tra digest raw local. Chỉ dọn batch đã upload và xác minh.
- **7 kiểm tra offline đã đạt**, gồm shard ID không liên tục, loại ID đã làm, pause/resume, giới hạn subdomain, hàng đợi/timeout, bảng lỗi HTML, payload CMS sai, trường ẩn Long Châu và pack/verify/prune/restore raw.
- Batch v2.2 **`eabafa6fc7d347649b547038bd5e1420`** của 10.000 ID đã upload và đọc lại xác minh SHA-256; giữ local. Payload gồm raw.tar 265.584.640 byte, pages.jsonl.gz 891.887 byte và documents.jsonl.gz 25.573.468 byte (tổng 292.049.995 byte). Ba batch cloud là ba phiên bản của cùng ID, không cộng thành 30.000 tài liệu. Hai đợt 1.000/40 mẫu mới còn local.

### Phạm vi full và kết luận

Reprobe 14 nguồn cũ: Long Châu/Việt Đức mở lại, 12 nguồn còn lỗi/chặn. **85 host với 3.632.818 URL** đã có bằng chứng HTML nội dung; **12 host với 761.900 URL** chưa truy cập được. Nhóm 39.net vẫn có CAPTCHA theo thời điểm.

Đã tạo `artifacts/crawl_full/plan.jsonl`, **4.381.984 URL**, loại hợp 12.734 ID đã thử/hoàn thành từ các đợt trước. Plan bao gồm mọi host còn URL (90 host); toàn corpus có 97 host. Đã đối chiếu plan: 4.381.984 ID duy nhất, mọi ID/URL khớp corpus; không thiếu ID ngoài hợp loại trừ và không trùng ID đã loại. Chia modulo 2: shard 0 có 2.190.903 URL, shard 1 có 2.191.081 URL. Khởi tạo checkpoint shard 0 mất 15,69 giây, peak RSS 71.932 KiB; mở lại/export không nhập lại plan mất 0,99 giây. Đã chuẩn bị thư mục Drive cho hai shard. Không khởi chạy full crawl. Lệnh chính thức cho hai máy, resume/status, storage và giới hạn nguồn nằm trong [README.md](README.md#chạy-toàn-bộ-corpus).

**Kết luận:** đủ khả năng bắt đầu full crawl có kiểm soát và giữ nguồn để sửa parser sau. Chưa thể xác nhận sẽ lấy đủ nội dung 4,4 triệu URL. Cảnh báo phân trang (31 mẫu cũ), media/PDF/ảnh, QA thiếu đáp án và vùng preview/paywall còn tồn tại; không tự biến chúng thành bản hoàn chỉnh cho indexing. Theo delay 2 giây, riêng CNKang có cận dưới khoảng 22,3 ngày; đây không phải cam kết thời gian hoàn thành.

## 13. Kế hoạch tại thời điểm 11/10/2026

1. Chạy full plan trực tiếp theo shard; theo dõi batch đầu, nguồn pause, tỷ lệ text/cảnh báo và dung lượng theo host.
2. Xử lý host bị chặn khi có phương án truy cập thực sự; mở lại pending sau khi xác minh, tránh retry liên tục.
3. Duyệt cảnh báo thiếu nội dung, bổ sung phân trang/media theo ROI trước khi chunk các nhóm đó.
4. Hợp kết quả các đợt theo doc_id, thiết kế chunking từ block/QA và triển khai sparse + dense baseline.

**Chưa hoàn thành:** crawl toàn bộ, chunking, indexing, retrieval, reranking, benchmark và bài nộp.

## 14. Chunking, embedding và đóng gói vector — 23/10/2026

### Chunking corpus trên Kaggle

Kernel `boinhbo/medxir-chunk-corpus` hoàn tất. [Manifest đã tải về](artifacts/kaggle/chunk-corpus-v1/chunk_manifest.json) ghi corpus v1, `clean-v1`, tham số chunk `350/50/40`, 118 ID bị loại theo cấu hình và 1.023.783 hash group.

| Ngôn ngữ | Chunk |
|---|---:|
| EN | 250.487 |
| VI | 2.128.601 |
| ZH | 242.465 |
| **Tổng** | **2.621.553** |

### Embedding toàn bộ corpus

Hai model đã chạy trên cùng 2.621.553 chunk / 987.297.945 token, gồm đủ 9 part: `en-part-000`, `vi-part-000..006`, `zh-part-000`.

| Model | Trạng thái | Kích thước vector | NaN | Artifact |
|---|---|---:|---:|---|
| Qwen3-Embedding-0.6B | Hoàn tất 9/9 part; người dùng chạy interactive session và upload thủ công | 1.024 chiều | 0 | [Dataset Kaggle](https://www.kaggle.com/datasets/boinhbo/medxir-vectors-qwen3-0-6b), gồm manifest, 9 `.npy` và 9 parquet `chunk_id` |
| Qwen3-Embedding-4B | Ba shard Kaggle `s0/s1/s2` đều hoàn tất 3/3 part | 2.560 chiều | 0 | Ba dataset và archive liệt kê bên dưới |

Các embedding được tính với BF16; vector `.npy` lưu dạng float16. Tổng chunk và token trên manifest hai model khớp nhau. Các shard 4B đã kiểm shape, dtype và số hàng ID parquet so với manifest trước khi upload.

| Shard | Part | Dataset | Archive cục bộ |
|---|---|---|---|
| s0 | EN-000, VI-002, VI-005 | [medxir-vectors-qwen3-4b-s0](https://www.kaggle.com/datasets/boinhbo/medxir-vectors-qwen3-4b-s0) | `artifacts/kaggle/medxir-vectors-qwen3-4b-s0.zip` |
| s1 | VI-000, VI-003, VI-006 | [medxir-vectors-qwen3-4b-s1](https://www.kaggle.com/datasets/boinhbo/medxir-vectors-qwen3-4b-s1) | `artifacts/kaggle/medxir-vectors-qwen3-4b-s1.zip` |
| s2 | VI-001, VI-004, ZH-000 | [medxir-vectors-qwen3-4b-s2](https://www.kaggle.com/datasets/boinhbo/medxir-vectors-qwen3-4b-s2) | `artifacts/kaggle/medxir-vectors-qwen3-4b-s2.zip` |

Mỗi dataset 4B có `embed_manifest.json` và ba cặp `vectors/<part>.npy` / `vectors/<part>-ids.parquet`, tổng cộng 7 file theo Kaggle file listing. Các dataset là private của `boinhbo`.

### Bước tiếp theo

1. Đợi lần chạy retrieval 4B trên Kaggle (`boinhbo/medxir-retrieve-qwen3-4b`, version 4) hoàn tất; kiểm tra `retrieval_manifest.json` và bài nộp 1.200 query.
2. Nộp baseline 4B lên leaderboard, rồi ghi điểm vào mục này.
3. Sau baseline, so sánh 0.6B nếu thời gian cho phép. BM25 vẫn hoãn tới khi chốt và đánh giá tokenizer riêng EN/VI/ZH.

Query retrieval, benchmark chất lượng và submission chưa hoàn tất. [Lịch Road to AI 2026](https://r2ai.aiguru.com.vn/) ghi hạn Vòng 1 là 30/10/2026; kế hoạch embedding cũ đặt mốc nội bộ khoảng 26/10.

## 15. Nối dữ liệu và FAISS Qwen3-Embedding-4B — 23/10/2026

### Đã rà soát và chốt data contract

Tải đủ chunk manifest, `hash_groups.parquet` và cả chín chunk parquet từ output của kernel Kaggle `boinhbo/medxir-chunk-corpus` vào `artifacts/kaggle/chunk-corpus-v1/` để đối chiếu thực tế.

- Với từng part, `vectors/<part>.npy[row]`, `<part>-ids.parquet.chunk_id[row]` và `chunks/<lang>/part-NNN.parquet.chunk_id[row]` khớp chính xác theo thứ tự hàng.
- Đã kiểm tra đủ 2.621.553 hàng, 2.621.553 `chunk_id` duy nhất; không có ID vector thiếu hoặc lặp giữa các part. ID khớp công thức `text_sha256[:16] + "-" + order`; đủ 1.023.783 prefix hash, không collision và mọi thứ tự chunk liên tục.
- Mỗi `(text_sha256, rep_doc_id)` khớp `hash_groups.parquet`. Lúc truy hồi: hàng vector → `<part>-ids.parquet` → `chunk_id` → chunk gốc để lấy text; `text_sha256` nối hash group. Chỉ fan-out sang nhiều doc ID khi `fanout_safe=true`; nếu không, dùng representative ID. (Mục 16 thay FAISS row-map bằng cách này.)
- Hợp đồng đầy đủ và quy tắc nối nằm trong [docs/RETRIEVAL_DATA_CONTRACT.md](docs/RETRIEVAL_DATA_CONTRACT.md).

### Index 4B đã tạo

Thêm [scripts/build_faiss_flat_index.py](scripts/build_faiss_flat_index.py) và dependency tùy chọn `retrieval` (`faiss-cpu`). Script kiểm tra manifest, dtype/shape/vector hữu hạn, so sánh row order vector-ID-chunk, xác minh hash group rồi ghi index cùng row-map cho từng part.

- Đã build đủ 9 `IndexFlatIP`, 2.621.553 hàng, dimension 2.560, metric inner product; FAISS 1.15.1. Mỗi index part độc lập, có thể search top-k riêng rồi merge chính xác thành top-k toàn corpus.
- Index float32 chiếm 26.844.703.125 byte (~25,0 GiB) tổng. Build theo part để vừa RAM; mỗi file khoảng 2,1–3,4 GB.
- Nguồn `.npy` giữ float16. Khi build, từng batch được nạp thành float32 và chuẩn hóa lại; index lưu float32 để làm baseline chính xác. Chưa thử so sánh ranking thực tế với GPU float16 trên query cuộc thi.
- Index và row-map ở `artifacts/faiss/qwen3-embedding-4b/` (nằm trong `.gitignore`). Đã upload lên Kaggle dataset `boinhbo/medxir-faiss-qwen3-4b-fp32` ngày 23/10 (khoảng 25 GB). Từ 24/10 retrieval không còn dùng index này; sẽ xóa dataset sau khi lần chạy retrieval 24/10 hoàn tất.

Kiểm tra đầu ra đã xác nhận 9/9 index đọc được, dimension/metric/count đúng, row-map khớp file ID và truy vấn self-search trên vectors nguồn trả similarity 1,0. Chưa tạo query embedding, chưa chạy retrieval trên 1.200 query, chưa sinh submission hoặc benchmark chất lượng.

## 16. Retrieval Qwen3-Embedding-4B trên Kaggle — 24/10/2026

**Trạng thái:** lần chạy đầu tiên của `boinhbo/medxir-retrieve-qwen3-4b` (version 4, RTX Pro 6000, Internet tắt) đang chạy. Chưa có điểm và chưa có bài nộp đã kiểm tra.

**Lỗi trước đó:** version 1 đến 3 dừng ở ô cuối với `ValueError`. Nguyên nhân: `doc_ids` đọc từ parquet là mảng numpy, còn ô ghép bài nộp dùng `if not doc_ids:`, trên mảng nhiều phần tử thì Python báo lỗi. Đã đổi sang list Python và kiểm tra logic bằng pandas trên máy local. Log trên Kaggle rỗng trong các lần đó nên không đọc được lỗi từ phía Kaggle; lỗi chỉ thấy khi chạy interactive.

**Thiết kế hiện tại:** không dùng FAISS nữa.
- Mã hóa 1.200 query bằng Qwen3-Embedding-4B, theo đúng tiền tố `Instruct: … Query:` và pooling token cuối của model card, chuẩn hóa L2.
- Nạp từng part vector (`vectors/<part>.npy`, float16) từ ba dataset `medxir-vectors-qwen3-4b-s0..s2` lên GPU, tính tích vô hướng bằng float32, lấy top 20 mỗi part, rồi gộp thành top 20 toàn corpus. Đây là tìm kiếm chính xác, tương đương `IndexFlatIP`.
- Map hàng vector → `chunk_id` qua `<part>-ids.parquet`, rồi lấy chunk gốc từ `chunks/<lang>/part-NNN.parquet` và nhóm trùng từ `hash_groups.parquet`. Chỉ fan-out doc ID khi `fanout_safe=true`.
- `TOP_CHUNKS = 20` là giá trị đặt tạm, chưa tune.

**Lý do đổi khỏi FAISS:** index FAISS chỉ là bản sao của cùng vector float16. Runtime FAISS không có sẵn trên image RTX của Kaggle và gây lỗi import trong quá trình debug. Tìm bằng PyTorch cho cùng phép tính.

**Dọn dẹp sau khi lần chạy xong:** xóa `boinhbo/medxir-faiss-qwen3-4b-fp32` và `boinhbo/medxir-retrieval-runtime` trên Kaggle. Bản local `artifacts/faiss/` đã xóa ngày 25/10 để giải phóng 26 GB; dựng lại bằng `scripts/build_faiss_flat_index.py` từ `artifacts/kaggle/vectors-qwen3-4b-s*`.

**Cập nhật 25/10:** lần chạy encode trên Kaggle (`medxir-encode-queries-4b`) hoàn tất ở 495 giây, RSS đỉnh 5,7 GiB. Kernel tìm kiếm trên Kaggle (`medxir-retrieve-qwen3-4b`) chết ngay khi đọc file vector, không có log; probe `medxir-vector-read-probe` đọc đầy đủ một file vector cũng chết. Kết luận tạm: đọc file vector từ dataset mount trên Kaggle là điểm lỗi. Tìm kiếm chuyển sang máy local bằng `scripts/retrieve_local_4b.py` (CPU, float32, 141 giây). Bài nộp `artifacts/submission/medxir-qwen3-4b-submission.json` đủ 1.200 query, mỗi query 20 chunk, mọi `chunk_text` là nguyên văn. Đã nộp lên leaderboard, kết quả baseline 4B (chưa tune, `TOP_CHUNKS = 20`):

| Chỉ số | Giá trị |
|---|---:|
| FINAL_SCORE | **0,0818** |
| Document F2 (macro) | 0,1214 |
| Document Precision / Recall | 0,3238 / 0,1174 |
| Chunk F2 (macro) | 0,0422 |
| Chunk Precision / Recall | 0,0978 / 0,0436 |

So với bài #1 (BM25, FINAL_SCORE 0,019): khoảng gấp 4 lần. Recall document chỉ 0,12 nên phần lớn tài liệu đúng chưa được tìm thấy; precision chunk thấp hơn precision document nên nhiều chunk nộp không khớp đoạn tham chiếu.

### Tách danh sách doc khỏi danh sách chunk, thêm BM25 — 25/10/2026

`relevant_docs` và `relevant_chunks` được chấm độc lập, nên `scripts/retrieve_local_4b.py` giờ lưu top 500 chunk mỗi query (`artifacts/submission/candidates-4b-k500.npz`), xếp doc theo điểm chunk tốt nhất, rồi xuất N doc và C chunk tùy chọn (`--doc-counts`, `--chunk-counts`). Mọi bản bên dưới dùng cùng vector 4B; FINAL = trung bình Doc F2 và Chunk F2.

| Bản nộp | Doc P / R | Chunk P / R | Doc F2 | Chunk F2 | FINAL |
|---|---|---|---:|---:|---:|
| Baseline (≈17 doc, 20 chunk) | 0,324 / 0,117 | 0,098 / 0,044 | 0,121 | 0,0422 | 0,0818 |
| d50, 20 chunk | 0,240 / 0,242 | 0,098 / 0,044 | 0,207 | 0,0422 | 0,1245 |
| d100, 20 chunk | 0,186 / 0,339 | 0,098 / 0,044 | 0,248 | 0,0422 | 0,1453 |
| d200, 20 chunk | 0,134 / 0,445 | 0,098 / 0,044 | 0,261 | 0,0422 | 0,1517 |
| d300, 20 chunk | 0,108 / 0,508 | 0,098 / 0,044 | 0,254 | 0,0422 | 0,1480 |
| **d200, 100 chunk** | 0,134 / 0,445 | 0,074 / 0,134 | 0,261 | 0,0954 | **0,1783** |
| RRF dense+BM25 (1:1) d100 / d200 | 0,174 / 0,325; 0,125 / 0,430 | 0,089 / 0,042 | 0,235; 0,247 | 0,0401 | 0,1377; 0,1433 |
| Chỉ BM25 tiếng Việt, d200 | 0,086 / 0,299 | 0,055 / 0,025 | 0,172 | 0,0248 | 0,0982 |

Nhận xét: số doc tối ưu quanh 200; tăng số chunk làm Chunk F2 tăng mạnh (precision chỉ giảm 0,098 → 0,074 khi chunk tăng 5 lần, recall tăng 3 lần), nên còn dư địa với 200–300 chunk. RRF 1:1 với BM25 làm điểm giảm; BM25 đứng riêng đạt 0,098 (chỉ phủ chunk tiếng Việt). Số chunk tham chiếu mỗi query ước tính khoảng 50, suy từ recall/precision.

BM25 (`scripts/bm25_vi.py`, chạy trên Kaggle CPU bằng `configs/kaggle/bm25_vi.yaml`): âm tiết và cặp âm tiết liền kề, chuẩn hóa NFKC và vị trí dấu thanh, bỏ stopword và câu xã giao ở phía query, IDF/độ dài toàn corpus. Chạy BM25 trên máy local bị hết RAM hai lần; việc nặng chạy trên Kaggle (AGENT.md).

**Mã nguồn:** [scripts/build_retrieval_notebook.py](scripts/build_retrieval_notebook.py), [src/notebooks/kaggle_retrieve_qwen3_4b.ipynb](src/notebooks/kaggle_retrieve_qwen3_4b.ipynb), [configs/kaggle/retrieve_qwen3_4b_rtx6000.yaml](configs/kaggle/retrieve_qwen3_4b_rtx6000.yaml).

## Dọn dẹp artifacts và kế hoạch corpus v1.1 — 25/10/2026

**Đã làm:**
- Xóa `artifacts/crawl_full/` (plan 4,38 triệu URL cũ, không còn trong corpus).
- Xóa `artifacts/crawl/` (pilot 11/10). Trước khi xóa, đối chiếu với `dataset/links_corpus_vi_en_zh.parquet`: 10.000 tài liệu gồm 5.344 ID không có trong parquet, 4.372 ID đã có trong `corpus_v1` (4.353 giống hệt văn bản), 284 ID có trong parquet nhưng chưa có trong v1. Trong 284 ID này, 76 thuộc 120ask (nguồn bị chặn, loại), còn 208 dùng được, trong đó 140 có văn bản. 140 tài liệu đó được trích vào [artifacts/corpus_v1_patch/crawl_gap_v1_1.jsonl](artifacts/corpus_v1_patch/crawl_gap_v1_1.jsonl) (đã kiểm tra: id khớp parquet, url khớp 100%). Trạng thái của cả 284 ID nằm ở [crawl_gap_status.csv](artifacts/corpus_v1_patch/crawl_gap_status.csv).
- Đã xóa `.work/` (debug/smoke) và `artifacts/readiness_check/`.

**Quyết định (25/10):**
- Không upload `cc_v1` và `pharmacity_v1` lên Drive: ước tính hơn 10 giờ với tốc độ ~7,8 MB/phút, và chưa có việc cần dùng ngay. Đã xóa bản `cc_v1` upload dở trên Drive.
- `wayback_v1` (9 MB) đã upload lên `gdrive:coordination/recrawl_20261015/local_only/wayback_v1` và kiểm tra khớp (435 file, 0 khác biệt).
- Giữ bản local của `cc_v1` (4,7 GB) và `pharmacity_v1` (772 MB). Đây là bản duy nhất của raw; ổ đĩa còn trống 922 GB nên không cần xóa.
- Không reparse ZH và không làm v1.1 bây giờ. Corpus v1 giữ nguyên làm nền cho thí nghiệm reranker. Tác động của ZH parser v2.3 chưa đo được; làm sau khi có kết quả reranker.
- 140 tài liệu trong patch chưa nhập vào corpus, vì nhập sẽ kéo theo chunk và embed mới. File patch giữ lại để làm sau.

**Kế hoạch corpus v1.1 (hoãn, chỉ làm nếu còn thời gian sau reranker):**
1. Thêm 140 tài liệu trong `crawl_gap_v1_1.jsonl`.
2. Thêm tài liệu VI từ 7 batch upload sau khi build v1 (4.753 ID chưa có trong v1, 4.733 có văn bản). Cần loại nguồn bị chặn trước khi nhập.
3. Reparse ZH từ parser v2.3 lên v2.4 trở lên (khoảng 72k tài liệu, raw ZH có trên Drive). Cần đo số văn bản thực sự đổi trước.
4. Chunk và embed phần đổi: chunk ID là `text_sha256[:16]-order`, nên văn bản không đổi thì giữ vector cũ.
5. Đăng ký v1.1 là một version mới của dataset Kaggle, giữ v1 để tái lập.
Pending 26.179 ID (suckhoedoisong, longchau theo quota) không đưa vào v1.1 vì cần crawl lại và đang bị chặn.

### Giới hạn kích thước bài nộp — 25/10/2026

Dashboard từ chối file trên 100 MB. Bản d200 với 200 chunk/query nặng 121 MB (đã nén mức 9, JSON gọn), nên không nộp được. Chunk text phải nguyên văn nên không rút gọn được nội dung. Chọn 150 chunk/query: 90 MB, nằm trong giới hạn.

Ghi chú: file `medxir-qwen3-4b-d200-c100-submission.zip` được tạo lại bằng nén mức 9 nên khác byte với bản đã nộp (59,7 MB thay vì 60,2 MB). Nội dung chunk và doc giống hệt.

| Bản | Kích thước | Ghi chú |
|---|---:|---|
| d200-c100 | 59,7 MB | Đã nộp, FINAL 0,1783 |
| **d200-c150** | 90,0 MB | **Đã nộp, FINAL 0,1847 (tốt nhất hiện tại)** |
| d200-c200 | 121 MB | Vượt giới hạn, không nộp được |

**Kết quả d200-c150 (đã nộp):** FINAL 0,1847; Doc F2 0,2612; Chunk F2 0,1081 (P 0,0665, R 0,1696).

Xu hướng Chunk F2 theo số chunk/query (cùng 200 doc): 20 → 0,042; 100 → 0,095; 150 → 0,108. Mỗi 50 chunk thêm tăng ít dần (+0,013 ở 100→150), trong khi precision giảm. Chunk tiếp theo bị giới hạn 100 MB chặn lại. Các cải thiện tiếp theo cần đến từ thứ tự xếp hạng (reranker, hybrid), không chỉ từ số lượng.

### Benchmark reranker đa ngôn ngữ — 26/10/2026 (chưa hoàn tất)

Gói model (`boinhbo/medxir-package-rerankers`, exp_0002, 12,8 GB): Qwen3-Reranker-0.6B (Apache-2.0, 2025-05-29), Qwen3-Reranker-4B (Apache-2.0, 2025-06-03), bge-reranker-v2-m3 (Apache-2.0, 2024-03-15), jina-reranker-v3.5 (CC-BY-NC-4.0, 2026-07-14, không thương mại; cần xác nhận quy định cuộc thi).

Chạy benchmark (exp_0004, 100 query = 15.000 cặp, top-150 dense):
- Phần cứng thực tế là **Tesla T4**, không phải RTX Pro 6000. Config yêu cầu `NvidiaRtxPro6000` nhưng CLI không áp dụng được; notebook không kiểm tra GPU nên không báo lỗi.
- Qwen3-Reranker-0.6B: chạy được, nhưng 2,1 cặp/giây (bf16 không có trên T4). Với 180.000 cặp của 1.200 query là khoảng 24 giờ.
- bge-reranker-v2-m3: chạy được, 4,4 cặp/giây, VRAM 3,5 GB.
- Qwen3-Reranker-4B: hết VRAM trên T4 (14,5 GB).
- jina-reranker-v3.5 (listwise, 150 tài liệu một ngữ cảnh): hết VRAM, cần kiến trúc chia nhỏ hoặc GPU lớn hơn.

Đo đồng thuận (không phải chất lượng, vì chưa có nhãn): hai model cho xếp hạng tương quan Spearman trung bình 0,51 và trùng 8,6/20 chunk đầu.

Cần làm: chạy lại trên RTX Pro 6000 chọn trong giao diện Kaggle (không qua CLI); dùng fp16 khi GPU không hỗ trợ bf16; thêm kiểm tra tên GPU trong notebook; chỉ dùng được Qwen3-4B và bge trên GPU lớn hơn.

### Rerank toàn bộ 1.200 query trên RTX Pro 6000 — 26/10/2026

Chạy `exp_0006` (NVIDIA RTX PRO 6000 Blackwell, bf16, Internet OFF, có cuộc thi ARC AGI 3 gắn vào). 180.000 cặp (1.200 query × top-150 dense) cho mỗi model. Kết quả lưu ở `artifacts/rerank/results/full/`.

| Model | Thời gian | Cặp/giây | VRAM đỉnh |
|---|---:|---:|---:|
| bge-reranker-v2-m3 | 339 s | 532 | 1,5 GB |
| Qwen3-Reranker-0.6B | 1.140 s | 158 | 13,1 GB |
| Qwen3-Reranker-4B | 4.750 s | 38 | 20,6 GB |

Jina v3.5 không chạy (CC-BY-NC-4.0, chờ BTC xác nhận).

Bài nộp (dựng bằng `scripts/build_rerank_submission.py`, 150 chunk đã sắp lại theo điểm rerank, 200 doc). Danh sách doc lấy thêm các ứng viên dense hạng 150 trở đi (không rerank) để đủ 200 doc như các bản d200 trước:
- `medxir-rr4b-d200-c150-submission.zip` (89 MB): Qwen3-4B, chỉ rerank.
- `medxir-rr06b-d200-c150-submission.zip` (89 MB): Qwen3-0.6B.
- `medxir-rrbge-d200-c150-submission.zip` (88 MB): bge-m3.
- `medxir-rr4b-f05-d200-c150-submission.zip` (89 MB): 4B + dense, trọng số dense 0,5.
- `medxir-rr4b-f10-d200-c150-submission.zip` (89 MB): 4B + dense, trọng số dense 1,0.

Thứ tự nộp đề xuất: rr4b → rr06b → rrbge → rr4b-f05 tùy kết quả trước đó. Mốc hiện tại: d200-c150 dense, FINAL 0,1847.

**Đính chính (26/10):** Các bài `rr4b`, `rr06b`, `rrbge` và `rr4b-f*` không đổi điểm so với dense, vì chúng chỉ đổi thứ tự 150 chunk đã có. Điểm chunk và doc đều theo tập hợp nên không đổi. Kiểm tra: 1.200/1.200 query có cùng tập chunk và cùng tập doc với bản d200-c150 dense. Reranker chỉ có tác dụng khi chọn 150 chunk từ một pool lớn hơn (top-500), không phải khi sắp lại top-150. Cần chạy lại 600.000 cặp (1.200 × 500) cho mỗi model.

## Hybrid retrieval và ghép chunk — 27–29/10/2026

### Notebook hybrid

`src/notebooks/kaggle_hybrid_retrieval.ipynb` (sinh từ `scripts/build_hybrid_notebook.py`, logic ở `scripts/hybrid_retrieval.py`). Cell config có toggle `USE_QUERY_SPLIT`, `USE_CUSTOM_INSTRUCT`, `USE_PRF`, `USE_BM25`, `USE_RERANK`, `DOC_AGG` (max/top3), `LANGS`, `RERANKER`, `N_DOCS=200`, `N_CHUNKS=150`, `POOL_MAX=600`. Vector 4B đọc theo khối 32k dòng qua mmap (RAM ~5,7 GB), search chính xác trên GPU. Output: `pool.parquet` (`query_id`, `chunk_id`, `final_rank`, …), `hybrid_manifest.json`, ZIP bài nộp. Bật hết ~26 phút trên RTX Pro 6000, trong đó rerank bge-v2-m3 720k cặp mất 19 phút.

### Ghép chunk khi nộp (`scripts/expand_chunks.py`)

Chạy Kaggle CPU, input `medxir-chunk-corpus` + output hybrid (`pool.parquet` và bài nộp, nhận cả `.zip` lẫn `.json` vì Kaggle tự giải nén zip). Giữ nguyên danh sách doc, chỉ dựng lại phần chunk:
- `nbN`: N chunk đầu lấy thêm chunk liền trước và liền sau; các chunk liền nhau ghép thành một đoạn (bỏ overlap theo offset start/end).
- `wholedoc`: nộp nguyên văn các doc đầu danh sách.
- Dung lượng mỗi query = bài gốc × 1,08, vòng lặp giữ ZIP < 99 MB. `ONLY = [...]` chọn biến thể.

### Kết quả leaderboard

| Bài | FINAL | Doc R | Chunk R | Chunk P | Nhận xét |
|---|---:|---:|---:|---:|---|
| Dense 4B d200-c150 | 0,1847 | ~0,44 | 0,170 | 0,067 | Mốc cũ |
| Hybrid bật hết | 0,1965 | 0,473 | 0,181 | 0,072 | |
| Hybrid tắt rerank + BM25 | 0,1759 | 0,433 | 0,159 | 0,061 | Split, PRF, instruct riêng, top3 làm giảm điểm; rerank trên pool sâu mới là phần có ích. 25% chunk ZH |
| Probe VI (bỏ 7% chunk ZH/EN) | 0,1944 | 0,473 | 0,169 | 0,071 | Recall mất đúng tỷ lệ: chunk ZH/EN trúng ngang chunk VI |
| expand nb30 | 0,2184 | 0,473 | 0,229 | 0,113 | Biên chunk đang cắt ngang đáp án |
| **expand nb60** | **0,2235** | 0,473 | 0,224 | 0,142 | Recall đỉnh quanh nb30, nb60 tăng nhờ precision |
| expand wholedoc | 0,2065 | 0,473 | 0,145 | 0,212 | Doc top có đáp án, nhưng doc có đáp án rải rác ở nhiều hạng |
| expand nb60-q20-dq20 | 0,2082 | 0,467 | 0,218 | 0,109 | Ép ≥ 20% chunk (theo byte) và doc ZH/EN lấy từ pool: cả recall lẫn precision đều giảm. ZH/EN hạng thấp trúng kém hơn VI bị thay. Bỏ hướng ép suất ngôn ngữ |
| expand nb60-d300 | 0,2125 | 0,519 | 0,224 | 0,142 | Chunk giữ nguyên, 300 doc: Doc R +0,046 nhưng Doc F2 0,2535 (≈ 0,2755 ở 200 doc). Giữ 200 doc; hạng 200–300 có doc đúng mà rerank doc cần kéo vào |
| **nb60-drr-bge** (rerank doc bge, RRF 1:1 với hạng cũ) | **0,2253** | 0,479 | 0,224 | 0,142 | `kaggle_doc_rerank` (25 phút). Doc F2 0,279. Thay ~11,6 doc/query |
| nb60-drr-bge-only (chỉ điểm bge) | 0,2253 | 0,481 | 0,224 | 0,142 | Thay ~25,6 doc/query nhưng chỉ thêm +0,002 Doc R. Mới lấy được ~15% phần +0,046 ở hạng 200–300 |
| nb60-drr-qwen4b-only (Qwen3-Reranker-4B, vùng 120–360, top 120 giữ nguyên) | 0,2176 | 0,457 | 0,224 | 0,142 | Doc R thấp hơn cả nb60 gốc (−0,016). Qwen-4B kém bge khi chấm cấp doc. Bỏ hướng doc rerank bằng Qwen |
| **v2 nb60-d200** (corpus v2 2,89M chunk, BM25 vẫn v1, không doc rerank; `R2/submission/v2/`) | **0,2437** | 0,519 | 0,249 | 0,155 | Cùng cấu hình nb60, chỉ đổi corpus: +0,020 FINAL, Doc R +0,046, Chunk R +0,025. Corpus mở rộng là đòn bẩy lớn nhất tới giờ |
| **v2 nb60-d200-drr-bge-only** (rerank doc bge 0–400, chỉ điểm bge) | **0,2464** | 0,530 | 0,249 | 0,155 | Doc F2 0,3042 (+0,005), Doc P 0,153. Doc R chỉ +0,011, chunk giữ nguyên. bge vẫn lấy được ít phần recall ở hạng 200–400 |
| v2 nb60-drr-bge-only-glow (global-low: byte của query tự tin chuyển sang query ít chunk điểm bge ≥ 0,5, γ 0,5) | 0,2352 | 0,530 | 0,249 | 0,173 | Chunk P tăng nhưng Chunk F2 macro giảm 0,1886 → 0,1676: query tự tin mất recall (F2 cao, mất nhiều), query yếu thêm chunk gần như không thêm F2 (P thấp chặn F2). Bỏ hướng nhường chỗ cho query yếu |
| **v2 nb60-drr-bge-only-ghigh** (ngược lại: byte dồn cho query tự tin, γ −0,5) | **0,2477** | 0,530 | 0,234 | 0,159 | Mốc tốt nhất. Chunk F2 0,1886 → 0,1908: R giảm 0,015 nhưng P tăng, F2 macro tăng vì query mạnh được thêm recall. Tiếp: γ −1,0 |
| **v2 Qwen3-8B hybrid nb60-d200-drr-bge-only** (`R2/submission/qwen8b/`, model đơn, BM25 v2) | **0,2509** | 0,544 | 0,2509 | 0,155 | Mốc tốt nhất. So với 4B: Doc R +0,014, Doc F2 0,3131; Chunk R gần như không đổi → chunk recall bị giới hạn ở cách chọn đoạn. Hạng 100–200 của danh sách doc chỉ 10% có đoạn được nộp, hạng 50–100 48%: biến thể `cover` |
| **v2 fusion 4B 0,35 · Qwen3-8B 0,35 · Octen-8B 0,30 nb60-d200-drr-bge-only** (`R2/submission/fusion-35-35-30/`, BM25 v2, pool 600) | **0,2539** | 0,551 | 0,2535 | 0,156 | Mốc tốt nhất. So với Qwen3-8B đơn: Doc R +0,007, Doc F2 0,3171 (+0,004), Doc P 0,160; Chunk F2 0,1906 (+0,002). Gộp 3 model tăng chủ yếu phía doc, chunk recall vẫn gần như đứng yên. Chưa có ghigh/cover trên bản này |
| v2 nb60-drr-bge-only-ghigh1-min200-s20 (γ −1 + bỏ đoạn < 200 ký tự + bỏ chunk bge < 0,2, gộp 3 bước chưa thử riêng) | 0,2450 | 0,530 | 0,2195 | 0,177 | Chunk P +0,018 nhưng R −0,015 so với ghigh: Chunk F2 0,1908 → 0,1844. Ở P ≈ 0,17, R ≈ 0,22, một đơn vị R đáng ≈ 2,6 đơn vị P trong F2; cắt chunk để lấy precision là lỗ. Bước nào làm mất R chưa tách được |

### Kết luận

- Doc recall 0,47 là trần chung của mọi bài hybrid; xếp hạng doc là nút thắt. Ưu tiên số một: tăng doc recall (~0,56).
- Small-to-big: giữ chunk 350/50 để search, ghép thành đoạn dài khi nộp. Không cắt lại corpus.
- Kernel embed 8B của nhóm đã có (Qwen3-8B, Octen-8B s0–s2 + query, ICD Octen) nhưng chưa kiểm tra output. Máy local hiện không có token `boinhbo` (`~/.config/agent-auto-research/kaggle/boinhbo.token`) nên chưa kiểm tra được qua `tools.kaggle`.

## Ensemble 8B, cửa sổ, PRF và biến thể query — 27–30/10/2026

Hướng B, toàn bộ trên corpus v1 (2,62 triệu chunk). Nhật ký đầy đủ 27–29/10 ở [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md); mục này tóm tắt và thêm phần 29–30/10. Ghi chú: máy này có token `boinhbo` và `dinhbaobao`, các kernel 8B ở mục cuối phần hướng A đã chạy xong và dùng ở đây.

### Cách làm

- Mỗi nguồn cho top-500 chunk mỗi query, lưu ở `artifacts/submission/candidates-*-k500.npz`. Tìm kiếm chính xác trên CPU local theo khối (`retrieve_local_8b.py`, đỉnh RAM 5,4 GB, 2,5 phút mỗi lần).
- `scripts/fuse_scores.py`: min-max điểm từng nguồn trong từng query, cộng theo trọng số; doc = chunk tốt nhất + 1,0 × (chunk thứ 2 + thứ 3) (`--doc-beta`); `--context 1` nộp mỗi chunk kèm chunk liền trước và sau (cửa sổ 3 chunk), bỏ chunk đã nằm trong cửa sổ cao hơn; 70 cửa sổ ≈ 92 MB zip.
- **PRF** (`scripts/prf_local_8b.py`): q' = chuẩn hóa(q + 0,3 × trung bình vector của 5 chunk đầu danh sách ens4), dùng vector corpus của chính model đó. Cosine(q, q') trung bình 0,990; top-20 đổi khoảng 4/20.
- **Biến thể query bằng LLM** (`scripts/build_variants_notebook.py` → `kaggle_query_variants.ipynb`, exp_0023, 16 phút RTX Pro 6000): Qwen3-8B (thinking tắt, greedy, prompt tiếng Việt kèm một ví dụ) sinh JSON cho mỗi query: `question` (câu hỏi chuẩn hóa, trung vị 16 từ), `terms` (thuật ngữ y khoa, 20 từ), `passage` (đoạn HyDE, 93 từ). 1.199/1.200 parse được. Octen-8B và Qwen3-Embedding-8B encode `question`, `terms` như query và `passage` như tài liệu. `scripts/retrieve_variants_8b.py` tìm local. Top-20 trùng với query gốc (Octen): question 14,1; passage 10,1; terms 5,2. Có thuật ngữ bịa ("thủy đậu răng", "tràng phù") nên trọng số biến thể nhỏ.
- Package mới: `boinhbo/medxir-package-qwen3llm8b` (Qwen/Qwen3-8B, sha b968826, 16,4 GB, exp_0021). Qwen3-14B (~30 GB) vượt giới hạn 20 GB của `/kaggle/working`.

### Kết quả leaderboard (cộng dồn)

| Bài | Thành phần | FINAL | Doc F2 | Chunk F2 | Doc R | Chunk P | Chunk R |
|---|---|---:|---:|---:|---:|---:|---:|
| ens4-b1 | 4B 0,25 · Octen 0,35 · 0.6B 0,15 · BM25 0,25 | 0,1975 | 0,2788 | 0,1163 | 0,4789 | 0,0719 | 0,1820 |
| ens4-b1-x1 | + cửa sổ 3 chunk × 70 | 0,2410 | 0,2788 | 0,2033 | 0,4789 | 0,2053 | 0,2426 |
| ens5-b1-x1 | + Qwen3-Embedding-8B 0,20 | 0,2406 | 0,2775 | 0,2037 | 0,4775 | 0,2049 | 0,2435 |
| ens4-prfoct-b1-x1 | ens4-b1-x1, Octen → Octen-PRF | 0,2429 | 0,2804 | 0,2055 | 0,4811 | 0,2086 | 0,2459 |
| **var-all-b1-x1** | 4B 0,20 · Octen-PRF 0,27 · 0.6B 0,10 · BM25 0,20 · Octen q/p/t 0,06/0,06/0,015 · Qwen-8B q/p/t 0,04/0,04/0,015 | **0,2444** | 0,2828 | 0,2060 | 0,4849 | 0,2082 | 0,2468 |

Nhận xét: cửa sổ 3 chunk là bước lớn nhất (+0,044). Qwen-8B thêm vào ens4 không giúp (cùng họ với 4B và 0.6B; đổi 10% tâm cửa sổ). PRF và biến thể mỗi bước khoảng +0,002, chủ yếu tăng Doc R. Chưa có ablation cho từng loại biến thể.

### Đã dựng, chưa nộp (ablation)

- `only8b-var-b1-d200-c70-x1`: bỏ 4B và 0.6B (Octen-PRF 0,35 · Qwen-PRF 0,22 · BM25 0,20 · biến thể như var-all). Nếu không thua, pipeline private chỉ cần hai model 8B và BM25.
- `var-all-b1-d200-c50-x2`: var-all với cửa sổ 5 chunk, 50 cửa sổ (89 MB; mỗi cửa sổ K=2 ≈ 1,76 MB zip). Doc giữ nguyên nên Doc F2 phải bằng 0,2828.
- Dự kiến tiếp: var-all bỏ `terms`, var-all bỏ `passage`.

### Ghép với hướng A

Corpus v2 cho +0,020 ở hướng A, bằng mọi thay đổi model ở hướng B cộng lại. Corpus v2 là v1 cộng một file chunk additions riêng ([docs/DATA_ADDITIONS_20261029.md](docs/DATA_ADDITIONS_20261029.md)), nên vector 8B của v1 vẫn dùng được. Để dùng ensemble 8B trên v2 cần embed phần chunk thêm vào v2 (~0,27 triệu chunk) bằng Octen-8B và Qwen3-Embedding-8B (khoảng 380 token mỗi chunk ở 19k token/s, ước tính khoảng 1,5 giờ GPU mỗi model), rồi chạy lại BM25 v2, PRF và tìm kiếm biến thể trên v2.

## Ghép hai hướng trên corpus v2 — 30/10/2026

**Kết quả: FINAL 0,2755**, cao nhất tới giờ (+0,0216 so với 0,2539 của hướng A; +0,031 so với 0,2444 của hướng B trên v1).

| Bài | FINAL | Doc F2 | Chunk F2 | Doc P | Doc R | Chunk P | Chunk R |
|---|---:|---:|---:|---:|---:|---:|---:|
| **v2ens-bge25-drr-d200-c76-x1** | **0,2755** | **0,3197** | **0,2314** | 0,1611 | **0,556** | **0,2194** | **0,2842** |
| Hướng A tốt nhất: v2 fusion 4B · Qwen3-8B · Octen-8B nb60-drr-bge-only | 0,2539 | 0,3171 | 0,1906 | 0,160 | 0,551 | 0,156 | 0,2535 |
| Hướng B tốt nhất (v1): var-all-b1-x1 | 0,2444 | 0,2828 | 0,2060 | 0,145 | 0,4849 | 0,2082 | 0,2468 |

Vế doc gần bằng hướng A (+0,003): cùng corpus v2, cùng bge doc rerank. **Phần tăng nằm ở vế chunk (+0,041)**: Chunk P 0,156 → 0,219 và Chunk R 0,254 → 0,284. Hai khác biệt so với nb60: (1) mọi mục nộp đều là cửa sổ 3 chunk, cửa sổ nào chứa chunk đã có trong cửa sổ xếp cao hơn thì bỏ, nên byte không bị lặp; (2) chunk được xếp theo điểm gộp của cả ensemble và bge, không chỉ bge. Cửa sổ chiếm hết ngân sách (76 cửa sổ, ZIP 96,5 MB so với 92 MB của 70 cửa sổ trên v1, vì chunk ZH nén tốt hơn).

### Công thức (tái lập được)

1. **Ứng viên trên v2** (`scripts/retrieve_additions.py`, CPU local, 5,5 phút, RAM 4,5 GB): mỗi danh sách top-500 v1 đã có được ghép với top-500 trên ba part mới (`vi-part-007`, `en-part-001`, `zh-part-001`) theo cosine. Cùng model nên cùng thang điểm, chunk id không trùng, nên kết quả bằng đúng tìm trên toàn v2. Vector additions: datasets `ntklinhfitus/medxir-vectors-{qwen3-4b,qwen3-8b,octen-8b}-additions` (tải về `artifacts/kaggle/vectors-{qwen3-4b,qwen8b,octen8b}-add/`), chunk v2 ở `artifacts/kaggle/chunk-corpus-v2/`. PRF dựng lại như `prf_local_8b.py` (5 chunk phản hồi lấy từ ens4 trên v1). Ra `artifacts/submission/candidates-<list>-v2-k500.npz`.
2. **Gộp điểm** (`scripts/fuse_scores.py --corpus v2`): min-max theo query, cộng có trọng số, pool hợp:

| Nguồn | Trọng số |
|---|---:|
| bge-reranker-v2-m3 (`final_score` của `ntklinhfitus/medxir-hybrid-v2` pool.parquet, 600 chunk/query từ 4B + BM25 trên v2) | 0,25 |
| BM25 tiếng Việt v2 (`ntklinhfitus/medxir-bm25-vi`) | 0,14 |
| Octen-8B query gốc / PRF | 0,09 / 0,12 |
| Qwen3-Embedding-8B query gốc / PRF | 0,08 / 0,09 |
| Qwen3-Embedding-4B query gốc | 0,09 |
| Biến thể LLM Octen question / passage / terms | 0,035 / 0,035 / 0,01 |
| Biến thể LLM Qwen3-8B question / passage / terms | 0,025 / 0,025 / 0,01 |

   Trọng số chọn tay, chưa tối ưu. 92,8% tâm cửa sổ (khi chưa có bge) nằm trong pool bge, nên điểm bge phủ gần hết. bge 0,25 đổi 20% tâm cửa sổ và 17,5% doc so với không bge.
3. **Danh sách doc:** điểm doc = chunk tốt nhất + 1,0 × (chunk thứ 2 + thứ 3), 200 doc. Xuất pool top-600 và bài nộp (`--export-pool 600`) lên dataset riêng `boinhbo/medxir-docrerank-ens-v2`; kernel `boinhbo/medxir-doc-rerank-ens-v2` (`src/notebooks/kaggle_doc_rerank_ens.ipynb` = `build_doc_rerank_notebook.py --gpu-bge`, config `configs/kaggle/doc_rerank_ens_v2_rtx6000.yaml`, exp_0028) chấm bge cho các doc hạng 0–400 (414.164 cặp, 20 phút RTX Pro 6000). `fuse_scores.py --doc-rank` xếp lại theo quy tắc bge-only của hướng A: ứng viên xếp theo điểm bge, giữ số doc như cũ, rồi thêm doc của mọi chunk đã nộp. bge thay trung bình 50,9 doc/query; 193–232 doc/query (trung vị 205).
4. **Chunk:** `--context 1 --chunk-counts 76`: 76 cửa sổ (chunk + chunk liền trước + liền sau, ghép theo offset), bỏ cửa sổ có tâm nằm trong cửa sổ cao hơn cùng văn bản.

```
A=artifacts/submission; E() { echo "--extra $A/candidates-$1-v2-k500.npz:$2"; }
uv run python scripts/fuse_scores.py --tag v2ens-bge25-drr --corpus v2 --bm25 artifacts/bm25-v2/vi-candidates-k500.npz --w-bm25 0.14 \
  $(E 4b 0.09) $(E octen8b 0.09) $(E octen8b-prf 0.12) $(E qwen8b 0.08) $(E qwen8b-prf 0.09) \
  $(E octen8b-question 0.035) $(E octen8b-passage 0.035) $(E qwen8b-question 0.025) $(E qwen8b-passage 0.025) \
  $(E octen8b-terms 0.01) $(E qwen8b-terms 0.01) --bge-pool artifacts/hybrid-v2/pool.parquet:0.25 \
  --pool union --doc-beta 1.0 --context 1 --chunk-counts 76 --doc-rank $A/docrerank-v2ens-bge25/doc_rank.parquet
```

Bỏ `--doc-rank` và thêm `--export-pool 600` để dựng bản trước doc rerank (input cho kernel).

### Hướng tiếp theo (xếp theo kỳ vọng)

1. **Thêm byte cho chunk.** Chunk R vẫn bị giới hạn bởi 100 MB (phân tích của hướng A). Nếu dashboard nhận ZIP bzip2 (đang thử bằng file `BZIP2TEST` của hướng A), cùng giới hạn chứa được ~1,55× văn bản: khoảng 115 cửa sổ thay vì 76.
2. **Byte theo độ tự tin (ghigh):** hướng A +0,0013 với γ −0,5. Áp vào số cửa sổ mỗi query.
3. **bge cho toàn pool của ensemble:** pool bge hiện chỉ là 600 chunk của 4B + BM25; 7% tâm cửa sổ của ta không có điểm bge.
4. **Doc list:** RRF 1:1 giữa thứ hạng ensemble và bge thay cho bge-only (bge thay tới 51 doc/query); `cover` của hướng A (doc hạng 100–200 thường không có đoạn nào được nộp).

## Lên 0,2984: bge toàn pool, nén, doc-prior, LLM rerank — 31/10/2026

Mọi bài dưới đây đều trên công thức 0,2755 (mục trước), corpus v2, d200, cửa sổ 3 chunk (`--context 1`).

| Bài | FINAL | Doc F2 | Chunk F2 | Doc P / R | Chunk P / R | Thay đổi so với dòng trên |
|---|---:|---:|---:|---|---|---|
| v2ens-bge25-drr-c76 (deflate) | 0,2755 | 0,3197 | 0,2314 | 0,161 / 0,556 | 0,219 / 0,284 | |
| v2ens-bgefull-drr-c120-bz | 0,2916 | 0,3206 | 0,2627 | 0,159 / 0,565 | 0,190 / 0,361 | bge chấm toàn pool + ZIP bzip2 (120 cửa sổ) |
| v2ens-bgefull-drr-dp03-c131-lz | 0,2965 | 0,3197 | 0,2734 | 0,161 / 0,558 | 0,191 / 0,381 | doc-prior 0,3 + ZIP LZMA (131 cửa sổ) |
| ...-dp03-llm05-c131-lz | 0,2976 | **0,3236** | 0,2716 | 0,164 / 0,559 | 0,189 / 0,382 | Qwen3-8B rerank doc (trọng số 0,5), có cả trong doc-prior |
| **...-dp03bge-llm10-c131-lz** | **0,2984** | 0,3235 | **0,2734** | 0,165 / 0,558 | 0,191 / 0,381 | LLM 1,0 chỉ ở vế doc; doc-prior chỉ bge |
| v2ens-bge35-drr-dp05bge-llm10-c131-lz | 0,2981 | 0,3233 | 0,2728 | 0,165 / 0,558 | 0,192 / 0,379 | bge 0,35 + doc-prior 0,5: P lên, R xuống |

### Các bước

1. **bge cho toàn pool** (`scripts/build_chunk_rerank_notebook.py` → `kaggle_chunk_rerank.ipynb`, config `chunk_rerank_ens_v2_rtx6000.yaml`, kernel `boinhbo/medxir-chunk-rerank-ens-v2`, exp_0029, 16 phút): bge-reranker-v2-m3 chấm 600.000 cặp (1.200 query × 500 chunk của pool ensemble trong dataset `boinhbo/medxir-docrerank-ens-v2`). Ra `pool.parquet` cùng định dạng pool hybrid (`final_score` = bge), tải về `artifacts/bge-ens-v2/`, dùng qua `--bge-pool ...:0.25`. Trước đó pool bge của hướng A thiếu 7% tâm cửa sổ.
2. **Nén ZIP mạnh hơn** (`--zip bzip2|lzma`, `retrieve_local_4b.write_submission(method=...)`): cùng JSON, deflate 96,7 MB = 76 cửa sổ; bzip2 = 120 cửa sổ; LZMA = 131 cửa sổ (98 MB). Dashboard nhận cả bzip2 và LZMA. Chunk R tăng theo số cửa sổ (0,284 → 0,361 → 0,381): mỗi cửa sổ thêm trúng ≈ 9%, trên ngưỡng p > F2/5 ≈ 4,6% để thêm một mục làm tăng F2.
3. **doc-prior** (`--doc-prior 0.3`): điểm chunk + 0,3 × điểm bge doc cao nhất trong các doc của chunk (`doc_rank.parquet` của doc rerank), trước khi chọn cửa sổ. Đổi ~12% tâm cửa sổ; doc thêm vào chỉ vì có chunk giảm (trung vị 212 → 203). 0,5 cùng bge 0,35 không tốt hơn.
4. **Qwen3-8B rerank doc** (`build_doc_rerank_notebook.py --llm Qwen3-8B --zone 0 120 --shard i 2`, configs `doc_rerank_llm_s{0,1}_rtx6000.yaml`, chạy song song boinhbo / dinhbaobao, ~53 phút mỗi shard; package `{boinhbo,dinhbaobao}/medxir-package-qwen3llm8b`): prompt yes/no của Qwen3-Reranker, cặp = 400 ký tự đầu + chunk tốt nhất, 1024 token, chấm 120 doc đầu (thứ tự fusion) mỗi query. Gộp: `--doc-rank-extra s0,s1:Qwen3-8B:1.0` (RRF với thứ hạng bge cho danh sách doc), `--doc-prior-bge-only` (LLM trong doc-prior làm Chunk F2 −0,0018). Spearman LLM–bge trung vị 0,233. Trọng số 0,5 hay 1,0 cho cùng Doc F2.

### Lỗi đã sửa: prompt Qwen bị cắt

Code chấm kiểu Qwen trong `build_doc_rerank_notebook.py` cắt cả chuỗi `prompt + query + doc + suffix` ở 512 token, nên mọi cặp dài mất phần đuôi `<|im_start|>assistant…</think>` và yes/no được đọc giữa văn bản (lần chạy đầu: Spearman với bge 0,06). Đã sửa theo code mẫu của Qwen3-Reranker: chỉ cắt phần doc, luôn giữ đuôi; logit chỉ tính ở vị trí cuối. **Kết quả Qwen3-Reranker-4B 0,2176 của hướng A (30/10) dùng code cũ nên chưa phản ánh đúng chất lượng model.**

### Prompt v2 (chưa dùng)

Instruct chi tiết (cùng bệnh / triệu chứng / thuốc và đúng điều người hỏi cần; "no" khi chỉ trùng từ) — `doc_rerank_llm2_s{0,1}_rtx6000.yaml`, exp_0036/0037, `artifacts/llm-docrank/doc_rank_v2_s*.parquet`. Spearman với bge 0,259, nhưng điểm bão hòa (trung vị 1,0). Trung bình v1 + v2 cho đúng cùng tập 200 doc với 0,2984: LLM chỉ chấm 120 doc đầu, vốn đã nằm trong tập, nên chỉ đổi thứ tự.

### Công thức 0,2984

```
A=artifacts/submission; E() { echo "--extra $A/candidates-$1-v2-k500.npz:$2"; }
uv run python scripts/fuse_scores.py --tag v2ens-bgefull-drr-dp03bge-llm10 --corpus v2 --bm25 artifacts/bm25-v2/vi-candidates-k500.npz --w-bm25 0.14 \
  $(E 4b 0.09) $(E octen8b 0.09) $(E octen8b-prf 0.12) $(E qwen8b 0.08) $(E qwen8b-prf 0.09) \
  $(E octen8b-question 0.035) $(E octen8b-passage 0.035) $(E qwen8b-question 0.025) $(E qwen8b-passage 0.025) \
  $(E octen8b-terms 0.01) $(E qwen8b-terms 0.01) --bge-pool artifacts/bge-ens-v2/pool.parquet:0.25 \
  --pool union --doc-beta 1.0 --context 1 --doc-rank $A/docrerank-v2ens-bge25/doc_rank.parquet \
  --doc-prior 0.3 --doc-prior-bge-only \
  --doc-rank-extra artifacts/llm-docrank/doc_rank_s0.parquet,artifacts/llm-docrank/doc_rank_s1.parquet:Qwen3-8B:1.0 \
  --chunk-counts 131 --zip lzma
```

### Nhận xét và hướng tiếp

- Byte vẫn là đòn bẩy mạnh nhất ở vế chunk; trọng số chọn chunk đã ở đỉnh cục bộ (±0,0006).
- Vế doc bão hòa ~0,32. LLM chỉ đổi tập doc khi chấm vùng quanh ngưỡng cắt 200 (hạng ~150–300 theo bge), chưa làm.
- LLM chấm từng cửa sổ (chọn 131 trong ~200 ứng viên) nhắm vào Chunk P 0,19; chưa làm (~1,5 giờ trên 2 GPU).
- Mở rộng query theo bệnh nghi ngờ cho query mơ hồ (~32%) nhắm vào Doc R 0,558.

