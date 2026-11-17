# Pipeline đạt 0,298 và đề xuất rerank có điểm tiêu đề — 01/11/2026

Tài liệu cho nhóm pomodoro và working notes paper. Phần 1–4 mô tả pipeline đã chấm; phần 5 là đề xuất **chưa chấm**.
Chỗ nào suy ra từ file nộp (chưa có code của Bảo trên remote) được ghi rõ "suy ra".

## 1. Kết quả

| Bản | FINAL | Doc F2 (P / R) | Chunk F2 (P / R) |
|---|---:|---|---|
| v2ens-bge25-drr-d200-c76-x1 (Bảo, 30/10) | 0,2755 | 0,3197 (0,161 / 0,556) | 0,2314 (0,219 / 0,284) |
| v2ens-bgefull-drr-d200-c120 (Bảo, 31/10) | 0,2916 | 0,3206 (0,159 / 0,565) | 0,2627 (0,190 / 0,361) |
| v2ens-bgefull-drr-dp03-d200 (Bảo, 31/10) | 0,2965 | 0,3197 (0,161 / 0,558) | 0,2734 (0,191 / 0,381) |
| **v2ens-bgefull-drr-dp03-title-d200 (31/10)** | **0,298** | **0,3226 (0,163 / 0,563)** | 0,2734 (0,191 / 0,381) |

File: `R2/submission/title/medxir-v2ens-bgefull-drr-dp03-title-d200-submission.zip` (98,1 MB, ZIP LZMA, một JSON 648 MB).

## 2. Sơ đồ

```
1.200 query tiếng Việt
 │
 ├─ (A) Ứng viên chunk trên corpus v2 (2,89 triệu chunk VI/EN/ZH, chunk 350 từ, chồng 50)
 │     11 danh sách top-500 + BM25 + điểm bge của pool hybrid → gộp min-max có trọng số → pool 500 chunk/query
 │
 ├─ (B) bge-reranker-v2-m3 chấm lại cả 500 chunk của pool  ("bgefull")
 │
 ├─ (C) Danh sách doc: ứng viên hạng 0–400 → bge chấm (400 ký tự đầu + chunk tốt nhất) → xếp chỉ theo bge
 │     + (E) doc mới tìm qua TIÊU ĐỀ, chấm cùng cách → xếp lại theo bge → 200 doc
 │
 ├─ (D) Đoạn nộp: cửa sổ 3 chunk (chunk tâm ± 1), xếp theo điểm gộp + doc prior 0,3, 131 cửa sổ/query
 │
 └─ ZIP LZMA (< 100 MB), mỗi doc của mọi đoạn được nối vào cuối danh sách doc
```

## 3. Từng bước

### (A) Ứng viên chunk — `scripts/fuse_scores.py --corpus v2` (Bảo, CPU local)

Công thức đầy đủ ở PROGRESS, mục "Ghép hai hướng trên corpus v2 — 30/10/2026". Tóm tắt:

| Nguồn (mỗi nguồn top-500, min-max theo query) | Trọng số |
|---|---:|
| bge-reranker-v2-m3 trên pool hybrid 600 chunk (4B + BM25, `ntklinhfitus/medxir-hybrid-v2`) | 0,25 |
| BM25 tiếng Việt v2 (âm tiết + cặp âm tiết, `ntklinhfitus/medxir-bm25-vi`) | 0,14 |
| Octen-8B: query gốc / PRF | 0,09 / 0,12 |
| Qwen3-Embedding-8B: query gốc / PRF | 0,08 / 0,09 |
| Qwen3-Embedding-4B: query gốc | 0,09 |
| Biến thể LLM (Qwen3-8B) question / passage / terms — Octen | 0,035 / 0,035 / 0,01 |
| Biến thể LLM question / passage / terms — Qwen3-8B | 0,025 / 0,025 / 0,01 |

PRF: 5 chunk đầu làm phản hồi, tìm lại lần hai. Pool xuất ra 500 chunk/query (dataset `boinhbo/medxir-docrerank-ens-v2`,
cột `final_score` = điểm gộp).

### (B) bge trên toàn pool — "bgefull" (Bảo)

bge-reranker-v2-m3 chấm (query, chunk) cho cả 500 chunk/query (600.000 cặp; kernel `medxir-chunk-rerank-ens-v2`, file
`artifacts/bge-ens-v2/pool.parquet`). Trước đó chỉ 92,8% tâm cửa sổ có điểm bge.

### (C) Danh sách doc — rerank doc bge-only

- Ứng viên: 200 doc của danh sách gộp, rồi các doc khác của pool (chunk tốt nhất trước), tới hạng 400.
- Văn bản cho bge: 400 ký tự đầu bài (thường là tiêu đề / mở đầu) + chunk tốt nhất trong pool, tối đa 512 token.
- Xếp chỉ theo điểm bge, giữ 200 doc (`doc_rank.parquet` của kernel `boinhbo/medxir-doc-rerank-ens-v2`;
  bản sao ở `ntklinhfitus/medxir-ens-v2-bgefull`).

### (D) Đoạn nộp — cửa sổ, doc prior, LZMA (Bảo; một phần suy ra)

- Mỗi đoạn = chunk tâm + chunk liền trước + liền sau cùng văn bản (ghép theo offset, bỏ phần chồng). Bỏ tâm đã nằm
  trong cửa sổ xếp cao hơn.
- Thứ tự tâm: điểm gộp 0,75 + bge 0,25 (bản 0,2916; audit của Codex khớp 144.096 tâm).
- **dp03 (suy ra từ tên file JSON `...-dp03-d200-c131-x1-lz`)**: cộng 0,3 × điểm doc vào điểm cửa sổ, nên doc mạnh có
  thêm cửa sổ thứ hai (tỉ lệ doc có ≥ 2 cửa sổ 13,5% → 17%).
- 131 cửa sổ/query, ZIP LZMA (bản 0,2916: 120 cửa sổ, BZIP2; bản 0,2755: 76 cửa sổ, DEFLATE). Nén chặt hơn cho phép
  nộp thêm đoạn trong giới hạn 100 MB; Chunk R tăng theo số đoạn trong khi Chunk P giữ ~0,19.

### (E) Tìm doc theo tiêu đề — bước mới của bản 0,298

`scripts/title_search.py`, kernel `ntklinhfitus/medxir-title-search` (RTX Pro 6000, ~27 phút), config
`configs/kaggle/title_search_norewrite_rtx6000.yaml`.

1. Tiêu đề bài (cột `title` của chunk corpus v2, dòng `order = 0`): 1.178.504 tiêu đề (VI 722.851, EN 239.789,
   ZH 215.864).
2. Embed tiêu đề bằng Qwen3-Embedding-8B (không instruction, tối đa 64 token); query gốc có instruction
   "Given a Vietnamese patient question, retrieve health articles whose title asks or answers the same question".
   Last-token pooling, chuẩn hóa L2, tích vô hướng, top 150 tiêu đề/query.
3. Doc của các tiêu đề đó **chưa có** trong ứng viên của (C): tối đa 100 văn bản/query (96.313 cặp).
4. Chấm như (C): bge(query, 400 ký tự đầu + chunk) với 6 chunk đầu của văn bản, lấy chunk tốt nhất (169.253 cặp).
5. Xếp lại bge-only trên ứng viên cũ ∪ doc mới, giữ 200; nối doc của mọi đoạn vào cuối như cũ. Đối chiếu: 200 doc đầu
   của bản 0,2965 trùng 100% với thứ tự bge-only cũ trên cả 1.200 query, nên đây đúng là cùng quy tắc, thêm nguồn
   ứng viên.
6. Đoạn nộp giữ nguyên bản 0,2965. Ghép (bước 5–6) đang chạy bằng đoạn Python local (cần chuyển thành script).

Kết quả: 11,6 doc mới/query (trung vị 10) vào top 200, thay các doc hạng ~189–200. Doc R 0,5577 → 0,5634, Doc P
0,1611 → 0,1626, Chunk giữ nguyên. Lưu ý: doc mới được chọn chunk tốt nhất trong 6 chunk bằng chính bge, doc cũ thì
chunk lấy từ pool, nên điểm doc mới hơi lạc quan.

## 4. Đã thử và lỗ (đừng lặp lại)

| Thử | Điểm | Bài học |
|---|---:|---|
| Dịch query EN/ZH làm biến thể + rerank cùng ngôn ngữ (max) | 0,2383 (nền 0,2539) | Doc ZH 15% → 35%, Doc R giảm. Đáp án nghiêng VI; đừng đẩy cân bằng ngôn ngữ |
| BM25 theo từ kiểu BTC (pyvi) | 0,1017 | Doc P 0,15 ở 40 doc; 0,88 của BTC không tái tạo được |
| Ép mỗi doc 1 cửa sổ, phủ top 50 doc trước (cov50) | 0,2829 (nền 0,2916) | Cửa sổ thứ hai ở doc mạnh trúng nhiều hơn cửa sổ đầu ở doc yếu: chọn cửa sổ thuần theo điểm |
| Passage: ghép cửa sổ trước rồi bge chấm nguyên đoạn (Codex) | 0,2609 | Kém cách chọn tâm theo điểm gộp |
| Cắt chunk/đoạn điểm thấp, bỏ đoạn ngắn | 0,2450 (nền 0,2477) | Ở P ~0,19, đoạn đuôi vẫn trúng ~10% > ngưỡng ~F2/5 ≈ 5,5%: cắt là lỗ |

## 5. Đề xuất (chưa chấm): rerank cửa sổ có điểm tiêu đề + cửa sổ cho doc mới

### Ý tưởng

Cộng một tín hiệu cấp doc mới — độ khớp giữa câu hỏi và **tiêu đề** của doc — vào điểm xếp cửa sổ, cùng kiểu doc
prior 0,3 đã có lời (0,2916 → 0,2965). Đồng thời cho các doc mới từ (E) một cửa sổ (hiện chúng chỉ có trong danh sách
doc, chưa có đoạn).

### Công thức

```
điểm cửa sổ = 0,75 × điểm gộp (tâm) + 0,25 × bge (tâm) + 0,3 × điểm doc + w_title × khớp_tiêu_đề(doc)
khớp_tiêu_đề = cosine(query, tiêu đề) theo Qwen3-Embedding-8B, min-max trong query;  w_title ≈ 0,1–0,2
```

- Cộng thêm, không thay điểm cũ; trọng số nhỏ vì bài tổng quát ("Trào ngược dạ dày: nguyên nhân và cách điều trị")
  khớp nhiều câu hỏi.
- Doc mới từ (E): ứng viên tâm = chunk tốt nhất bge đã chọn ở bước (E4), điểm theo cùng công thức (điểm gộp = 0).
- Ngân sách 100 MB đang đầy (98,1 MB): thêm cửa sổ nào thì bỏ cửa sổ điểm thấp nhất tương ứng.
- **Không** dùng embedding thay bge cho chunk EN/ZH: điểm xếp cửa sổ đã 75% là embedding đa ngôn ngữ, và mọi cách nâng
  EN/ZH đã đo đều lỗ.

### Cần làm

1. Chạy lại `title_search.py` lưu cosine tiêu đề cho **mọi** doc ứng viên (hiện chỉ lưu doc mới) — ~25 phút RTX.
2. Code chọn cửa sổ dp03 của Bảo (push lên repo); nếu không có, viết lại trong `expand_chunks.py` (điểm gộp + doc
   prior + LZMA) và kiểm tra dựng lại khớp bản 0,2965 trước khi thêm điểm tiêu đề.
3. Chuyển bước ghép (E5–E6) thành script.
4. Một bài private (5 lượt, có hiện điểm): giữ danh sách doc như 0,298, chỉ đổi đoạn — đo riêng tác dụng.

### Có thể kết hợp sau

Câu rewrite kiểu tiêu đề (`translate_queries.py` task `vq`, kernel `ntklinhfitus/medxir-rewrite-queries`) để tìm tiêu
đề tốt hơn ở bước (E2): bản 2 của title search (`configs/kaggle/title_search_rtx6000.yaml`).
