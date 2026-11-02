# Nghiên cứu tăng recall — 31/10/2026

## Mục tiêu và trạng thái

Mục tiêu người dùng: Chunk Recall 0,45–0,50; Doc/Chunk Precision 0,20–0,30;
FINAL > 0,40 bằng pipeline truy hồi có thể tái lập. **Chưa đạt**. Baseline đã chấm:
`v2ens-bge25-drr-d200-c76-x1`, FINAL 0,2755, Doc F2 0,3197,
Chunk F2 0,2314, Doc P/R 0,1611/0,5560, Chunk P/R 0,2194/0,2842.

Người dùng xác nhận còn **3 lượt public**, hai bản win-cov50 **chưa nộp**.
Không tự nộp. Ưu tiên cải tiến trên đúng pool của bản 0,2755; không trộn doc/chunk
của hai hệ, không nén bzip2, không sửa tham số theo ID query, không dùng nhãn bị lộ.
Không commit/push trong đợt này nếu chưa được yêu cầu.

## Bằng chứng và cách đo

- Đã đọc PROGRESS.md, docs/EXPERIMENTS.md, docs/RESEARCH_20261030.md,
  generator doc rerank, expand_chunks và phần xuất pool của fuse_scores.
- Đọc trực tiếp API công khai của BTC: luật, lịch, bảng điểm tổng hợp; không đọc
  endpoint kết quả chi tiết. Snapshot: `artifacts/recall_research_20261031/`.
- `scripts/audit_submission_coverage.py` đọc ZIP tuần tự từng query, không nạp
  corpus hoặc toàn bộ JSON vào RAM. Báo cáo `submission_audit.json` chứa số đo
  trên đủ 1.200 query của 7 bản, độ dài đoạn, độ phủ theo hạng và digest tập kết quả.
- Dataset `boinhbo/medxir-docrerank-ens-v2` đã truy cập được sau khi được chia sẻ.
  Đã tải pool 7,33 MB và doc_rank 2,37 MB từ kernel
  `boinhbo/medxir-doc-rerank-ens-v2` (COMPLETE).
- Pool thật có **500 chunk/query**, tổng 600.000 hàng; không phải 600/query như
  mô tả `--export-pool 600`. Nguồn đầu vào chỉ còn 500 thì slice 600 không tăng được.
  Có trung bình 345,9 doc/query trong doc_rank, tối đa 400.
- Cột `final_score` trong pool này là **điểm ensemble**, `rrf` toàn 0. Không được
  diễn giải final_score là xác suất bge hoặc chạy lại pha trộn như pool hybrid.
- Ổ D hiện còn khoảng 7,08 GB (đã kiểm tra lại), khác recap 300 MB. Chỉ tải bảng
  nhỏ; corpus/model tiếp tục xử lý trên Kaggle.

## Luật chấm và những suy luận cần đính chính

Nguồn chính: https://leaderboard.aiguru.com.vn/api/competitions/15/ (Evaluation).
Chunk đúng khi cùng doc_id và `LCS_tokens(c,g) / max(len(c),len(g)) >= 0.4`.
Không phải chỉ chia cho độ dài tham chiếu. Nộp cả doc có thể bị phạt vì quá dài.
Chuẩn hóa NFKC, HTML entities, lowercase, whitespace. Trang chính thức không chỉ
rõ tokenizer; không tự coi `.split()` của repo khác là scorer BTC.

Macro F2 tính **từng query rồi trung bình**. Không tính F2 từ macro P/R để báo điểm
thật. Hàm F2 lõm cho phép dùng F2(macro P,macro R) làm cận trên: với Chunk P <=0,30,
R <=0,50 thì Chunk F2 <=0,4412. Giữ Doc F2 0,3197 cho FINAL <=0,3805.
Vì vậy mục tiêu FINAL >0,40 đòi hỏi cải thiện phía doc cùng lúc.

Không suy số gold/doc mỗi query chính xác từ tỉ số các chỉ số macro. Chưa có căn cứ
độc lập cho khẳng định mỗi doc đúng chỉ có một chunk vàng. Không dùng ước lượng này
làm lý do loại tuyệt đối cửa sổ thứ hai. Không coi sigmoid bge là xác suất đã được
hiệu chỉnh; ngưỡng 0,208 hiện là heuristic, không phải ngưỡng F2 đã chứng minh.

`wholedoc` kém không tách được lỗi xếp hạng doc khỏi độ dài đoạn/độ phủ doc.
Tắt đồng thời BM25 và rerank cũng không chứng minh riêng split/PRF/instruction gây
giảm điểm: đó là thí nghiệm thay nhiều yếu tố.

## Số đo trực tiếp từ bài nộp

Trung bình trên 1.200 query, chưa dùng nhãn:

| Bản | Doc | Đoạn | Doc có đoạn | Ký tự/đoạn trung vị | Top-20 có đoạn |
|---|---:|---:|---:|---:|---:|
| Dense 4B 0,1847 | 199,97 | 150,10 | 110,05 | 1.481 | 100% |
| nb60 v1 0,2235 | 201,83 | 86,97 | 82,97 | 1.822 | 94,15% |
| wholedoc v1 0,2065 | 200,00 | 33,82 | 33,82 | 5.866 | 99,03% |
| Fusion nb60 0,2539 | 204,87 | 88,23 | 85,01 | 1.745 | 98,00% |
| Ensemble 0,2755 | 206,53 | 76,05 | 67,03 | 3.074 | 77,13% |
| win-cov50 (chưa chấm) | 208,34 | 80,62 | 80,57 | 2.904 | 100% |

Ở bản 0,2755, các hạng doc 21–50 / 51–100 / 101–150 / 151–200 có đoạn với tỷ lệ
53,29% / 31,18% / 16,90% / 10,07%. Có trung bình 9,02 đoạn thêm trong các doc đã có
đoạn. Đó là ngân sách có thể phân bổ lại, **chưa chứng minh những đoạn này thừa**.

Wholedoc có đoạn dài nhất 235.537 ký tự; chỉ phủ 33,82 doc. Fusion nb60 có đoạn tới
21.281 ký tự; cửa sổ cố định của bản tốt nhất có tối đa 6.041 ký tự. Điều này phù hợp
với giả thuyết đoạn quá dài làm mất điểm, chưa xác định được số gold bị mất.

## Các phương pháp đã thử và ý nghĩa còn dùng được

| Nhóm | Bằng chứng trong lịch sử | Quyết định nghiên cứu |
|---|---|---|
| d50/100/200/300 | d200 tốt hơn d300 ở dense cũ; tăng doc tăng R nhưng giảm F2 | Không mặc định giảm/tăng số doc sẽ đạt mục tiêu |
| Tăng số chunk đơn | c150 tốt hơn c20/100; bị giới hạn ZIP | Có ích nhưng chưa giải quyết độ dài phù hợp |
| Cắt đôi chunk | Chunk F2 giảm 0,0972→0,0343 | Không lặp lại cắt ngắn vô điều kiện |
| Max 1 chunk/doc | Chunk F2 giảm 0,1080→0,0972 | Không coi diversity tự động có lợi |
| Cửa sổ ±1 | FINAL +~0,044 ở hai nền | Giữ làm đối chứng chắc nhất |
| nb30/60/wholedoc | 0,2184/0,2235/0,2065 trên v1 | Độ dài và phân bổ đều quan trọng |
| Corpus v2 | +0,020 FINAL cùng cấu hình hướng A | Đã dùng; không cần embed lại |
| 3 model dense | Fusion 0,2539 > Qwen8B 0,2509 | Giữ pool tốt nhất hiện có |
| PRF, query LLM | Tăng nhỏ trong hướng B, có query sinh sai | Không mở rộng thêm khi chưa tách tác động |
| ICD | 0,1966 <0,1975 | Chưa ưu tiên |
| BM25, BM25 theo từ | Riêng 0,0982/0,1017; fusion có ích nhỏ | Không tìm cách tái tạo P=0,88 BTC bằng đoán tham số |
| Quota ZH/EN | Đã giảm cả doc và chunk | Không ép suất ngôn ngữ |
| bge doc rerank | Cải thiện nhỏ trên v1/v2 | Kiểm tra đầu vào trước khi đổi model |
| Qwen4B doc rerank | 0,2176 <0,2253 (v1) | Có nghi vấn lỗi truncation prompt, chưa kết luận model yếu |
| ghigh/glow | 0,2477/0,2352 so với 0,2464 | Không tối ưu byte trước khi sửa chọn đoạn |
| min200 + minscore + gamma | Gộp 3 thay đổi, giảm điểm | Không suy nguyên nhân riêng |
| win-cov50/conf | Chưa nộp; nền 0,2539 | Giữ đối chứng; chưa dùng 2/3 lượt còn lại |

## Phát hiện mã nguồn cần đo

1. Doc rerank hiện nhìn 400 ký tự đầu + **một chunk**, cap 512 token. Chưa phải
   đánh giá toàn doc; có nguy cơ phần đầu chiếm chỗ của đoạn chứa đáp án.
2. Qwen doc rerank ghép cả prefix + nội dung + suffix rồi tokenize/truncate. Nếu
   vượt cap, đuôi assistant dùng để chấm yes/no bị cắt. Model card chính thức cắt
   nội dung trước rồi mới thêm prefix/suffix:
   https://huggingface.co/Qwen/Qwen3-Reranker-4B#using-transformers .
   Cần đo tỷ lệ và kiểm thử; chưa được dùng điểm cũ để loại toàn bộ Qwen.
3. Seed được chấm nhưng đoạn nộp là cửa sổ lớn hơn; mô hình chưa chấm đúng đoạn
   thực sự nộp. Hai cửa sổ bên trái/bên phải có thể chứa thông tin khác nhau.
4. Tăng export_pool không tự tăng ứng viên nếu danh sách nguồn đã bị cắt còn 500.

Notebook CPU `ntklinhfitus/medxir-retrieval-input-audit-20261031` **SUCCESS**, 276 giây
tính toán trên 64 query chọn bằng hash ID cố định /22.092 cặp. Kết quả:

| Đầu vào | Cap512 bị cắt | Cap1024 bị cắt | Token trung vị |
|---|---:|---:|---:|
| Query + chunk | 14,43% | 0,0045% | 426 |
| Query + lead + chunk (doc rerank hiện tại) | **43,96%** | **0,0136%** | 490 |
| Query + 2 chunk liên tiếp | 75,19% | 1,50% | 740 |
| Query + 3 chunk liên tiếp | 76,19% | **35,23%** | 840 |

57,50% seed không ở đầu doc; lead thêm trung bình114,9token. Vì vậy không thể chỉ
đổi sang chấm cửa sổ3chunk mà giữ cap512/1024. Đây là kiểm tra kỹ thuật không có
nhãn, không phải đo recall/precision. Report ở
`runs/experiments/retrieval-input-audit-20261031-v1/artifacts/retrieval_input_audit.json`.

Đã sửa Qwen prompt trong generator doc/hybrid, dựng lại notebook. 3 test prompt,
4 test offset/span và26testhybrid qua. Notebook GPU pilot16query được chuẩn bị từ
generator mới: mỗi doc tối đa2seed khác vùng, đoạn nguyên văn tối đa768BGEtokens,
pair cap2048; chấm BGE rồi Qwen4B, đo mất suffix của code cũ và throughput. Bộ lọc
unknown gap chỉ dùng seed gốc, mọi overlap không nhất quán phải lỗi. Pilot không
dựng/nộp ZIP; dùng kết quả để chốt fullrun và tối đa3lượt cần xin phép người dùng.

## Nguồn ngoài và lựa chọn hướng tiếp

- `quyen244/medial-retrieval`: có lịch sử cửa sổ so với chunk ngắn; nhiều phân tích
  cũ dùng luật trước 25/10. Không chép kết luận hay thông số như chuẩn BTC.
- `HungPhamNoob/thh`: scorer tham khảo chia cả hai độ dài nhưng tokenize whitespace;
  không phải bằng chứng tương đương scorer thật.
- `Platypus27-coder/VietMedBridge`: chấm doc từ nhiều đoạn, chấm cửa sổ, bảo toàn
  offset và MaxP. Có ích về thiết kế; tài liệu tự ghi chưa chứng minh điểm.
- Đã tìm đúng repo qua GitHub API: TranVu2005/r2ai-stage3, PromptSmithX/vibiomir-rag,
  minhtb51107/vibiomir_retrieval, Kyoyooo/ViBioMIR, tmanhococ/R2AI_MedicalRetrieval.
  Không suy tên repo tương ứng một tài khoản top leaderboard nếu chưa có chứng cứ.
- BGE Gemma là lựa chọn rerank đa ngôn ngữ có weights mở; chưa có bằng chứng nó
  thắng trên corpus này: https://huggingface.co/BAAI/bge-reranker-v2-gemma .
- `TranVu2005/r2ai-stage3/README_RELEASE.md` tự báo bản0,4101 dùng tổ hợp tín hiệu
  rerank và cửa sổ theo token; `src/r2ai/submit/window_chunks.py` giữ substring theo
  offset và mở rộng quanh đoạn có điểm tốt. Đây là tham khảo phương pháp, chưa
  tái lập hay xác nhận danh tính trên leaderboard; không tải nhãn/cache đáp án.
  https://github.com/TranVu2005/r2ai-stage3/blob/main/README_RELEASE.md

Hướng ưu tiên: dùng pool tốt nhất → dựng các đoạn liền mạch từ cùng doc → chấm
chính các đoạn này với đủ ngân sách token → lấy điểm đoạn tốt nhất hỗ trợ xếp doc
→ chọn đoạn có ích dưới ngân sách ZIP và kiểm tra offset/ID. Tách thí nghiệm chọn
đoạn khỏi thay đổi model và số doc để 3 lượt còn lại có ý nghĩa. Không cần thay
corpus/vector ở bước này. Chỉ chọn lần chạy GPU sau khi có số đo CPU và kế hoạch cặp.

Completion gate: pipeline và bài nộp đúng dữ liệu + kết quả chấm thực đạt đủ mục tiêu.
Kiểm thử xanh hoặc báo cáo cấu trúc tốt hơn chưa chứng minh đạt mục tiêu điểm.

## Tiếp tục sau pilot

- Phiên GPU version 1 đã gửi lúc 02:45 UTC, kết thúc **SUCCESS** lúc 03:17 UTC.
  Thời gian thực thi 358,9 giây, gồm 16 query / 6.302 cặp trên RTX PRO6000.
  Log/hai bảng parquet/report đã tải về và kiểm tra bằng `check_passage_pilot.py`.
  BGE: 26,55 giây /237,37 cặp/s. Qwen: 290,87 giây /21,67 cặp/s.
  111/128 prompt Qwen cũ mất suffix; helper mới giữ đủ suffix trong mọi mẫu kiểm tra.
  Không có truncation trong các cặp chấm mới. Top20 agreement theo text_sha chỉ
  28,125%; không thể suy chất lượng từ agreement hay so trực tiếp thang sigmoid.
  Dự tính 472.650 cặp fullrun: BGE33,2 phút, Qwen6,06 giờ, chưa gồm chuẩn bị/I/O/queue.
- **Vấn đề phát hiện ở pilot1:** có2.775 cửa sổ chứa gap ngoài seed nên nhánh fallback
  chỉ dùng chunk seed. Chunking theo paragraph/heading có thể không overlap, và
  chỉ các chunk riêng lẻ không cung cấp ký tự trong khoảng trống. Không tự đoán
  separator để rồi tuyên bố đoạn nguyên văn. Đã thêm `contiguous_around_anchor`:
  giữ component liên tục lớn nhất chứa seed, kiểm tra overlap của mọi component,
  chỉ dừng ở gap thật. Hai test mới kiểm tra gap xa và overlap sai ở component khác.
  Tổng8test span/output qua. Pilot2 cùng16query được gửi để đo số đoạn khôi phục,
  điểm và tốc độ sau thay đổi; chưa khởi chạy toàn bộ1200query.
- `scripts/check_passage_pilot.py` đối chiếu hai bảng candidate/score từng hàng,
  kiểm tra số query/cặp, fanout, offset, giới hạn token, điểm hữu hạn trong [0,1],
  và suffix Qwen. Hai test bao gồm điểm NaN, điểm >1, text/offset sai và fanout sai
  đều qua. Đây không phải kiểm chứng chất lượng retrieval hay thay thế việc
  đối chiếu substring với nguồn (đã assert khi dựng passage).
- Audit toàn thư mục ZIP lưu dần vào
  `artifacts/recall_research_20261031/submission_audit_all.json`. Một số bản cũ
  có chunk trùng hoàn toàn, trung bình dưới 0,1/query trong các bản đã đọc;
  không đủ căn cứ coi đây là nút thắt chính. Bản BZIP2TEST và DEFLATE ghigh có cùng
  digest doc/chunk; không phải hai thí nghiệm retrieval khác nhau. Không dùng BZIP2.
- Đối chiếu digest chunk xác nhận nb60 v1, các bản doc-rerank BGE/Qwen/both v1
  và nb60-d300 có cùng tập `(doc_id, chunk_text)` trên mọi query theo thứ tự file.
  Chúng thay danh sách doc nhưng không thay tập đoạn nộp; không dùng lịch sử này
  để kết luận Qwen/BGE chấm lại passage có hay không cải thiện chunk retrieval.

Trước full run, cần pilot xác nhận thời gian thực của từng model và tỷ lệ cắt
đầu vào Qwen (768 token BGE không bảo đảm tương đương 768 token Qwen). Nếu full run
quá dài, chọn model/phạm vi theo thời gian đo được; không chạy lại embedding.
Full run cần chia nhóm query, ghi candidate/score từng nhóm và manifest để resume;
không giữ toàn bộ text của mọi ứng viên cùng model trong RAM. Dự tính số cặp từ
pilot chỉ là ước lượng, phải đếm chính xác ở bước dựng candidate.

Ưu tiên tạo bản đối chứng BGE chấm đoạn thực nộp trên pool tốt nhất. Qwen đã sửa
prompt là nhánh so sánh trên cùng candidate; chưa mặc định cộng điểm xác suất của
hai model vì thang điểm chưa hiệu chỉnh. Nếu phối hợp, dùng thứ hạng với trọng số
toàn cục cố định. Xếp doc từ điểm các đoạn của chính doc đó; không trộn danh sách
doc của bài khác. Giữ nguyên doc_id và chỉ mở rộng nhóm fanout_safe.

Chỉ chọn ZIP sau khi kiểm tra đủ query, ID, substring nguồn, không lặp đoạn, nén
DEFLATE và dưới giới hạn dung lượng. Với 3 lượt còn lại, không nộp cả hai bản
win-cov50 cũ trước khi có kết quả pilot. Mọi điểm kỳ vọng vẫn là giả thuyết; xin
người dùng duyệt đúng file ZIP trước khi dùng một lượt chấm.

## Pipeline full đã triển khai (chưa có kết quả chấm)

`prepare_passage_corpus.py` chạy Kaggle CPU, đọc Parquet theo4096hàng và tạo index
SQLite tạm cho các text nằm trong pool. Mỗi query chỉ nạp các text cần thiết.
Nguồn gốc chunk, offset và nhóm fanout_safe được giữ; shard candidate mỗi32query có
SHA256 và danh sách query. RAM dự kiến vàiGB, disk index tối đa phần text liên quan
trong corpus; index xóa sau chuẩn bị. Mã checkpoint cho phép nối các part cũ khi
spec/modeltokenizer/implementation/sourcepool khớp. Không giả định nạp toàncorpus
vào Python RAM hay chạy tokenization corpus trênlocal.

`score_passage_corpus.py`: một model/job, model chỉ nạp một lần và chấm từng part.
BGEbatch32, Qwenbatch16, bf16SDPA; paircap4096 là giới hạn bảo vệ, **không cắt âm
thầm**: nếu thật sự vượt cap thì dừng và giữ checkpoint. Mọi part score gắn hash
candidate và passage_id, được xác minh trước nối lại. Job dựngZIP sau chấm đủ.

`assemble_passage_submission.py`: MaxP xếpdoc, tối đa200doc/query; các đoạn dưới
40BGEtokens bị coi là stub và bỏ khỏi cả doc/chunk ranking. Chọn đoạn tốt nhất
cho50docđầu, rồi bổ sung theo điểm, cho phép vùng thứhai nhưng bỏ overlap>=80%
độ dài đoạn ngắn hơn của cùngsha. Fanout chỉ trong doc đã chọn, không append doc
từ hệkhác. Cap chunk bắt đầu100; giảm5%toàncục nếu ZIP>=99.000.000byte. Mọi đoạn
copy nguyên văn từ candidate đã hash, không cắt/ghép lại sau chấm.

`run_passage_pipeline.py` điều phối local nhẹ: chờ pilot2+CPU, xác minh16query cho
kết quả candidate giống hệt, mới gửi hai job full. Sau complete tải ZIP và kiểm
tra1..1200query, hash, DEFLATE, kích thước, không lặp chunk hay doc ngoài danh sách.
State kết thúc `READY_FOR_USER_REVIEW`, **không gọi API nộp thi**. Không tự gửi lại
job khi queued hoặc sau lỗi push mơ hồ. Nếu cần chạy lại điều phối, đọc state cũ
để tránh duplicate. State/log ở `runs/passage_pipeline_20261031/`.

Các job:
- https://www.kaggle.com/code/ntklinhfitus/medxir-passages-prepare-20261031 (đã RUNNING)
- https://www.kaggle.com/code/ntklinhfitus/medxir-passages-bge-20261031 (đợi gate, chưa gửi)
- https://www.kaggle.com/code/ntklinhfitus/medxir-passages-qwen-20261031 (đợi gate, chưa gửi)

8test mới đã qua: source index/fanout1, candidate2, selection/ZIP4, gate1. TestZIP
kiểm tra cả CRC qua xuất, đọc lạiJSON, nguyên văn và từ chối input thay hash.
Chưa có GPUscore full hay ZIP thật, không suy P/R/F2 từ các test này.
