# Kế hoạch ba lượt còn lại — 31/10/2026

## Cập nhật sau khi chấm — thay thế khuyến nghị phía dưới

Người dùng đã nộp passage-BGE-d200: FINAL0,2609, DocF2 0,3147, ChunkF2 0,2072;
DocP/R 0,161/0,5408, ChunkP/R 0,1631/0,2746. Cả hai F2 giảm so với0,2755.
Không tiếp tục mặc định nộp các bản samechunks d160/d120.

Bạn trong nhóm có `medxir-v2ens-bgefull-drr-d200-c120-.zip`, FINAL0,2916,
DocF2 0,3206, ChunkF2 0,2627, DocP/R 0,1593/0,565, ChunkP/R 0,1901/0,361.
ZIP được kiểm tra thực tế dùng BZIP2(method12). **Người dùng đã chấp nhận BZIP2**,
thay thế yêu cầu chỉ dùng DEFLATE trước đây. Đây là baseline điểm tốt hơn, nhưng không
phải ablation riêng model vì số chunk và cách nén khác nhau.

Ưu tiên mới: lấy code/manifest của bản0,2916, xác định pool và cách chọn cửa sổ,
rồi cải thiện trên nền này. Pull remote hiện vẫn ed1a4ac, chưa có code mới.
Qwen passage đang RUNNING; giữ làm phép so model, không mặc định sẽ nộp khi xong.
Chọn lượt nộp tiếp dựa trên baseline mới. Giới hạn file vẫn dưới100MB; việc đổi
cách nén chỉ có ích về điểm khi tận dụng phần dung lượng để chọn thêm nội dung.
Exporter đã có tùy chọn BZIP2, xuất tên/manifest riêng để giữ bản DEFLATE đã chấm.
Job Qwen đang chạy dùng snapshot cũ nên đầu ra đầu tiên vẫn là DEFLATE; nếu cần
đổi ngân sách/số đoạn thì xử lý lại từ điểm đã lưu, không chạy GPU lại.

## Trạng thái xác minh

- Pilot2 SUCCESS: giữ lại được ngữ cảnh liên tục ở 1.719 cửa sổ trước đó rơi về seed.
  Đủ 6.302 cặp/16 query; Qwen giữ suffix; không cắt đầu vào. Đây là kiểm tra kỹ thuật.
- Chuẩn bị CPU SUCCESS: 466.571 đoạn ứng viên cho 1.200 query. Đối chiếu 16 query
  với pilot2 khớp cả văn bản, offset, seed và fanout trước khi gửi job đầy đủ.
- BGE full SUCCESS: chấm đủ 466.571 cặp, không cắt đầu vào, khoảng 32,3 phút trước
  bước xuất ZIP. ZIP 93.572.363 byte, đủ query ID 1..1200, DEFLATE, không trùng
  đoạn hay có doc của chunk nằm ngoài danh sách doc. Hash tải về khớp manifest.
- Qwen full đã gửi, lần kiểm tra 06:49 UTC vẫn QUEUED. Ước tính cần khoảng 6–8 giờ
  kể từ khi được GPU, không tính thời gian chờ. Không gửi lại job đang queued.
- API BTC vừa kiểm tra vẫn ghi public kết thúc 16:59 UTC, tức 23:59 Việt Nam ngày
  31/10. Private từ 1/11 đến 4/11, tổng 5 lượt. Không tự dùng lượt nộp.

## Bản nên chấm trước

`R2/submission/passages-20261031/medxir-passage-bge-d200.zip`

| Chỉ số cấu trúc | Bản 0,2755 | BGE passage mới |
|---|---:|---:|
| Doc/query trung bình | 206,53 | 199,99 |
| Đoạn/query trung bình | 76,05 | 100 |
| Doc có đoạn/query | 67,03 | 88,71 |
| Top20 doc có đoạn | 77,13% | 100% |
| Ký tự/đoạn trung vị | 3.074 | 2.637 |

Các số này không phải relevance metrics. Độ phủ tăng không bảo đảm Recall/F2 tăng.
Mốc chấm hiện tại: FINAL 0,2755; Doc F2 0,3197; Chunk F2 0,2314;
Doc P/R 0,1611/0,556; Chunk P/R 0,2194/0,2842.

## Thứ tự quyết định

1. **Một lượt cho BGE d200** nếu người dùng duyệt. Cần đủ FINAL, Doc/Chunk F2,
   Precision và Recall; không chỉ FINAL. Mục tiêu là kiểm tra cả pipeline mới.
2. **Chọn theo kết quả đầu, không cố định trước hai lượt tiếp.**
   - Nếu chunk tăng nhưng Doc P còn thấp, ưu tiên cutoff d160. Nếu cần thay đổi
     mạnh hơn và chấp nhận rủi ro giảm Doc R, có d120. Cả hai giữ nguyên tập chunk;
     đây là kiểm tra riêng tác động số doc, không trộn đầu ra hai hệ khác nhau.
   - Nếu BGE không tăng và Qwen đã xong, Qwen là đối chứng đổi model trên cùng
     bộ đoạn và quy tắc xuất. Không suy BGE thắng từ review 16 query.
   - Nếu Qwen còn chờ quá lâu, không bỏ phí thời gian phản hồi public để chờ nó.
     Dùng nhánh CPU đã chuẩn bị nếu điểm BGE cho thấy có lý do cụ thể để thử.
3. **Lượt cuối dành cho biến thể có căn cứ sau phản hồi.** Có thể là cutoff trên
   nhánh tốt hơn, hoặc RRF giữa thứ hạng BGE/Qwen nếu hai model bổ sung nhau. RRF
   dùng lại điểm, không cần chạy GPU mới. Chưa coi agreement thấp là bằng chứng
   fusion thắng; không cộng sigmoid thô giữa model và không chỉnh theo query ID.

## Vì sao chuẩn bị d160/d120

Audit đủ 1.200 query xác nhận mọi đoạn của BGE mới thuộc top100 doc. Vì vậy cắt
d200 xuống d160/d120 vẫn giữ được toàn bộ chunk và quan hệ doc–chunk. Script
`build_passage_doc_depth_variants.py` kiểm tra điều kiện này từng query, đối chiếu
hash payload chunk trước/sau, CRC ZIP, số query và dung lượng. Đây là thao tác
xuất top-k nhỏ, không chạy corpus/model ở local. Các bản này chưa được chấm.

Doc P có thể tăng khi giữ ít doc, nhưng Recall giảm có thể làm F2 giảm. Chỉ đo
thực tế mới quyết định; không suy số gold/query từ tỷ lệ các chỉ số macro.

## Hướng lớn hơn cho giai đoạn sau

Nếu những bản trên chưa đạt mục tiêu, ưu tiên tạo tập dev có nhãn độc lập và
đo độ bao phủ của pool. Sau đó mới quyết định mở rộng pool bằng BM25 theo từ,
tiêu đề hoặc nguồn truy hồi bổ sung. Reranker không cứu được doc không có trong
pool. Nếu pool đã có doc đúng nhưng hạng thấp, ưu tiên doc ranking/calibration;
nếu thiếu doc đúng, ưu tiên retrieval. Các kết luận này cần nhãn, không được
thay bằng cảm giác từ vài mẫu. Không mở thêm embedding/model mới ngay trước hạn
chỉ vì model lớn hơn. Mục tiêu FINAL >0,40 vẫn chưa có bằng chứng đạt được.

Lịch BTC: https://leaderboard.aiguru.com.vn/api/competitions/15/
