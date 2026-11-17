# Đối chiếu BGE và Qwen reranker — 31/10/2026

**Chưa có bằng chứng model nào thắng trên điểm thi.** Tạm giữ BGE làm bản đối chứng vì nhanh hơn khoảng 11 lần. Qwen4B đã sửa prompt vẫn là nhánh cần so sánh. Không suy chất lượng từ số tham số hoặc giá trị sigmoid giữa hai model.

## Dữ liệu hiện có

Pilot1: 16 query chọn bằng hash ID, 6.302 cặp cùng đầu vào cho cả hai model. Sau bỏ đoạn dưới 40 BGE token như full run, còn 6.184 cặp. Điểm sắp giảm dần, phá hòa theo seed_rank cố định. Cả hai không bị cắt đầu vào ở giới hạn 2.048 token. Qwen dùng helper giữ nguyên prefix/suffix. Mẫu này không có nhãn chuẩn.

| Chỉ số chẩn đoán | BGE-v2-m3 | Qwen3-Reranker-4B |
|---|---:|---:|
| Cặp/giây, phép đo gồm nạp model | 237,37 | 21,67 |
| Số token đoạn top1 trung vị, cùng tokenizer BGE | 428 | 280,5 |
| Top20 text có chữ Hán chiếm ưu thế | 7,81% | 39,38% |
| Số đoạn đồng điểm tại ngưỡng top20 trung bình | 1,25 | 2,31 |

Nhận diện chữ Hán là heuristic: ít nhất 20 ký tự CJK và 20% ký tự alphabetic. Đây không phải metadata ngôn ngữ corpus. Qwen chọn nhiều bài Trung phù hợp ở một số query, nhưng cũng chọn bài gần nghĩa mà lệch ý; không vì vậy ép quota VI.

Hai model cùng nguồn text top1 ở 2/16 query (12,5%). Giao nhau trung bình ở top20 là 29,38%, top80 là 45,39%, top200 là 68,69%, tính theo text_sha256, không nhân fanout. Số 28,1% trước đây dùng cách lọc/phá hòa khác. Agreement không phải gold metric và không chứng minh model nào đúng hơn hoặc fusion sẽ thắng.

## Đọc nội dung top1 của đủ 16 query

Codex đọc cặp A/B với tên model ẩn, chấm mức độ trả lời 0–3 rồi mở mapping. Một số ví dụ từng được xem trước đó: đây không phải thí nghiệm mù độc lập. Review đánh giá relevance, không đánh giá tính đúng của hướng dẫn y khoa.

**BGE nhỉnh hơn ở 4 query, Qwen ở 3 query, hòa 9 query.** Mẫu quá nhỏ để chốt model thắng. Hòa gồm cả trường hợp hai model cùng chưa trả lời được ý chính.

- Nhiệt độ hạ đột ngột khi sốt Dengue: Qwen chọn đoạn trực tiếp nói về hiện tượng này; BGE chọn bài biến chứng nói về tụt huyết áp, lệch trọng tâm.
- Viêm cổ tử cung không đặc hiệu: BGE giữ đúng vị trí bệnh; Qwen chọn viêm âm đạo.
- Đau chân/teo cơ và điều trị: Qwen top1 chỉ có câu hỏi của người bệnh; BGE có đoạn nêu nguyên nhân và các nhóm điều trị.
- Nguyên nhân thiếu D3: Qwen chọn bài Trung liệt kê nguyên nhân chung; BGE chọn bài riêng trẻ sơ sinh dù query không giới hạn tuổi.
- Một số query cả hai chỉ khớp chủ đề, bỏ lỡ con số hoặc triệu chứng cần giải thích.

## Kiểm tra thêm tài liệu tiếng Anh

Candidate có 645 đoạn nhận diện English-like qua stopwords và không có CJK; đây cũng chỉ là heuristic. Đọc một số abstract và so cùng nội dung ở hai thứ hạng:

| Ví dụ nguồn EN | Hạng BGE | Hạng Qwen |
|---|---:|---:|
| Thiếu D3 do ít tiếp xúc ánh sáng | 3 | 9 |
| Beta-Crosslaps theo tuổi/giới | 31 | 7 |
| Ý nghĩa xét nghiệm ANA | 36 | 25 |
| Kết quả G6PD có nhắc 0,3 U/gHb | 24 | 19 |

Qwen kéo lên một số nguồn EN đúng thuật ngữ, nhưng abstract chưa chắc trả lời đầy đủ ngữ cảnh query hay trúng đoạn vàng. Chưa đủ đánh giá P/R/F2 riêng từng ngôn ngữ.

## Cách quyết định tiếp

Giữ cùng corpus, pool, query, đoạn, doc_depth, quy tắc chọn đoạn và giới hạn ZIP giữa hai nhánh. Không cộng điểm thô của hai model hay chỉnh tham số theo ID. Full run đã chuẩn bị cả BGE và Qwen. Cùng giới hạn byte có thể cho ra số chunk khác nhau giữa hai model: cần phân biệt hiệu quả cả pipeline dưới giới hạn ZIP với ablation cố định cùng số chunk. Chỉ kết quả chấm có nhãn mới xác nhận model nào tốt hơn theo mục tiêu cuộc thi.

Pilot2 còn chờ GPU để xác nhận sau sửa gap. Review pilot1 không phải kết quả của code mới. Không dùng các grade trên để sửa riêng từng query.

Artifacts trong artifacts/recall_research_20261031/reranker_comparison/: pilot1_diagnostics.json, blind_top1.json, blind_mapping.json, manual_top1_review.json. Script tái lập thống kê: scripts/compare_passage_rerankers.py.
