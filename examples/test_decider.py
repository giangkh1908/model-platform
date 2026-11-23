"""Thử Strands Decider với 12 tình huống (tiếng Việt và tiếng Anh) và đo tốc độ.

Chạy khi model đang bật:  python examples/test_decider.py
"""

import json
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/systemone"

CASES = [
    ("Hỗ trợ khách hàng (VI)",
     "Tôi đã chuyển khoản 3 lần mà đơn hàng vẫn báo chưa thanh toán. Làm ăn kiểu gì vậy?!",
     {"team": ("choice", "Bộ phận nào nên xử lý?", ["thanh toán", "kỹ thuật", "bán hàng", "vận chuyển"]),
      "urgent": ("noul", "Khách có đang rất gấp hoặc bức xúc không?", None),
      "mood": ("score", "Mức độ bực tức của khách?", ["bình tĩnh", "khó chịu", "rất tức giận"])}),
    ("Customer support (EN)",
     "Hi, could you tell me if the Pro plan includes API access? Thinking of upgrading next month.",
     {"team": ("choice", "Which team should handle this?", ["billing", "technical", "sales", "shipping"]),
      "urgent": ("noul", "Is this urgent?", None)}),
    ("Chọn công cụ cho agent",
     "Người dùng: Thời tiết ở Hà Nội ngày mai thế nào?",
     {"tool": ("choice", "Agent nên gọi công cụ nào?", ["tìm kiếm web", "dự báo thời tiết", "máy tính", "lịch"])}),
    ("Chọn công cụ cho agent (EN)",
     "User: What is 17% of 2,340 plus 125?",
     {"tool": ("choice", "Which tool should the agent call?", ["web_search", "weather", "calculator", "calendar"])}),
    ("Định tuyến model",
     "Viết cho tôi một bài phân tích chi tiết 3000 chữ về tác động của AI tới thị trường lao động Việt Nam, có số liệu và trích dẫn.",
     {"route": ("choice", "Nên gửi yêu cầu này cho model nào?",
                ["model nhỏ (nhanh, rẻ)", "model lớn (mạnh, đắt)"])}),
    ("Định tuyến model (dễ)",
     "Dịch giúp tôi chữ 'cảm ơn' sang tiếng Anh.",
     {"route": ("choice", "Nên gửi yêu cầu này cho model nào?",
                ["model nhỏ (nhanh, rẻ)", "model lớn (mạnh, đắt)"])}),
    ("Kiểm duyệt nội dung",
     "Mày mà còn đăng bài nữa thì tao tìm đến tận nhà đấy.",
     {"threat": ("noul", "Tin nhắn có chứa đe doạ bạo lực không?", None),
      "spam": ("noul", "Đây có phải tin rác quảng cáo không?", None)}),
    ("Phát hiện spam",
     "CHÚC MỪNG! Bạn đã trúng iPhone 17. Bấm vào link bit.ly/abc123 để nhận quà trong 24h!",
     {"spam": ("noul", "Đây có phải tin nhắn lừa đảo/spam không?", None)}),
    ("Cảm xúc đánh giá sản phẩm",
     "Pin trâu, màn đẹp nhưng camera chụp tối hơi nhoè. Nhìn chung vẫn đáng tiền.",
     {"sentiment": ("score", "Đánh giá chung của người viết?", ["rất tệ", "tệ", "trung bình", "tốt", "rất tốt"])}),
    ("Kiểm tra câu trả lời",
     "Câu hỏi: Thủ đô của Úc là gì?\nTrả lời: Thủ đô của Úc là Sydney.",
     {"correct": ("noul", "Câu trả lời có đúng không?", None)}),
    ("Kiểm tra câu trả lời (đúng)",
     "Câu hỏi: Nước sôi ở bao nhiêu độ C ở áp suất thường?\nTrả lời: 100 độ C.",
     {"correct": ("noul", "Câu trả lời có đúng không?", None)}),
    ("Áp dụng chính sách",
     "Chính sách: Hoàn tiền nếu yêu cầu trong vòng 30 ngày và sản phẩm chưa mở hộp.\n"
     "Trường hợp: Khách mua 12 ngày trước, đã mở hộp dùng thử, muốn trả hàng.",
     {"refund": ("noul", "Theo chính sách, khách có được hoàn tiền không?", None)}),
]


def to_request(state, qs):
    questions = {}
    for key, (kind, text, opts) in qs.items():
        if kind == "noul":
            questions[key] = {"type": "noul", "instructions": text}
        elif kind == "choice":
            questions[key] = {"type": "choice", "instructions": text, "criteria": {o: None for o in opts}}
        else:
            questions[key] = {"type": "score", "instructions": text, "criteria": opts}
    return {"state": state, "questions": questions}


def post(body):
    req = urllib.request.Request(URL, json.dumps(body).encode("utf-8"), {"content-type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        res = json.load(r)
    return res, (time.perf_counter() - t) * 1000


def show(a):
    if a["type"] == "noul":
        return f"{'CÓ' if a['noul'] >= 0.5 else 'KHÔNG'} (P(có) = {a['noul']:.2f})"
    if a["type"] == "choice":
        return f"{a['choice']} (tin cậy {a['confidence']:.2f})"
    top = max(a["probabilities"], key=a["probabilities"].get)
    return f"{a['legend'].get(top, top)} (điểm {a['score']:.2f}, tin cậy {a['confidence']:.2f})"


def main():
    post(to_request("warmup", {"x": ("noul", "ok?", None)}))
    total_q, total_ms = 0, 0.0
    for i, (name, state, qs) in enumerate(CASES, 1):
        res, ms = post(to_request(state, qs))
        total_q += len(qs)
        total_ms += ms
        print(f"\n{i:2}. {name}  [{ms:.0f} ms, {len(qs)} câu hỏi]")
        print(f"    {state.splitlines()[-1][:90]}")
        for key, (_, text, _) in qs.items():
            print(f"    - {text} -> {show(res['answers'][key])}")
    print(f"\nTổng: {len(CASES)} văn bản, {total_q} câu hỏi, {total_ms:.0f} ms "
          f"(trung bình {total_ms / len(CASES):.0f} ms mỗi văn bản)")

    # Gộp 15 câu hỏi về cùng một văn bản vào một request
    state = CASES[0][1]
    many = {f"q{i}": ("noul", q, None) for i, q in enumerate([
        "Khách có bức xúc không?", "Có liên quan đến thanh toán không?", "Có liên quan đến vận chuyển không?",
        "Khách có dọa huỷ đơn không?", "Có cần chuyển cho người thật xử lý không?", "Có phải câu hỏi về giá không?",
        "Có lỗi kỹ thuật trên website không?", "Khách có dùng từ ngữ thô tục không?", "Có yêu cầu hoàn tiền không?",
        "Có phải khách hàng mới không?", "Có nhắc tới số lần chuyển khoản không?", "Có cần xử lý trong hôm nay không?",
        "Khách có khen sản phẩm không?", "Có phải tin rác không?", "Có hỏi về khuyến mãi không?"])}
    res, ms = post(to_request(state, many))
    print(f"\n15 câu hỏi trong 1 request: {ms:.0f} ms")
    for key, (_, text, _) in many.items():
        print(f"    - {text} -> {show(res['answers'][key])}")


if __name__ == "__main__":
    main()
