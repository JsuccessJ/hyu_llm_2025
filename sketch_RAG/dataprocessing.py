import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 입력 JSON 파일들
input_files = [
    "../data/raw_data/perfumes_complete_man_data.json",
    "../data/raw_data/perfumes_complete_woman_data.json"
]

# 텍스트 청크 분할 도구
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=30
)

def preprocess_entries(entries, gender):
    docs = []

    for idx, entry in enumerate(entries):
        brand = entry.get("brand_name", "")
        name = entry.get("name", "")
        short_desc = entry.get("short_description", "")
        rating_value = entry.get("rating_value", "N/A")
        rating_count = entry.get("rating_count", "0")
        review_count = entry.get("review_count", "0")
        seasonal = entry.get("seasonal", {})

        # 상위 5개 accords 가져오기 (각 accord는 딕셔너리로 유지)
        accords_raw = entry.get("accords", [])[:5]
        accords_text = ", ".join([f"{a['accord']} ({a['strength']})" for a in accords_raw])

        # 검색용 본문 구성
        full_text = f"""{brand} - {name}

Short Description:
{short_desc}

Main Accords:
{accords_text}

Rating: {rating_value} ({rating_count} ratings, {review_count} reviews)
""".strip()

        chunks = text_splitter.split_text(full_text)

        for i, chunk in enumerate(chunks):
            docs.append({
                "perfume_id": f"{gender[:1]}_{idx:04}",
                "brand_name": brand,
                "accords": accords_raw,
                "rating_value": rating_value,
                "rating_count": rating_count,
                "review_count": review_count,
                "short_description": short_desc,
                "seasonal": seasonal,
                "gender": gender,
                "chunk_index": i,
                "text_chunk": chunk
            })

    return docs

# 전체 청크 리스트
all_chunks = []

# 각 JSON 파일 순회
for file_path in input_files:
    gender = "male" if "men" in file_path else "female"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        entries = data.get("perfumes_data", [])
        chunks = preprocess_entries(entries, gender)
        all_chunks.extend(chunks)

# 결과를 JSONL로 저장
output_path = Path("./perfumes_docs.jsonl")
with output_path.open("w", encoding="utf-8") as f:
    for chunk in all_chunks:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print(f"✅ 총 청크 수: {len(all_chunks)}")
# ✅ 총 청크 수: 5609

'''
# 출력 JSONL 예시 (1개 청크)
{
  "perfume_id": "m_0000",
  "brand_name": "Parfums de Marly",
  "accords": [
    {"accord": "sweet", "strength": "100%"},
    {"accord": "vanilla", "strength": "97.571%"},
    {"accord": "warm spicy", "strength": "89.562%"},
    {"accord": "cinnamon", "strength": "68.0724%"},
    {"accord": "musky", "strength": "65.573%"}
  ],
  "rating_value": "4.40",
  "rating_count": "6,655",
  "review_count": "1106",
  "short_description": "Althaïr by Parfums de Marly is a Oriental Vanilla fragrance for men...",
  "seasonal": {
    "winter": "100%",
    "spring": "33.526%",
    "summer": "14.1137%",
    "fall": "90.7274%"
  },
  "gender": "male",
  "chunk_index": 0,
  "text_chunk": "Parfums de Marly - Althaïr Parfums de Marly for men\n\nShort Description:\nAlthaïr by Parfums de Marly is a Oriental Vanilla fragrance for men...\n\nMain Accords:\nsweet (100%), vanilla (97.571%), warm spicy (89.562%), cinnamon (68.0724%), musky (65.573%)\n\nRating: 4.40 (6,655 ratings, 1106 reviews)"
}

# text_chunk
f"""{brand} - {name}

Short Description:
{short_desc}

Main Accords:
{accords_text}

Rating: {rating_value} ({rating_count} ratings, {review_count} reviews)
'''