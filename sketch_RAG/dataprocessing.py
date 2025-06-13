import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 안전한 숫자 변환 함수들
def safe_int(val, default=0):
    try:
        return int(str(val).replace(",", ""))
    except (TypeError, ValueError):
        return default

def safe_float(val, default=0.0):
    try:
        return float(str(val).replace("%", ""))
    except (TypeError, ValueError):
        return default

def create_llm_text(entry):
    name = entry.get("name", "Unknown")
    brand = entry.get("brand_name", "Unknown")
    target = entry.get("target", "unisex")
    rating = safe_float(entry.get("rating_value", 0))
    rating_count = safe_int(entry.get("rating_count", "0"))
    review_count = safe_int(entry.get("review_count", "0"))

    accords = sorted(entry.get("accords", []), key=lambda x: safe_float(x.get("strength", "0")), reverse=True)
    top_accords = ", ".join([f"{a['accord']} ({a['strength']})" for a in accords[:3]])

    seasonal = [
        k for k, v in entry.get("seasonal", {}).items()
        if isinstance(v, str) and safe_float(v) >= 50
    ]
    time_of_day = [
        k for k, v in entry.get("time_of_day", {}).items()
        if isinstance(v, str) and safe_float(v) >= 50
    ]

    season_str = ", ".join(seasonal)
    time_str = ", ".join(time_of_day)

    text = f"{name} by {brand} is a {target} fragrance with dominant accords of {top_accords}. "
    text += f"It holds a rating of {rating} out of 5 based on {rating_count} ratings and {review_count} reviews. "

    if entry.get("short_description"):
        text += entry["short_description"] + " "

    if season_str:
        text += f"Best suited for {season_str}"
        if time_str:
            text += f", especially at {time_str}."
        else:
            text += ". "

    if entry.get("detailed_description"):
        text += "\n\n" + entry["detailed_description"]

    return text.strip()

def create_metadata(entry):
    accords = sorted(entry.get("accords", []), key=lambda x: safe_float(x.get("strength", "0")), reverse=True)
    accord_list = [a["accord"] for a in accords]
    accord_strengths = {
        a["accord"]: round(safe_float(a.get("strength", "0")) / 100, 4)
        for a in accords
    }

    seasonal = [
        k for k, v in entry.get("seasonal", {}).items()
        if isinstance(v, str) and safe_float(v) >= 50
    ]
    time_of_day = [
        k for k, v in entry.get("time_of_day", {}).items()
        if isinstance(v, str) and safe_float(v) >= 50
    ]

    return {
        "name": entry.get("name", ""),
        "brand_name": entry.get("brand_name", ""),
        "target": entry.get("target", ""),
        "accords": accord_list,
        "accord_strengths": accord_strengths,
        "rating_value": safe_float(entry.get("rating_value", 0)),
        "review_count": safe_int(entry.get("review_count", 0)),
        "rating_count": safe_int(entry.get("rating_count", 0)),
        "seasonal": seasonal,
        "time_of_day": time_of_day
    }

def convert_perfume_json_to_hybrid_jsonl(input_paths, output_path):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=30)
    all_entries = []

    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_entries.extend(data.get("perfumes_data", []))

    total_docs = 0
    total_chunks = 0

    with open(output_path, "w", encoding="utf-8") as f_out:
        for idx, entry in enumerate(all_entries):
            text_full = create_llm_text(entry)
            text_chunks = text_splitter.split_text(text_full)
            metadata = create_metadata(entry)

            record = {
                "perfume_id": f"p_{idx:05}",
                "metadata": metadata,
                "text_full": text_full,
                "text_chunks": text_chunks
            }
            json.dump(record, f_out, ensure_ascii=False)
            f_out.write("\n")

            total_docs += 1
            total_chunks += len(text_chunks)

    print(f"✅ JSONL 생성 완료: {output_path}")
    print(f"총 문서 수: {total_docs}")
    print(f"총 청크 수: {total_chunks}")

if __name__ == "__main__":
    input_files = [
        "../data/raw_data/perfumes_complete_man_data.json",
        "../data/raw_data/perfumes_complete_woman_data.json"
    ]
    output_file = "./perfumes_rag.jsonl"
    convert_perfume_json_to_hybrid_jsonl(input_files, output_file)

"""
✅ JSONL 생성 완료: ./perfumes_rag.jsonl
총 문서 수: 1737
총 청크 수: 7085
"""