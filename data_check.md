#  Sample data 데이터 확인

## 존재하는 key
- successful_count: 크롤링에 성공한 수
- failed_count: 크롤링에 실패한 수
- total_count: 총 크롤링 시도 수
- **perfumes_data: 향수 정보**
- failed_urls: 크롤링 실패한 url

## 🌿 perfumes_data
- 총 15개의 key 존재:
  'url', 'name', 'target', 'brand_name', 'accords', 'rating_value', 'best_rating', 'rating_count', 'review_count', 'short_description', 'detailed_description', 'seasonal', 'time_of_day', 'pros', 'cons'
- 사용할 10개의 key만 추출:
  **'name', 'target', 'brand_name', 'accords', 'short_description', 'detailed_description', 'seasonal', 'time_of_day', 'pros', 'cons'**

### 🔧 향수 정보 -> instruction tuning 전처리
- Template1
  - instruction: Given any given perfume, answer which scent is the strongest. (번역: 주어진 어떤 향수의 향 중에서 가장 강한 향은 무엇인지 답하세요.)
  - input: vanilla: 100%, lavender: 56.6092%, fresh spicy: 49.2696%, cacao: 45.3949%.
  - output: vanilla
- Template2
  - instruction: Summarize the description of the entered perfume. (번역: 입력된 향수의 설명을 요약해주세요.)
  - input: Fragrance has always been a great love of mine, and I am fascinated with the power that scent can have and...
  - output: Goddess by Burberry is a Aromatic fragrance for women. This is a new fragrance...
