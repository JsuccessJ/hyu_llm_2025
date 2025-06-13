#  Sample data 데이터 확인 and 전처리 예시

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
- **Template1** (가장 높은 accords를 파악하기 위한 템플릿)
  - **instruction:** Answer which of the given scents is the strongest. (번역: 주어진 향 중에서 가장 강한 향은 무엇인지 답하세요.)
  - **input:** vanilla: 100%, lavender: 56.6092%, fresh spicy: 49.2696%, cacao: 45.3949%.
  - **output:** vanilla
    
- **Template2** (향수에 대한 정보를 잘 이해하기 위한 템플릿1)
  - **instruction:** Summarize the description of the entered perfume. (번역: 입력된 향수의 설명을 요약해주세요.)
  - **input:** Fragrance has always been a great love of mine, and I am fascinated with the power that scent can have and...
  - **output:** Goddess by Burberry is a Aromatic fragrance for women. This is a new fragrance...
    
- **Template3** (가장 높은 선호 계절을 파악하기 위한 템플릿)
  - **instruction:** Given a numerical representation of the seasons for which a perfume is appropriate, answer which season is the most appropriate. (번역: 어떤 향수가 어울리는 계절을 수치로 표시한 정보가 주어졌을 때, 가장 어울리는 계절은 무엇인지 답하세요.)
  - **input:** spring: 63.2391%, summer: 41.7738%, fall: 100%, winter: 91.0797%.
  - **output:** fall
    
- **Template4** (적합한 사용 시간대를 파악하기 위한 템플릿)
  - **instruction:** Given a numerical representation of the time of day when a perfume is appropriate to wear, answer the appropriate time of day. (번역: 어떤 향수가 사용하기 적합한 시간대를 수치로 표시한 정보가 주어졌을 때, 적합한 시간대를 답하세요.)
  - **input:** day: 88.8432%, night: 74.4216%.
  - **output:** day
    
- **Template5** (향수에 대한 정보를 잘 이해하기 위한 템플릿2)
  - **instruction:** Determine the relationship between the following given sentences. (번역: 다음 주어진 문장간의 관계를 파악하세요.)
  - **input:** Fragrance has always been a great love of mine, and I am fascinated with the power that scent can have and... Goddess by Burberry is a Aromatic fragrance for women. This is a new fragrance...
  - **output:** Summarization
