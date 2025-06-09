import json
import time
import random
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium_stealth import stealth

def scrape_perfume(url, index=None):
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--remote-debugging-port=0")  # 포트 충돌 방지
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(options=options)

    stealth(driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
    )

    wait = WebDriverWait(driver, 7)

    try:
        driver.get(url)

        if "잠시만 기다리십시오" in driver.page_source or "Just a moment..." in driver.page_source:
            raise Exception("⚠️ Cloudflare 차단 페이지 감지됨.")

        wait.until(EC.presence_of_element_located((By.ID, "main-content")))
        data = {'url': url}

        name_elem = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'h1[itemprop="name"]')))
        data['name'] = name_elem.text.strip().split('\n')[0]
        try:
            small_tag = name_elem.find_element(By.TAG_NAME, 'small')
            data['target'] = small_tag.text.strip()
        except:
            data['target'] = None

        try:
            brand_elem = driver.find_element(By.CSS_SELECTOR, 'p[itemprop="brand"] a[itemprop="url"]')
            data['brand_name'] = brand_elem.find_element(By.CSS_SELECTOR, 'span[itemprop="name"]').text.strip()
        except:
            data['brand_name'] = None

        data['accords'] = []
        for bar in driver.find_elements(By.CSS_SELECTOR, 'div.accord-bar'):
            accord = bar.text.strip()
            style = bar.get_attribute('style') or ""
            strength = style.split('width:')[-1].split(';')[0].strip() if "width:" in style else None
            data['accords'].append({'accord': accord, 'strength': strength})

        for field, selector in [
            ('rating_value', 'span[itemprop="ratingValue"]'),
            ('best_rating', 'span[itemprop="bestRating"]'),
            ('rating_count', 'span[itemprop="ratingCount"]')
        ]:
            try:
                data[field] = driver.find_element(By.CSS_SELECTOR, selector).text.strip()
            except:
                data[field] = None

        try:
            data['review_count'] = driver.find_element(By.CSS_SELECTOR, 'meta[itemprop="reviewCount"]').get_attribute('content').strip()
        except:
            data['review_count'] = None

        try:
            data['short_description'] = driver.find_element(By.CSS_SELECTOR, 'div[itemprop="description"] > p:first-of-type').text.strip()
        except:
            data['short_description'] = None

        detailed = [p.text.strip() for p in driver.find_elements(By.CSS_SELECTOR, 'div.fragrantica-blockquote > p') if p.text.strip()]
        data['detailed_description'] = "\n".join(detailed) if detailed else None

        for cat, labels in [('seasonal', ['winter', 'spring', 'summer', 'fall']), ('time_of_day', ['day', 'night'])]:
            data[cat] = {}
            for lbl in labels:
                try:
                    legend = driver.find_element(By.XPATH, f"//span[@class='vote-button-legend' and normalize-space(text())='{lbl}']")
                    colored = legend.find_element(By.XPATH, "../following-sibling::div[contains(@class,'voting-small-chart-size')]/div/div")
                    style = colored.get_attribute('style') or ""
                    pct = style.split('width:')[-1].split(';')[0].strip() if "width:" in style else None
                except:
                    pct = None
                data[cat][lbl] = pct

        for field in ['Pros', 'Cons']:
            try:
                header = driver.find_element(By.XPATH, f"//h4[normalize-space(text())='{field}']")
                section = header.find_element(By.XPATH, "./ancestor::div[contains(@class,'small-12') and contains(@class,'medium-6')]")
                spans = section.find_elements(By.XPATH, ".//span")
                items = [s.text.strip() for s in spans if s.text.strip() and not s.text.strip().replace(",", "").isdigit()]
                data[field.lower()] = items
            except:
                data[field.lower()] = []

        return data

    finally:
        if index is not None and "잠시만 기다리십시오" in driver.page_source:
            with open(f"blocked_html_{index}.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
        driver.quit()

def load_perfume_links(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f).get('perfume_links', [])
    except:
        return []

def get_processed_urls(result_file):
    """기존 결과 파일에서 이미 처리된(성공한) URL들을 가져옵니다."""
    processed_urls = set()
    
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
                
            # 성공한 URL들 추가
            for perfume in result.get('perfumes_data', []):
                if 'url' in perfume:
                    processed_urls.add(perfume['url'])
                    
            print(f"✅ 이미 성공적으로 처리된 URL: {len(processed_urls)}개")
            
        except Exception as e:
            print(f"⚠️ 결과 파일 읽기 오류: {e}")
    else:
        print("📂 기존 결과 파일이 없습니다. 처음부터 시작합니다.")
    
    return processed_urls

def get_remaining_urls(all_urls, processed_urls):
    """전체 URL에서 아직 처리되지 않은 URL들만 필터링합니다."""
    remaining_urls = []
    
    for i, url in enumerate(all_urls):
        if url not in processed_urls:
            remaining_urls.append((i + 1, url))  # 원래 인덱스와 함께 저장
    
    return remaining_urls

def save_single_result(data, filename):
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                result = json.load(f)
        except:
            result = {'successful_count': 0, 'failed_count': 0, 'total_count': 0, 'perfumes_data': [], 'failed_urls': []}
    else:
        result = {'successful_count': 0, 'failed_count': 0, 'total_count': 0, 'perfumes_data': [], 'failed_urls': []}

    if 'error' in data:
        result['failed_urls'].append(data)
        result['failed_count'] += 1
    else:
        result['perfumes_data'].append(data)
        result['successful_count'] += 1
    result['total_count'] = result['successful_count'] + result['failed_count']

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 설정
    links_file = "perfume_links_man.json"  # 원본 링크 파일
    result_file = "perfumes_complete_man_data.json"  # 결과 저장 파일
    MAX_RETRIES = 2  # 실패 시 재시도 횟수
    
    print("🔄 크롤링 재시작 준비 중...")
    
    # 1. 전체 URL 로드
    all_urls = load_perfume_links(links_file)
    print(f"📋 전체 URL 개수: {len(all_urls)}개")
    
    # 2. 이미 처리된 URL 확인
    processed_urls = get_processed_urls(result_file)
    
    # 3. 남은 URL들만 필터링
    remaining_urls = get_remaining_urls(all_urls, processed_urls)
    
    if not remaining_urls:
        print("🎉 모든 URL이 이미 처리되었습니다!")
        exit()
    
    print(f"⏭️ 남은 URL 개수: {len(remaining_urls)}개")
    print(f"🚀 {remaining_urls[0][0]}번째 URL부터 재시작합니다.")
    
    # 4. 남은 URL들 크롤링
    for original_index, url in remaining_urls:
        for i in range(MAX_RETRIES + 1):
            print(f"[{original_index}/{len(all_urls)}] 크롤링 중: {url} (시도 {i+1}/{MAX_RETRIES+1})")
            try:
                result = scrape_perfume(url, index=original_index)
                save_single_result(result, result_file)
                print(f"✅ 성공: {result.get('name')} [{result.get('brand_name')}]")
                break # 성공 시 재시도 루프 탈출
            except Exception as e:
                print(f"❌ 실패: {e}")
                if i < MAX_RETRIES:
                    wait_time = 300  # 5분
                    print(f"⏳ {wait_time}초 후 재시도합니다... (재시도 {i+1}/{MAX_RETRIES})")
                    time.sleep(wait_time)
                else:
                    print(f"🚫 최대 재시도 횟수({MAX_RETRIES+1})를 초과했습니다. 이 URL은 실패로 기록합니다.")
                    error = {'url': url, 'error': f"최대 재시도 후에도 실패: {str(e)}"}
                    save_single_result(error, result_file)

        # 마지막 URL이 아니면 대기
        if (original_index, url) != remaining_urls[-1]:
            sleep_time = random.uniform(5, 8)
            print(f"💤 {sleep_time:.1f}초 대기 중...")
            time.sleep(sleep_time)

    print("\n🎉 이어서 크롤링 완료!")
    
    # 최종 통계 출력
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                final_result = json.load(f)
            print(f"📊 최종 통계:")
            print(f"   ✅ 성공: {final_result.get('successful_count', 0)}개")
            print(f"   ❌ 실패: {final_result.get('failed_count', 0)}개")
            print(f"   📈 전체: {final_result.get('total_count', 0)}개")
        except:
            pass 