import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Hàm xác định biến động dựa vào nội dung bài viết
VOLATILITY_MAP = {
    'biến động': ['biến động', 'khó dự đoán', 'không lường trước', 'không dự báo', 'bất ổn', 'khó đoán', 'khó dự báo'],
    'giảm mạnh': ['giảm mạnh', 'lao dốc', 'rớt sâu', 'giảm sâu', 'giảm kịch sàn', 'giảm sàn', 'bán tháo', 'tụt dốc'],
    'giảm nhẹ': ['giảm nhẹ', 'điều chỉnh giảm', 'suy giảm', 'nhích xuống', 'hạ nhẹ', 'giảm'],
    'ổn định': ['ổn định', 'đi ngang', 'sideway', 'không đổi', 'ít biến động', 'bình ổn'],
    'tăng nhẹ': ['tăng nhẹ', 'nhích lên', 'phục hồi', 'tăng', 'nhích tăng'],
    'tăng mạnh': ['tăng mạnh', 'tăng vọt', 'tăng trần', 'tăng kịch trần', 'bứt phá', 'leo dốc', 'tăng sốc']
}

VOLATILITY_LABELS = {
    'biến động': 'Biến động (khó dự đoán)',
    'giảm mạnh': 'Giảm mạnh',
    'giảm nhẹ': 'Giảm nhẹ',
    'ổn định': 'Ổn định',
    'tăng nhẹ': 'Tăng nhẹ',
    'tăng mạnh': 'Tăng mạnh'
}

def get_article_details(article_url):
    try:
        response = requests.get(article_url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            print(f"Failed to fetch article: {article_url}")
            return None, None, None
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract article date
        date_element = soup.select_one("span.post-time")
        article_date = date_element.get_text(strip=True) if date_element else "Unknown"
        
        # Extract article title
        title_element = soup.select_one("div.title-detail")
        header = title_element.get_text(strip=True) if title_element else "Unknown"
        
        # Extract article content
        sapo_detail = soup.select_one("div.sapo_detail")
        content_details = soup.select_one("div#content_detail_news")
        content = ""
        
        if sapo_detail:
            content += sapo_detail.get_text(strip=True) + "\n"
        if content_details:
            paragraphs = content_details.find_all("p")
            for p in paragraphs:
                content += p.get_text(strip=True) + "\n"
                
        return article_date, header, content
    except Exception as e:
        print(f"Error in get_article_details for {article_url}: {e}")
        return None, None, None

def crawl_baodautu(num_pages=10, output_csv="baodautu_articles.csv"):
    base_url = "https://baodautu.vn/tai-chinh-chung-khoan-d6/"
    articles = []
    seen_articles = set()  # Keep track of articles we've already processed
    
    try:
        for page in range(1, num_pages+1):
            url = f"{base_url}/p{page}" if page > 1 else base_url
            try:
                response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
                if response.status_code != 200:
                    print(f"Failed to fetch page {url}")
                    continue
                
                soup = BeautifulSoup(response.text, "html.parser")
                main_article = soup.select_one("div.col810.mr20")
                
                if main_article:
                    list_articles = main_article.select("li article a")
                    for a_tag in list_articles:
                        article_url = a_tag.get("href")
                        if not article_url:
                            continue
                            
                        # Skip if we've already processed this article
                        if article_url in seen_articles:
                            continue
                            
                        seen_articles.add(article_url)
                        article_date, header, content = get_article_details(article_url)
                        
                        if article_date and header and content:
                            articles.append([article_date, header, content])
                            
                print(f"Crawled page {page}")
                time.sleep(10)  # Avoid being blocked
                
            except Exception as e:
                print(f"Error crawling page {page}: {e}")
                continue
                
    except Exception as e:
        print(f"Critical error during crawling: {e}")
        
    finally:
        if articles:  # Only save if we have articles
            df = pd.DataFrame(articles, columns=["Created At", "Header", "Content"])
            df["Content"] = df["Content"].astype(str)
            
            # Remove any duplicates based on Header and Content
            df = df.drop_duplicates(subset=["Header", "Content"], keep="first")
            
            # Save to CSV
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"Crawling completed. Saved {len(df)} unique articles to {output_csv}")
        else:
            print("No articles were successfully crawled")

if __name__ == "__main__":
    crawl_baodautu(num_pages=2, output_csv="baodautu_articles.csv") 