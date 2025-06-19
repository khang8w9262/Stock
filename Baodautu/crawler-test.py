import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from datetime import datetime

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

def determine_volatility(text):
    text = text.lower()
    for key, synonyms in VOLATILITY_MAP.items():
        for word in synonyms:
            if word in text:
                return VOLATILITY_LABELS[key]
    return 'Không xác định'

# Trích xuất giá và % thay đổi từ nội dung bài viết
PRICE_PATTERNS = [
    r'giá (?:đóng cửa|mở cửa|hiện tại|là|ở mức|chốt phiên|đạt|còn|được giao dịch ở mức)\s*([\d\.]+)\s*(?:đồng|vnđ|vnd|₫)',
    r'([\d\.]+)\s*(?:đồng|vnđ|vnd|₫)'
]
PERCENT_PATTERNS = [
    r'([\+\-]?\d{1,3}(?:[\.,]\d+)?\s*%)',
    r'(tăng|giảm)\s*([\d\.,]+)\s*%'
]

def extract_price(text):
    for pattern in PRICE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).replace('.', '').replace(',', '.')
    return ''

def extract_percent(text):
    for pattern in PERCENT_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if len(match.groups()) == 1:
                return match.group(1).replace(' ', '')
            elif len(match.groups()) == 2:
                sign = '-' if 'giảm' in match.group(1) else '+'
                return sign + match.group(2).replace(' ', '') + '%'
    return ''

# Trích xuất tên chứng khoán/cổ phiếu từ tiêu đề hoặc nội dung
STOCK_PATTERNS = [
    r'cổ phiếu ([A-Z]{3,5})',
    r'mã ([A-Z]{3,5})',
    r'([A-Z]{3,5})\b'
]
def extract_stock_name(title, content):
    for pattern in STOCK_PATTERNS:
        match = re.search(pattern, title)
        if match:
            return match.group(1)
    for pattern in STOCK_PATTERNS:
        match = re.search(pattern, content)
        if match:
            return match.group(1)
    return ''

def crawl_vietstock_search(keyword, num_pages):
    headers = {"User-Agent": "Mozilla/5.0"}
    articles = []
    for page in range(1, num_pages+1):
        if page == 1:
            url = f'https://vietstock.vn/Search?q={keyword.replace(" ", "+")}'
        else:
            url = f'https://vietstock.vn/Search?q={keyword.replace(" ", "+")}&page={page}'
        print(f'Đang lấy kết quả tìm kiếm trang {page}...')
        resp = requests.get(url, headers=headers)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Các bài viết nằm trong div.search-list > div.search-item
        items = soup.select('div.search-list div.search-item')
        for item in items:
            a_tag = item.select_one('a.search-title')
            if not a_tag:
                continue
            link = a_tag['href']
            if not link.startswith('http'):
                link = 'https://vietstock.vn' + link
            title = a_tag.get_text(strip=True)
            date_tag = item.select_one('span.search-date')
            date = date_tag.get_text(strip=True) if date_tag else ''
            articles.append({'title': title, 'link': link, 'date': date})
        time.sleep(1)
    print(f'Tìm thấy {len(articles)} bài viết với từ khóa "{keyword}".')
    # Vào từng bài viết lấy nội dung chi tiết
    data = []
    for art in articles:
        print(f"Đang xử lý: {art['title']}")
        try:
            resp = requests.get(art['link'], headers=headers)
            soup = BeautifulSoup(resp.text, "html.parser")
            content_tag = soup.select_one('div#contentdetail')
            content = content_tag.get_text(separator='\n', strip=True) if content_tag else ''
            stock = extract_stock_name(art['title'], content)
            price = extract_price(content)
            percent = extract_percent(content)
            volatility = determine_volatility(content)
            data.append({
                'Ngày': art['date'],
                'Tên chứng khoán/cổ phiếu': stock,
                'Giá': price,
                '%': percent,
                'Biến động': volatility
            })
            time.sleep(1)
        except Exception as e:
            print(f'Lỗi khi xử lý bài: {e}')
    # Xuất ra file CSV
    df = pd.DataFrame(data)
    df.to_csv(f'vietstock_search_{keyword.replace(" ", "_")}.csv', index=False, encoding='utf-8-sig')
    print(f'Đã lưu file vietstock_search_{keyword.replace(" ", "_")}.csv')

if __name__ == "__main__":
    # Chỉnh từ khóa và số trang muốn cào ở đây
    crawl_vietstock_search(keyword='vietstock daily', num_pages=5)