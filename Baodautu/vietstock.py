from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.chrome.options import Options
import time

# Tuỳ chọn: không hiện cửa sổ trình duyệt (headless)
# chrome_options = Options()
# chrome_options.add_argument('--headless')
# driver = webdriver.Chrome(options=chrome_options)

def vietstock_search(keyword):
    driver = webdriver.Chrome()  # Nếu cần headless thì thêm options=chrome_options
    driver.get("https://vietstock.vn")
    time.sleep(2)

    # Nhấn vào icon tìm kiếm (img.btSearchbymobile)
    try:
        search_icon = driver.find_element(By.CSS_SELECTOR, "img.btSearchbymobile")
        search_icon.click()
        time.sleep(1)
    except Exception as e:
        print("Không tìm thấy icon tìm kiếm:", e)
        driver.quit()
        return

    # Tìm ô nhập liệu trong popup (thường là input[type='text'] hoặc class .search-input)
    try:
        # Có thể cần điều chỉnh selector nếu trang thay đổi
        search_input = driver.find_element(By.CSS_SELECTOR, "input.search-input")
        search_input.clear()
        search_input.send_keys(keyword)
        search_input.send_keys(Keys.ENTER)
        print(f"Đã nhập từ khóa: {keyword}")
    except Exception as e:
        print("Không tìm thấy ô nhập liệu:", e)
        driver.quit()
        return

    # Đợi kết quả hiện ra (có thể điều chỉnh thời gian hoặc dùng WebDriverWait)
    time.sleep(5)

    # (Tuỳ chọn) Lấy tiêu đề các bài viết đầu tiên
    try:
        results = driver.find_elements(By.CSS_SELECTOR, "a.search-title")
        print("Các bài viết đầu tiên:")
        for i, a in enumerate(results[:5]):
            print(f"{i+1}. {a.text}")
    except Exception as e:
        print("Không lấy được kết quả:", e)

    # Đóng trình duyệt
    driver.quit()

if __name__ == "__main__":
    vietstock_search("vietstock daily") 