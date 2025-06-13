import os
import sys
import streamlit.web.bootstrap
from pyngrok import ngrok
import webbrowser
import socket

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_free_port(start=8501, max_port=8599):
    """Tìm một port trống bắt đầu từ start đến max_port"""
    for port in range(start, max_port + 1):
        if not is_port_in_use(port):
            return port
    raise IOError("Không tìm thấy port trống nào trong khoảng từ {} đến {}".format(start, max_port))

def run_app_with_ngrok():
    # Tìm port trống
    port = find_free_port()
    
    # Thiết lập ngrok tunnel
    public_url = ngrok.connect(port)
    print(f"Ứng dụng Streamlit của bạn đã được chia sẻ tại URL: {public_url}")
    print(f"Chia sẻ URL này cho người khác để họ có thể truy cập ứng dụng của bạn")
    print(f"Nhấn Ctrl+C để dừng")
    
    # Mở trình duyệt với URL ngrok
    webbrowser.open(public_url)
    
    # Chạy ứng dụng Streamlit trên port đã chọn
    sys.argv = ["streamlit", "run", "app.py", "--server.port", str(port)]
    streamlit.web.bootstrap.run()

if __name__ == "__main__":
    run_app_with_ngrok()
