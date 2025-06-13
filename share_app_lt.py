import os
import sys
import subprocess
import webbrowser
import time
import socket

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def run_app_with_localtunnel():
    port = 8501
    
    # Kiểm tra xem port 8501 có đang được sử dụng không
    if is_port_in_use(port):
        print(f"Port {port} đã đang được sử dụng. Có thể Streamlit đã chạy.")
    else:
        # Chạy ứng dụng Streamlit
        streamlit_process = subprocess.Popen(
            ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        # Đợi Streamlit khởi động
        time.sleep(5)
    
    # Cài đặt localtunnel nếu chưa có
    try:
        subprocess.run(["npm", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Node.js và npm cần được cài đặt để sử dụng LocalTunnel.")
        print("Vui lòng cài đặt Node.js từ https://nodejs.org/")
        return
    
    # Cài đặt localtunnel nếu chưa có
    subprocess.run(["npm", "install", "-g", "localtunnel"], check=True)
    
    # Chạy localtunnel
    lt_process = subprocess.Popen(
        ["lt", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    # Đọc URL từ output của localtunnel
    for line in lt_process.stdout:
        if "your url is:" in line.lower():
            url = line.strip().split("your url is: ")[1]
            print(f"Ứng dụng Streamlit của bạn đã được chia sẻ tại URL: {url}")
            print(f"Chia sẻ URL này cho người khác để họ có thể truy cập ứng dụng của bạn")
            print(f"Nhấn Ctrl+C để dừng")
            webbrowser.open(url)
            break
    
    # Giữ script chạy
    try:
        lt_process.wait()
    except KeyboardInterrupt:
        print("Đang dừng ứng dụng...")
        lt_process.terminate()
        if not is_port_in_use(port):
            streamlit_process.terminate()

if __name__ == "__main__":
    run_app_with_localtunnel()
