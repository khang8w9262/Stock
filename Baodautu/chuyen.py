import re
import csv
import os
import datetime
import pandas as pd
from typing import List, Dict, Tuple, Optional

def phan_tich_do_giao_dong(text: str) -> str:
    """Phân tích độ giao động từ đoạn văn bản."""
    text = text.lower()
    if "tăng mạnh" in text:
        return "Tăng mạnh"
    elif "giảm mạnh" in text:
        return "Giảm mạnh"
    elif "tăng" in text and not "giảm" in text:
        return "Tăng nhẹ"
    elif "giảm" in text and not "tăng" in text:
        return "Giảm nhẹ"
    elif "đi ngang" in text or "ổn định" in text or "đứng giá" in text:
        return "Ổn định"
    else:
        return "Bất thường"

def trích_xuất_thông_tin_từ_đoạn(đoạn_văn: str) -> List[Dict]:
    """Trích xuất thông tin từ một đoạn văn cụ thể."""
    kết_quả = []
    
    # Tách các phần theo dấu phẩy + khoảng trắng để phân tích từng mã riêng biệt
    các_đoạn_nhỏ = đoạn_văn.split(", ")
    
    for đoạn_nhỏ in các_đoạn_nhỏ:
        if not đoạn_nhỏ.strip():
            continue
            
        # Tìm mã chứng khoán
        mã_chứng_khoán = tìm_mã_chứng_khoán(đoạn_nhỏ)
        if not mã_chứng_khoán:
            continue
            
        # Tìm thông tin tăng/giảm
        độ_giao_động = phan_tich_do_giao_dong(đoạn_nhỏ)
        
        # Tìm giá trị thay đổi
        giá_trị_thay_đổi = trích_xuất_giá_trị_thay_đổi(đoạn_nhỏ)
        
        # Tìm phần trăm thay đổi 
        phần_trăm = trích_xuất_phần_trăm(đoạn_nhỏ)
        
        # Tìm giá trị hiện tại
        giá_trị = trích_xuất_giá_hiện_tại(đoạn_nhỏ)
        
        if mã_chứng_khoán:
            kết_quả.append({
                "mã_chứng_khoán": mã_chứng_khoán,
                "độ_giao_động": độ_giao_động,
                "giá_trị_thay_đổi": giá_trị_thay_đổi,
                "phần_trăm_thay_đổi": phần_trăm,
                "giá_trị": giá_trị
            })
    
    return kết_quả

def tìm_mã_chứng_khoán(đoạn: str) -> str:
    """Tìm mã chứng khoán trong đoạn văn."""
    # Danh sách mã phổ biến
    các_mã_phổ_biến = ['VN-Index', 'HNX-Index', 'UPCOM-Index', 'VN30']
    
    # Kiểm tra các mã phổ biến trước
    for mã in các_mã_phổ_biến:
        if mã in đoạn:
            return mã

    # Tìm mã 3 chữ cái viết hoa
    pattern_mã = r'\b[A-Z]{3}\b'
    các_mã = re.findall(pattern_mã, đoạn)
    if các_mã:
        return các_mã[0]
    
    return ""

def trích_xuất_giá_trị_thay_đổi(đoạn: str) -> str:
    """Trích xuất giá trị thay đổi từ đoạn."""
    đoạn = đoạn.lower()
    
    # Kiểm tra xem đoạn nói về tăng hay giảm
    dấu_hiệu = ""
    if "tăng" in đoạn and not "giảm" in đoạn:
        dấu_hiệu = "+"
    elif "giảm" in đoạn and not "tăng" in đoạn:
        dấu_hiệu = "-"
        
    # Tìm số kèm theo từ "tăng" hoặc "giảm" + "x,xx điểm" hoặc "x,xx%" hoặc "đồng/cp" hoặc "đồng/cổ phiếu"
    pattern_giá_trị = r'(?:tăng|giảm)\s+(\d+[,.]\d+)(?:\s+điểm|\s+%|\s+đồng(?:/cp|/cổ phiếu)?|\s*$)'
    kết_quả = re.search(pattern_giá_trị, đoạn)
    
    if kết_quả:
        giá_trị = kết_quả.group(1)
        if dấu_hiệu and not (giá_trị.startswith('+') or giá_trị.startswith('-')):
            giá_trị = dấu_hiệu + giá_trị
        return giá_trị
    
    # Trường hợp đặc biệt cho định dạng "đóng góp x,xx điểm"
    pattern_đóng_góp = r'đóng góp\s+(\d+[,.]\d+)(?:\s+điểm|\s+đồng(?:/cp|/cổ phiếu)?)'
    kết_quả = re.search(pattern_đóng_góp, đoạn)
    if kết_quả:
        giá_trị = kết_quả.group(1)
        if dấu_hiệu and not (giá_trị.startswith('+') or giá_trị.startswith('-')):
            giá_trị = dấu_hiệu + giá_trị
        return giá_trị
    
    return ""

def trích_xuất_phần_trăm(đoạn: str) -> str:
    """Trích xuất phần trăm thay đổi từ đoạn."""
    đoạn = đoạn.lower()
    
    # Kiểm tra dấu hiệu tăng/giảm
    dấu_hiệu = ""
    if "tăng" in đoạn and not "giảm" in đoạn:
        dấu_hiệu = "+"
    elif "giảm" in đoạn and not "tăng" in đoạn:
        dấu_hiệu = "-"
    
    # Tìm phần trăm trong đoạn có dạng (x,xx%) hoặc x,xx% hoặc tăng/giảm x,xx%
    pattern_phần_trăm = r'(\d+[,.]\d+%)(?:\)|\s|$)'
    kết_quả = re.search(pattern_phần_trăm, đoạn)
    
    if kết_quả:
        phần_trăm = kết_quả.group(1)
        if dấu_hiệu and not (phần_trăm.startswith('+') or phần_trăm.startswith('-')):
            phần_trăm = dấu_hiệu + phần_trăm
        return phần_trăm
    
    # Mẫu phức tạp hơn với dấu ngoặc đơn
    pattern_phần_trăm_ngoặc = r'\((\d+[,.]\d+%)\)'
    kết_quả = re.search(pattern_phần_trăm_ngoặc, đoạn)
    if kết_quả:
        phần_trăm = kết_quả.group(1)
        if dấu_hiệu and not (phần_trăm.startswith('+') or phần_trăm.startswith('-')):
            phần_trăm = dấu_hiệu + phần_trăm
        return phần_trăm
    
    return ""

def trích_xuất_giá_hiện_tại(đoạn: str) -> str:
    """Trích xuất giá hiện tại từ đoạn."""
    # Tìm các mẫu như "lên x.xxx,xx điểm", "đạt x.xxx,xx điểm", "còn x.xxx,xx điểm", kể cả đồng/cp, đồng/cổ phiếu
    pattern_giá = r'(?:lên|đạt|còn|xuống)\s+(\d{1,3}(?:[.]\d{3})*[,]\d{2})\s+(?:điểm|đồng(?:/cp|/cổ phiếu)?)'
    kết_quả = re.search(pattern_giá, đoạn.lower())
    
    if kết_quả:
        return kết_quả.group(1)
    
    # Mẫu đơn giản hơn tìm số có dạng "x.xxx,xx" kèm đơn vị
    pattern_số = r'(\d{1,3}(?:[.]\d{3})*[,]\d{2})\s+(?:điểm|đồng(?:/cp|/cổ phiếu)?)'
    kết_quả = re.search(pattern_số, đoạn.lower())
    
    if kết_quả:
        return kết_quả.group(1)
    
    # Tìm giá trị theo sau mã chứng khoán
    mã_regex = r'\b[A-Z]{3}\b'
    các_mã = re.findall(mã_regex, đoạn)
    if các_mã:
        mã = các_mã[0]
        # Tìm giá trị theo sau mã
        pattern_sau_mã = r'{}\s+(?:\w+\s+)*?(\d{{1,3}}(?:[,.]\d{{3}})*(?:[,.]\d{{2}})?)\s+(?:điểm|đồng(?:/cp|/cổ phiếu)?)'.format(mã)
        kết_quả = re.search(pattern_sau_mã, đoạn, re.IGNORECASE)
        if kết_quả:
            return kết_quả.group(1)
    
    return ""

def đọc_file_csv(đường_dẫn: str) -> pd.DataFrame:
    """Đọc file CSV và trả về DataFrame."""
    try:
        # In thông tin về file
        print(f"Đang đọc file: {đường_dẫn}")
        if not os.path.exists(đường_dẫn):
            print(f"Lỗi: File {đường_dẫn} không tồn tại!")
            return pd.DataFrame()
            
        # Đọc file CSV bằng pandas
        df = pd.read_csv(đường_dẫn)
        
        # In thông tin về cấu trúc file
        print(f"Cấu trúc file CSV:")
        print(f"- Số dòng: {len(df)}")
        print(f"- Các cột: {', '.join(df.columns.tolist())}")
        
        return df
    
    except Exception as e:
        print(f"Lỗi khi đọc file {đường_dẫn}: {e}")
        return pd.DataFrame()

def phân_tích_bài_viết(ngày: str, nội_dung: str) -> List[Dict]:
    """Phân tích bài viết và trích xuất thông tin chứng khoán."""
    kết_quả = []
    
    # Tách thành các đoạn để phân tích
    các_đoạn = nội_dung.split(".")
    
    for đoạn in các_đoạn:
        đoạn = đoạn.strip()
        if not đoạn:
            continue
        
        # Tìm kiếm thông tin trong đoạn
        thông_tin_đoạn = trích_xuất_thông_tin_từ_đoạn(đoạn)
        
        # Thêm ngày vào mỗi thông tin
        for item in thông_tin_đoạn:
            item["ngày"] = ngày
            kết_quả.append(item)
    
    # Xử lý các trường hợp đặc biệt
    kết_quả = xử_lý_trường_hợp_đặc_biệt(kết_quả, nội_dung)
    
    # Loại bỏ các dòng trùng lặp
    kết_quả_cuối = []
    mã_đã_xử_lý = set()
    
    for item in kết_quả:
        mã = item["mã_chứng_khoán"]
        # Nếu có đủ thông tin và chưa có mã này
        if mã and mã not in mã_đã_xử_lý and (item["giá_trị_thay_đổi"] or item["phần_trăm_thay_đổi"]):
            kết_quả_cuối.append(item)
            mã_đã_xử_lý.add(mã)
    
    return kết_quả_cuối

def xử_lý_trường_hợp_đặc_biệt(kết_quả: List[Dict], nội_dung: str) -> List[Dict]:
    """Xử lý các trường hợp đặc biệt trong bài viết."""
    
    # Tìm thông tin chỉ số VN-Index từ kết thúc bài viết
    pattern_vnindex = r'VN-Index\s+tăng\s+(\d+[,.]\d+)\s+điểm\s+\((\d+[,.]\d+%)\)\s+lên\s+(\d{1,3}(?:[.]\d{3})*[,]\d{2})\s+điểm'
    kết_quả_vnindex = re.search(pattern_vnindex, nội_dung)
    
    if kết_quả_vnindex:
        giá_trị_thay_đổi = "+" + kết_quả_vnindex.group(1)
        phần_trăm = "+" + kết_quả_vnindex.group(2)
        giá_trị = kết_quả_vnindex.group(3)
        
        # Kiểm tra xem VN-Index đã có trong kết quả chưa
        vn_index_exists = False
        for i, item in enumerate(kết_quả):
            if item["mã_chứng_khoán"] == "VN-Index":
                vn_index_exists = True
                # Cập nhật thông tin chính xác hơn
                kết_quả[i]["giá_trị_thay_đổi"] = giá_trị_thay_đổi
                kết_quả[i]["phần_trăm_thay_đổi"] = phần_trăm
                kết_quả[i]["giá_trị"] = giá_trị
                kết_quả[i]["độ_giao_động"] = "Tăng nhẹ"
                break
        
        # Nếu chưa có, thêm mới vào kết quả
        if not vn_index_exists:
            for item in kết_quả:
                if "ngày" in item:
                    ngày = item["ngày"]
                    kết_quả.append({
                        "ngày": ngày,
                        "mã_chứng_khoán": "VN-Index",
                        "độ_giao_động": "Tăng nhẹ",
                        "giá_trị_thay_đổi": giá_trị_thay_đổi,
                        "phần_trăm_thay_đổi": phần_trăm,
                        "giá_trị": giá_trị
                    })
                    break
    
    # Tìm thông tin chỉ số HNX-Index
    pattern_hnx = r'HNX-Index\s+giảm\s+(\d+[,.]\d+)\s+điểm\s+\((\d+[,.]\d+%)\)\s+xuống\s+(\d{1,3}(?:[.]\d{3})*[,]\d{2})\s+điểm'
    kết_quả_hnx = re.search(pattern_hnx, nội_dung)
    
    if kết_quả_hnx:
        giá_trị_thay_đổi = "-" + kết_quả_hnx.group(1)
        phần_trăm = "-" + kết_quả_hnx.group(2)
        giá_trị = kết_quả_hnx.group(3)
        
        # Kiểm tra xem HNX-Index đã có trong kết quả chưa
        hnx_exists = False
        for i, item in enumerate(kết_quả):
            if item["mã_chứng_khoán"] == "HNX-Index":
                hnx_exists = True
                # Cập nhật thông tin chính xác hơn
                kết_quả[i]["giá_trị_thay_đổi"] = giá_trị_thay_đổi
                kết_quả[i]["phần_trăm_thay_đổi"] = phần_trăm
                kết_quả[i]["giá_trị"] = giá_trị
                kết_quả[i]["độ_giao_động"] = "Giảm nhẹ"
                break
        
        # Nếu chưa có, thêm mới vào kết quả
        if not hnx_exists:
            for item in kết_quả:
                if "ngày" in item:
                    ngày = item["ngày"]
                    kết_quả.append({
                        "ngày": ngày,
                        "mã_chứng_khoán": "HNX-Index",
                        "độ_giao_động": "Giảm nhẹ",
                        "giá_trị_thay_đổi": giá_trị_thay_đổi,
                        "phần_trăm_thay_đổi": phần_trăm,
                        "giá_trị": giá_trị
                    })
                    break
                    
    # Tương tự cho UPCOM-Index
    pattern_upcom = r'UPCoM-Index\s+giảm\s+(\d+[,.]\d+)\s+điểm\s+\((\d+[,.]\d+%)\)\s+xuống\s+(\d{1,3}(?:[.]\d{3})*[,]\d{2})\s+điểm'
    kết_quả_upcom = re.search(pattern_upcom, nội_dung)
    
    if kết_quả_upcom:
        giá_trị_thay_đổi = "-" + kết_quả_upcom.group(1)
        phần_trăm = "-" + kết_quả_upcom.group(2)
        giá_trị = kết_quả_upcom.group(3)
        
        # Thêm vào kết quả
        upcom_exists = False
        for i, item in enumerate(kết_quả):
            if item["mã_chứng_khoán"] == "UPCOM-Index":
                upcom_exists = True
                kết_quả[i]["giá_trị_thay_đổi"] = giá_trị_thay_đổi
                kết_quả[i]["phần_trăm_thay_đổi"] = phần_trăm
                kết_quả[i]["giá_trị"] = giá_trị
                kết_quả[i]["độ_giao_động"] = "Giảm nhẹ"
                break
        
        if not upcom_exists:
            for item in kết_quả:
                if "ngày" in item:
                    ngày = item["ngày"]
                    kết_quả.append({
                        "ngày": ngày,
                        "mã_chứng_khoán": "UPCOM-Index",
                        "độ_giao_động": "Giảm nhẹ",
                        "giá_trị_thay_đổi": giá_trị_thay_đổi,
                        "phần_trăm_thay_đổi": phần_trăm,
                        "giá_trị": giá_trị
                    })
                    break
    
    return kết_quả

def lưu_vào_csv(dữ_liệu: List[Dict], đường_dẫn_csv: str):
    """Lưu dữ liệu vào file CSV."""
    tiêu_đề = ["Ngày tháng năm", "Tên chứng khoán/cổ phiếu", "Độ giao động", 
               "Giá trị thay đổi", "% thay đổi", "Giá trị"]
    
    try:
        with open(đường_dẫn_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(tiêu_đề)
            
            for dòng in dữ_liệu:
                writer.writerow([
                    dòng.get("ngày", ""),
                    dòng.get("mã_chứng_khoán", ""),
                    dòng.get("độ_giao_động", ""),
                    dòng.get("giá_trị_thay_đổi", ""),
                    dòng.get("phần_trăm_thay_đổi", ""),
                    dòng.get("giá_trị", "")
                ])
        print(f"Đã lưu dữ liệu thành công vào file {đường_dẫn_csv}")
    except Exception as e:
        print(f"Lỗi khi lưu file CSV: {e}")

def sắp_xếp_theo_tên_chứng_khoán(danh_sách: List[Dict]) -> List[Dict]:
    """Sắp xếp danh sách kết quả theo tên chứng khoán theo thứ tự ABC."""
    return sorted(danh_sách, key=lambda x: x.get("mã_chứng_khoán", ""))

def sắp_xếp_theo_ngày(danh_sách: List[Dict]) -> List[Dict]:
    """Sắp xếp danh sách kết quả theo ngày từ mới đến cũ."""
    return sorted(danh_sách, key=lambda x: x.get("ngày", ""), reverse=True)

def sắp_xếp_theo_giá_trị_thay_đổi(danh_sách: List[Dict]) -> List[Dict]:
    """Sắp xếp danh sách kết quả theo giá trị thay đổi từ cao đến thấp."""
    def giá_trị_thay_đổi_số(item):
        giá_trị = item.get("giá_trị_thay_đổi", "0")
        if not giá_trị:
            return 0
        # Chuyển đổi từ chuỗi sang số
        giá_trị = giá_trị.replace(",", ".")
        giá_trị = giá_trị.replace("+", "")
        try:
            return float(giá_trị)
        except:
            return 0
    
    return sorted(danh_sách, key=giá_trị_thay_đổi_số, reverse=True)

def sắp_xếp_theo_phần_trăm_thay_đổi(danh_sách: List[Dict]) -> List[Dict]:
    """Sắp xếp danh sách kết quả theo phần trăm thay đổi từ cao đến thấp."""
    def phần_trăm_thay_đổi_số(item):
        phần_trăm = item.get("phần_trăm_thay_đổi", "0%")
        if not phần_trăm:
            return 0
        # Loại bỏ dấu % và chuyển đổi từ chuỗi sang số
        phần_trăm = phần_trăm.replace("%", "")
        phần_trăm = phần_trăm.replace(",", ".")
        phần_trăm = phần_trăm.replace("+", "")
        try:
            return float(phần_trăm)
        except:
            return 0
    
    return sorted(danh_sách, key=phần_trăm_thay_đổi_số, reverse=True)

def lọc_dữ_liệu_trùng_lặp(danh_sách: List[Dict]) -> List[Dict]:
    """Lọc bỏ các dòng dữ liệu trùng lặp hoàn toàn trong kết quả."""
    kết_quả = []
    đã_thêm = set()  # Tập hợp theo dõi các mục đã thêm
    
    for mục in danh_sách:
        # Tạo khóa duy nhất từ thông tin quan trọng
        khóa = (
            mục.get("ngày", ""),
            mục.get("mã_chứng_khoán", ""),
            mục.get("độ_giao_động", ""),
            mục.get("giá_trị_thay_đổi", ""),
            mục.get("phần_trăm_thay_đổi", ""),
            mục.get("giá_trị", "")
        )
        
        # Chỉ thêm vào kết quả nếu chưa gặp khóa này
        if khóa not in đã_thêm:
            kết_quả.append(mục)
            đã_thêm.add(khóa)
    
    print(f"Đã lọc bỏ {len(danh_sách) - len(kết_quả)} dòng trùng lặp hoàn toàn.")
    return kết_quả

def lọc_dữ_liệu_theo_mã_chứng_khoán(danh_sách: List[Dict]) -> List[Dict]:
    """Lọc để chỉ giữ lại thông tin mới nhất và đầy đủ nhất cho mỗi mã chứng khoán."""
    
    # Nhóm các mục theo mã chứng khoán
    dữ_liệu_theo_mã = {}
    
    for mục in danh_sách:
        mã = mục.get("mã_chứng_khoán", "")
        if not mã:
            continue
            
        # Nếu mã chưa có trong từ điển, thêm vào
        if mã not in dữ_liệu_theo_mã:
            dữ_liệu_theo_mã[mã] = mục
        else:
            # Nếu đã có, so sánh để lấy thông tin đầy đủ hơn
            mục_hiện_tại = dữ_liệu_theo_mã[mã]
            
            # Ưu tiên mục có ngày mới hơn
            if _so_sánh_ngày(mục.get("ngày", ""), mục_hiện_tại.get("ngày", "")):
                dữ_liệu_theo_mã[mã] = mục
                continue
                
            # Nếu cùng ngày, ưu tiên mục có nhiều thông tin hơn
            if mục.get("ngày", "") == mục_hiện_tại.get("ngày", ""):
                điểm_đầy_đủ_hiện_tại = _đánh_giá_độ_đầy_đủ(mục_hiện_tại)
                điểm_đầy_đủ_mới = _đánh_giá_độ_đầy_đủ(mục)
                
                if điểm_đầy_đủ_mới > điểm_đầy_đủ_hiện_tại:
                    dữ_liệu_theo_mã[mã] = mục
    
    # Chuyển từ dictionary sang danh sách
    kết_quả = list(dữ_liệu_theo_mã.values())
    
    print(f"Đã lọc từ {len(danh_sách)} xuống {len(kết_quả)} mục (mỗi mã chứng khoán chỉ giữ lại 1 mục).")
    return kết_quả

def _so_sánh_ngày(ngày_mới: str, ngày_cũ: str) -> bool:
    """So sánh hai chuỗi ngày, trả về True nếu ngày_mới mới hơn ngày_cũ."""
    if not ngày_mới or not ngày_cũ:
        return bool(ngày_mới)
        
    try:
        # Chuyển đổi chuỗi ngày thành đối tượng datetime
        ngày_mới_obj = None
        ngày_cũ_obj = None
        
        # Thử với định dạng dd/mm/yyyy
        try:
            ngày_mới_obj = datetime.datetime.strptime(ngày_mới, "%d/%m/%Y")
        except:
            try:
                # Thử với định dạng dd/mm/yyyy HH:MM
                ngày_mới_obj = datetime.datetime.strptime(ngày_mới, "%d/%m/%Y %H:%M")
            except:
                return False
                
        try:
            ngày_cũ_obj = datetime.datetime.strptime(ngày_cũ, "%d/%m/%Y")
        except:
            try:
                # Thử với định dạng dd/mm/yyyy HH:MM
                ngày_cũ_obj = datetime.datetime.strptime(ngày_cũ, "%d/%m/%Y %H:%M")
            except:
                return True
        
        # So sánh hai ngày
        return ngày_mới_obj > ngày_cũ_obj
    except:
        # Nếu có lỗi, giả định ngày_mới không mới hơn
        return False

def _đánh_giá_độ_đầy_đủ(mục: Dict) -> int:
    """Đánh giá độ đầy đủ của một mục dữ liệu, trả về điểm số."""
    điểm = 0
    
    # Tính điểm cho mỗi trường thông tin không rỗng
    if mục.get("độ_giao_động", ""):
        điểm += 1
    if mục.get("giá_trị_thay_đổi", ""):
        điểm += 2  # Trọng số cao hơn cho thông tin quan trọng
    if mục.get("phần_trăm_thay_đổi", ""):
        điểm += 2
    if mục.get("giá_trị", ""):
        điểm += 2
        
    return điểm

def phân_tích_tất_cả_bài_viết(df: pd.DataFrame, file_đầu_ra: str, kiểu_sắp_xếp: str = "tên"):
    """Phân tích tất cả các bài viết trong DataFrame."""
    tất_cả_kết_quả = []
    số_bài_có_dữ_liệu = 0
    
    print(f"Bắt đầu phân tích {len(df)} bài viết...")
    
    # Xử lý từng dòng trong DataFrame
    for i, row in df.iterrows():
        if i % 100 == 0 and i > 0:
            print(f"Đã phân tích {i}/{len(df)} bài viết...")
        
        # Lấy ngày và nội dung từ mỗi dòng
        ngày = ""
        if 'Created At' in df.columns:
            ngày_raw = row['Created At']
            if isinstance(ngày_raw, str):
                ngày = ngày_raw.strip()
                if ngày.startswith('-'):
                    ngày = ngày[2:].strip()
        
        nội_dung = ""
        if 'Content' in df.columns:
            nội_dung = row['Content']
            if not isinstance(nội_dung, str):
                continue
        
        if not nội_dung:
            continue
            
        # Phân tích bài viết
        kết_quả = phân_tích_bài_viết(ngày, nội_dung)
        
        # Nếu có dữ liệu, thêm vào kết quả chung
        if kết_quả:
            tất_cả_kết_quả.extend(kết_quả)
            số_bài_có_dữ_liệu += 1
    
    print(f"Đã phân tích xong. Tìm thấy thông tin chứng khoán trong {số_bài_có_dữ_liệu}/{len(df)} bài viết.")
    
    # Lưu kết quả vào CSV
    if tất_cả_kết_quả:
        print(f"Tổng số mục dữ liệu ban đầu: {len(tất_cả_kết_quả)}")
        
        # Chỉ lọc bỏ dữ liệu trùng lặp hoàn toàn (trùng về mọi giá trị)
        tất_cả_kết_quả = lọc_dữ_liệu_trùng_lặp(tất_cả_kết_quả)
        
        # Không sử dụng lọc_dữ_liệu_theo_mã_chứng_khoán vì chúng ta muốn giữ lại
        # tất cả các dòng không trùng lặp hoàn toàn, ngay cả khi chúng cùng mã chứng khoán
        
        print(f"Tổng số mục dữ liệu sau khi lọc hoàn tất: {len(tất_cả_kết_quả)}")
        
        # Sắp xếp kết quả theo kiểu được chọn
        if kiểu_sắp_xếp == "tên":
            tất_cả_kết_quả = sắp_xếp_theo_tên_chứng_khoán(tất_cả_kết_quả)
            print("Đã sắp xếp kết quả theo thứ tự tên chứng khoán (A-Z)")
        elif kiểu_sắp_xếp == "ngày":
            tất_cả_kết_quả = sắp_xếp_theo_ngày(tất_cả_kết_quả)
            print("Đã sắp xếp kết quả theo ngày (mới nhất trước)")
        elif kiểu_sắp_xếp == "giá_trị":
            tất_cả_kết_quả = sắp_xếp_theo_giá_trị_thay_đổi(tất_cả_kết_quả)
            print("Đã sắp xếp kết quả theo giá trị thay đổi (cao nhất trước)")
        elif kiểu_sắp_xếp == "phần_trăm":
            tất_cả_kết_quả = sắp_xếp_theo_phần_trăm_thay_đổi(tất_cả_kết_quả)
            print("Đã sắp xếp kết quả theo phần trăm thay đổi (cao nhất trước)")
        
        lưu_vào_csv(tất_cả_kết_quả, file_đầu_ra)
    else:
        print("Không tìm thấy thông tin chứng khoán nào trong tất cả các bài viết!")

def main():
    print("Chương trình chuyển đổi dữ liệu chứng khoán từ bài viết sang CSV")
    print("===============================================================")
    
    # Cho phép người dùng chọn file đầu vào
    file_đầu_vào = input("Nhập tên file CSV đầu vào (hoặc để trống để dùng baodautu_articles.csv): ")
    if not file_đầu_vào:
        file_đầu_vào = "baodautu_articles.csv"
    
    file_đầu_ra = input("Nhập tên file CSV đầu ra (hoặc để trống để dùng baodautu_chuyen.csv): ")
    if not file_đầu_ra:
        file_đầu_ra = "baodautu_chuyen.csv"
    
    # Cho phép người dùng chọn kiểu sắp xếp
    print("\nChọn kiểu sắp xếp kết quả:")
    print("1. Theo tên chứng khoán (A-Z)")
    print("2. Theo ngày (mới nhất trước)")
    print("3. Theo giá trị thay đổi (cao nhất trước)")
    print("4. Theo phần trăm thay đổi (cao nhất trước)")
    
    lựa_chọn = input("Nhập lựa chọn của bạn (1-4, mặc định là 1): ")
    
    kiểu_sắp_xếp = "tên"  # Mặc định
    if lựa_chọn == "2":
        kiểu_sắp_xếp = "ngày"
    elif lựa_chọn == "3":
        kiểu_sắp_xếp = "giá_trị"
    elif lựa_chọn == "4":
        kiểu_sắp_xếp = "phần_trăm"
    
    print(f"File đầu vào: {file_đầu_vào}")
    print(f"File đầu ra: {file_đầu_ra}")
    print(f"Kiểu sắp xếp: {kiểu_sắp_xếp}")
    
    # Kiểm tra file đầu vào
    if not os.path.exists(file_đầu_vào):
        print(f"Lỗi: File {file_đầu_vào} không tồn tại!")
        return
    
    # Đọc dữ liệu từ file CSV
    df = đọc_file_csv(file_đầu_vào)
    
    if df.empty:
        print("Không đọc được dữ liệu từ file CSV!")
        return
    
    # Phân tích tất cả bài viết với kiểu sắp xếp đã chọn
    phân_tích_tất_cả_bài_viết(df, file_đầu_ra, kiểu_sắp_xếp)

if __name__ == "__main__":
    main() 