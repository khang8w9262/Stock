import os
import re

SKIP_DIRS = {'.venv', '__pycache__', '.git', '.ipynb_checkpoints'}

def find_imports_in_file(filepath):
    imports = set()
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # Loại bỏ comment
                line = line.split("#")[0]
                match_import = re.match(r'^\s*import\s+([a-zA-Z0-9_\.]+)', line)
                match_from = re.match(r'^\s*from\s+([a-zA-Z0-9_\.]+)', line)
                if match_import:
                    lib = match_import.group(1).split('.')[0]
                    imports.add(lib)
                elif match_from:
                    lib = match_from.group(1).split('.')[0]
                    imports.add(lib)
    except Exception as e:
        print(f"Lỗi đọc file {filepath}: {e}")
    return imports

def find_all_imports(root_dir="."):
    all_imports = set()
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Bỏ qua các thư mục không cần thiết
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            if filename.endswith(".py"):
                filepath = os.path.join(dirpath, filename)
                print(f"Đang quét: {filepath}")
                all_imports.update(find_imports_in_file(filepath))
    return all_imports

def update_requirements(imports, req_file="requirements.txt"):
    with open(req_file, "w", encoding="utf-8") as f:
        for lib in sorted(imports):
            f.write(f"{lib}\n")
    print(f"Đã cập nhật {req_file} với {len(imports)} thư viện.")

if __name__ == "__main__":
    imports = find_all_imports(".")
    update_requirements(imports)