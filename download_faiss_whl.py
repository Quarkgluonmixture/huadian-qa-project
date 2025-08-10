import requests
import os

url = "https://github.com/kyamagu/faiss-wheels/releases/download/v1.11.0/faiss_gpu-1.11.0-cp311-cp311-win_amd64.whl"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# 临时禁用代理（如果 Clash 干扰）
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

response = requests.get(url, headers=headers, stream=True)

if response.status_code == 200:
    with open('faiss_gpu-1.11.0-cp311-cp311-win_amd64.whl', 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("下载成功！文件保存在当前目录。")
else:
    print(f"下载失败：状态码 {response.status_code}。请检查网络或代理。")