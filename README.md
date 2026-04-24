# 轻量化打码工具

一个基于 PySide6 和 OpenCV 的本地图片打码工具，支持单张图片和文件夹批量处理，可对框选区域应用马赛克、高斯模糊或纯色块。
<img width="1751" height="1164" alt="1" src="https://github.com/user-attachments/assets/fc5f40aa-4f48-4492-99a3-8b08e44aa668" />

## 运行

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## 打包

打包需要额外安装 PyInstaller：

```powershell
pip install -r requirements-dev.txt
.\build.bat
```

打包产物会输出到：

```text
dist\轻量化打码工具\轻量化打码工具.exe
```
