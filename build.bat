@echo off
chcp 65001 >nul
echo ======================================
echo  微信聊天打码工具 - 打包脚本
echo ======================================
echo.

REM Check if pyinstaller is installed
pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 PyInstaller，请先安装：
    echo   pip install -r requirements-dev.txt
    pause
    exit /b 1
)

echo [1/3] 清理旧构建...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo [2/3] 执行 PyInstaller 打包...
pyinstaller main.spec --clean --noconfirm

echo [3/3] 跳过本地配置文件...

if exist "dist\轻量化打码工具\轻量化打码工具.exe" (
    echo.
    echo ======================================
    echo  打包成功！
    echo  输出文件：dist\轻量化打码工具\轻量化打码工具.exe
    echo ======================================
) else (
    echo.
    echo [错误] 打包失败，请检查上方日志
)

pause
