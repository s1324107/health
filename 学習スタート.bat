@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo 先にインストール.batを実行してください。
 pause
 exit /b 1
)
.venv\Scripts\python.exe train.py
pause
