@echo off
echo ========================================================
echo   YOLO Vision Studio — Python AI Engine
echo   Target Conda Env: D:\WorkSpace\CondaEnv\AmalAI
echo ========================================================
"D:\WorkSpace\CondaEnv\AmalAI\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
