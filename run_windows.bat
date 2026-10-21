@echo off
chcp 65001 > nul
cd /d "%~dp0"
if not exist ".venv" (
    echo 처음 실행: 필요한 프로그램을 설치합니다. 몇 분 걸릴 수 있어요...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)
streamlit run app.py
pause
