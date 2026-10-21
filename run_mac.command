#!/bin/bash
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  echo "처음 실행: 필요한 프로그램을 설치합니다. 몇 분 걸릴 수 있어요..."
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
else
  source .venv/bin/activate
fi
streamlit run app.py
