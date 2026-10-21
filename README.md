# 📈 퀀트 투자 연구실

퀀트 투자를 **공부하면서 직접 분석**해 보는 파이썬 대시보드입니다.
한국(코스피·코스닥 종목, 국내 ETF)과 미국(주식, ETF)을 함께 다룹니다.

> ⚠️ 학습용 도구입니다. 분석 결과는 투자 권유가 아니며, 투자 판단과 책임은 본인에게 있습니다.

## 1. 설치와 실행

**준비물**: 파이썬 3.10 이상 ([python.org](https://www.python.org/downloads/) — 윈도우는 설치할 때 *Add Python to PATH* 체크)

### 가장 쉬운 방법
- **윈도우**: `run_windows.bat` 더블클릭
- **맥**: 터미널에서 한 번 `chmod +x run_mac.command` 실행 후, `run_mac.command` 더블클릭

처음에는 필요한 라이브러리를 설치하느라 몇 분 걸리고, 그다음부터는 바로 켜집니다.
브라우저에 대시보드가 자동으로 열려요(안 열리면 http://localhost:8501).

### 직접 실행
```bash
cd quant_lab
python -m venv .venv
# 윈도우: .venv\Scripts\activate    맥/리눅스: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 2. 화면 구성

| 탭 | 하는 일 |
|---|---|
| 📚 퀀트 기초 | 수익률·위험 지표, 대표 전략, 백테스트 함정까지 7개 강의 |
| 🔍 자산 분석 | 가격 비교, 종목별 성과표, 낙폭, 상관관계, 모멘텀 순위 |
| 🧪 백테스트 | 전략 하나를 골라 설정을 바꿔 가며 상세 분석 (자산 가치·낙폭·비중 변화·월별 수익률·파라미터 민감도) |
| ⚖️ 전략 비교 | 여러 전략을 같은 기간·같은 비용으로 나란히 비교 |

**왼쪽 사이드바**에서 데이터 종류, 종목, 기간, 원화 환산, 리밸런싱 주기, 거래비용, 현금 이자를 정합니다.

### 데이터
- **연습용 데이터**: 인터넷 없이 바로 써 볼 수 있는 가상 시장. '가상 시장 번호'를 바꾸면 다른 시장이 만들어져요.
- **실제 시장 데이터**: 한국 종목은 6자리 코드(예: `005930`, `069500`), 미국은 티커(예: `SPY`, `AAPL`).
  yfinance(배당 반영 수정주가)를 먼저 쓰고, 실패하면 FinanceDataReader로 받아요.
  받은 데이터는 `data_cache` 폴더에 12시간 보관합니다. 최신으로 받으려면 사이드바의 **데이터 새로 받기**.
- **원화 환산**을 켜면 미국 자산에 원/달러 환율을 곱해, 한국 투자자가 실제 체감하는 수익률로 계산해요.

### 들어 있는 전략
매수 후 보유 · 동일 비중 리밸런싱 · 고정 비중 자산배분 · 상대 모멘텀(Top N) · 듀얼 모멘텀 · 추세추종(이동평균) · 역변동성

### 백테스트 규칙
- 리밸런싱일 **종가까지만** 보고 비중을 정하고, **다음 거래일부터** 적용 (미래 데이터 사용 없음)
- 리밸런싱 사이에는 가격에 따라 비중이 자연스럽게 변함
- 매매한 만큼 거래비용 차감, 남는 비중은 현금(설정한 이자율)

## 3. 파일 구조

```
quant_lab/
├── app.py           # 대시보드 화면
├── data.py          # 데이터 수집·캐시·환율·연습용 데이터, 종목 프리셋
├── backtest.py      # 백테스트 엔진, 성과 지표
├── strategies.py    # 전략 함수와 설명
├── lessons.py       # '퀀트 기초' 강의 내용
└── requirements.txt
```

## 4. 직접 전략 추가하기

`strategies.py`에 함수를 하나 만들고 `STRATEGIES`에 등록하면 화면에 바로 나타나요.

```python
def low_volatility_top(hist, vol_days=60, top_n=2):
    """변동성이 가장 낮은 top_n 종목을 동일 비중으로."""
    if len(hist) < vol_days + 1:
        return None                       # 데이터 부족 → 현금
    vol = hist.iloc[-vol_days - 1:].pct_change().std()
    picks = vol.nsmallest(top_n).index
    return pd.Series(1 / top_n, index=picks)   # 비중 합 < 1 이면 나머지는 현금

STRATEGIES["저변동성 Top N"] = {
    "fn": low_volatility_top,
    "params": [("vol_days", "변동성 측정 기간(거래일)", "int", 60, 20, 252, ""),
               ("top_n", "보유 종목 수", "int", 2, 1, 5, "")],
    "summary": "가장 덜 흔들린 종목만 보유합니다.",
}
```

규칙: 함수는 `hist`(리밸런싱일까지의 종가 표)만 보고 판단해야 해요. 그래야 미래 참조가 생기지 않아요.

## 5. 휴대폰에서 쓰기 (무료 클라우드 배포)

Streamlit Community Cloud에 올리면 휴대폰 브라우저에서 링크로 열 수 있어요. 휴대폰으로 접속하면 화면이 자동으로 모바일용으로 바뀌어요.

1. [github.com](https://github.com)에서 계정 만들기
2. 오른쪽 위 **+ → New repository** → 이름 `quant-lab`, **Public** 선택 → **Create repository**
3. 새 저장소 화면의 **uploading an existing file** 링크 → 이 폴더 안의 파일을 모두 끌어다 놓기 → **Commit changes**
4. [share.streamlit.io](https://share.streamlit.io) → **Continue with GitHub**로 로그인
5. **Create app → Deploy a public app from GitHub** → Repository `내아이디/quant-lab`, Branch `main`, Main file `app.py` → **Deploy**
6. 몇 분 뒤 생기는 주소(`https://….streamlit.app`)를 휴대폰에서 열고 **홈 화면에 추가**

참고: 링크를 아는 사람은 누구나 볼 수 있어요(개인 정보는 저장되지 않음). 며칠 안 쓰면 앱이 잠들고, 다시 열 때 **Yes, get this app back up** 버튼을 누르면 1분 안에 깨어나요.

## 6. 문제 해결

- **"데이터를 받지 못한 종목"**: 종목 코드 확인 → 인터넷 확인 → 잠시 후 **데이터 새로 받기**. 야후 파이낸스가 일시적으로 요청을 막는 경우가 있어요.
- **분석 시작일이 늦어짐**: 가장 늦게 상장한 종목의 시작일에 맞춰요(화면에 안내가 떠요).
- **국내 ETF 수익률이 낮게 나옴**: FinanceDataReader로 받은 경우 분배금이 반영되지 않은 가격일 수 있어요.
