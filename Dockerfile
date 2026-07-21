FROM python:3.12-slim

# OS 패키지 설치 (OpenCV 구동 등 필수 패키지)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉터리 설정
WORKDIR /app

# 파이썬 패키지 복사 및 설치
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# 소스코드 복사
COPY src/ ./src/
COPY streamlit_app.py ./
# COPY tests/ ./tests/
# Note: 모델 가중치(*.pt)와 데이터셋은 이미지 크기를 위해 .dockerignore로 제외됩니다.
# 컨테이너 실행 시 볼륨 마운트를 권장합니다.

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
