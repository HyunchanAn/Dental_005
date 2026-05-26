import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# 페이지 설정
st.set_page_config(page_title="Caries Detection from Photo", page_icon="🦷", layout="wide")

st.title("🦷 Caries Detection from Photo: AI 기반 치아 우식증 진단 보조 시스템")
st.markdown("입안 사진을 업로드하면 인공지능이 충치(Caries) 부위를 자동으로 분석해줍니다.")

# 모델 로드 (캐싱하여 속도 최적화)
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

# 클래스별 색상 매핑 (시각화용)
COLORS = px.colors.qualitative.Plotly

# 사이드바 설정
st.sidebar.header("⚙️ 분석 설정")
model_type = st.sidebar.radio(
    "모델 선택",
    ("4-Class 모델 (단순화)", "9-Class 모델 (G.V. Black 분류법)")
)

conf_threshold = st.sidebar.slider("신뢰도 임계값 (Confidence)", min_value=0.1, max_value=1.0, value=0.25, step=0.05)

# 가중치 파일 결정
if "4-Class" in model_type:
    model_path = "weights/yolov8x_AlphaDent_4_classes_960px.pt"
else:
    model_path = "weights/yolov8x_AlphaDent_9_classes_960px.pt"

model = load_model(model_path)

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 시각화 설정")
show_boxes = st.sidebar.checkbox("바운딩 박스 표시", value=False, help="마우스 오버 외에 기본적으로 표시할지 여부입니다.")
line_width = st.sidebar.slider("선 두께", min_value=1, max_value=5, value=2)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Streamlit Cloud 환경에서는 CPU로 구동되므로 이미지를 분석하는 데 약간의 시간이 소요될 수 있습니다.")

# 메인 화면: 파일 업로드
uploaded_file = st.file_uploader("구강 사진을 업로드하세요 (JPG, PNG, JPEG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # 이미지 읽기
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # BGR 변환 (OpenCV 형식)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.subheader("원본 이미지")
    st.image(image, use_container_width=True)

    st.subheader("분석 결과 (마우스를 올려보세요!)")
    with st.spinner("생각하는 중... 🤔"):
        # 추론
        results = model(img_bgr, imgsz=960, conf=conf_threshold)
        res = results[0]

        # Plotly Figure 생성
        fig = go.Figure()

        # 배경 원본 이미지 추가
        fig.add_trace(go.Image(z=img_array))

        detected_count = 0
        if res.masks is not None and len(res.masks.xy) > 0:
            detected_count = len(res.masks.xy)
            for i, mask in enumerate(res.masks.xy):
                if len(mask) == 0:
                    continue

                cls_idx = int(res.boxes.cls[i].item())
                conf = res.boxes.conf[i].item()
                label_name = res.names[cls_idx]
                hover_text = f"<b>{label_name}</b><br>Confidence: {conf:.2f}"

                # 클래스별 색상 할당
                color_hex = COLORS[cls_idx % len(COLORS)]

                x_coords = mask[:, 0].tolist()
                y_coords = mask[:, 1].tolist()

                # 폴리곤 닫기
                x_coords.append(x_coords[0])
                y_coords.append(y_coords[0])

                # 바운딩 박스가 켜져있다면, 마스크 선도 같이 렌더링되게 하거나 박스 그리기
                line_opacity = 0.8 if show_boxes else 0.0

                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    fill='toself',
                    fillcolor=color_hex,
                    line=dict(color=color_hex, width=line_width if show_boxes else 0),
                    opacity=0.5, # 투명도 일괄 적용
                    mode='lines',
                    name=label_name,
                    text=hover_text,
                    hoverinfo='text',
                    showlegend=False
                ))

        # 레이아웃 설정 (여백 제거 및 원본 비율 유지)
        fig.update_xaxes(visible=False, range=[0, img_array.shape[1]])
        fig.update_yaxes(visible=False, range=[img_array.shape[0], 0], scaleanchor="x", scaleratio=1)
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            hovermode='closest',
            autosize=True
        )

    # Plotly 차트 렌더링 (컨테이너 폭에 맞춤)
    st.plotly_chart(fig, use_container_width=True)

    if detected_count > 0:
        st.success(f"분석 완료! 총 {detected_count}개의 병변이 감지되었습니다.")
    else:
        st.info("분석 완료! 병변이 감지되지 않았습니다.")
