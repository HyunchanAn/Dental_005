# Phase-Field 기반 Image Segmentation을 활용한 정확도 향상 기획

본 문서는 YOLOv8 기반의 인스턴스 분할 한계를 극복하고, 치아 우식 경계면(interface)의 정확도를 극대화하기 위해 Phase-Field 모델 및 Energy Stable Scheme을 적용하는 장기적인 성능 향상(Academic Enhancement) 계획을 정리합니다.

## 1. 개요 및 필요성
기존 YOLO 모델은 객체의 형태를 다각형(Polygon) 형태로 추출하여 빠른 추론을 제공하지만, 픽셀 단위의 미세한 곡선이나 치아 우식 특유의 흐릿한 경계(fuzzy boundary)를 완벽하게 잡아내기에는 한계가 있습니다. 이를 보완하기 위해 수학적 배경을 가진 Phase-Field 모델을 도입합니다.

## 2. 주요 참고 연구
- **Phase-Field Image Segmentation**: Vector-valued Allen-Cahn phase-field 모델을 이미지 분할에 직접 적용한 연구 (J. Lee et al., 2014)
- **Energy Stable Numerical Schemes**: Allen-Cahn 방정식의 수치적 안정성을 보장하여, 분할 과정에서 경계가 안정적으로 수렴하도록 돕는 High-Order scheme 및 Energy quadratization 기법 적용 (2022)
- **Inverse Problems 기반 전처리**: 빛 반사, 그림자 등 구강 내 촬영 이미지의 노이즈를 역문제(Inverse Problem) 기반 접근(MREIT 등)을 통해 전처리

## 3. 적용 방향 (Phase-Field + Deep Learning Hybrid)
- **하이브리드 파이프라인**: 
  - 단계 1: YOLOv8을 통한 러프한 Bounding Box 및 초기 마스크 획득 (초깃값 제공)
  - 단계 2: 획득된 마스크를 초기 조건(initial condition)으로 삼아, Phase-Field PDE를 풀이함으로써 마스크 경계선을 픽셀 단위로 정밀하게 다듬음(post-processing)
- **안정성 강화**: Energy stable scheme을 도입하여 미분방정식 수치해석 과정에서 발산 없이 최적의 경계면에 도달하도록 보장

## 4. 향후 로드맵
이러한 수학적 수치해석 기법과 딥러닝 기반 탐지 기술의 하이브리드 결합은 기존 컴퓨터 비전 기반의 의료 영상 진단 솔루션에 비해 높은 해상도와 안정성을 제공할 수 있으며, 장기적으로 기술 고도화 및 논문화(Academic paper)를 추진할 수 있는 주요 연구 방향입니다.
