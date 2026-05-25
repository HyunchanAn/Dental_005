import pytest

def test_imports():
    try:
        import cv2
        import numpy
        import ultralytics
        import streamlit
        import plotly
        import PIL
        
        # 프로젝트 내부 모듈 테스트
        import app
        import benchmark
        import inference
        import streamlit_app
        import train
        import valid
        
        imports_successful = True
    except ImportError as e:
        imports_successful = False
        pytest.fail(f"Import failed: {e}")
        
    assert imports_successful
