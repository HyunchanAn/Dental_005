import pytest


def test_imports():
    try:
        import cv2
        import numpy
        import PIL
        import plotly
        import streamlit
        import ultralytics

        # 패키지 내부 모듈 테스트
        from alphadent import app
        from alphadent import benchmark
        from alphadent import inference
        from alphadent import train
        from alphadent import valid
        import streamlit_app

        imports_successful = True
    except ImportError as e:
        imports_successful = False
        pytest.fail(f"Import failed: {e}")

    assert imports_successful
