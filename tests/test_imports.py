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
        from alphadent import app  # noqa: F401
        from alphadent import benchmark  # noqa: F401
        from alphadent import inference  # noqa: F401
        from alphadent import train  # noqa: F401
        from alphadent import valid  # noqa: F401
        import streamlit_app  # noqa: F401

        imports_successful = True
    except ImportError as e:
        imports_successful = False
        pytest.fail(f"Import failed: {e}")

    assert imports_successful
