"""
test_ui_smoke.py - Streamlit AppTest smoke tests verifying UI rendering and page loads.
"""

from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).parent.parent / "app.py"


def test_app_smoke_initial_render():
    """Verify that app.py initializes cleanly without unhandled exceptions or crashes."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    # Check that there are no uncaught Python exceptions
    assert not at.exception


def test_app_smoke_navigation_switch():
    """Verify switching pages renders cleanly."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    assert not at.exception

    # Switch radio to second page (Dataset Profiler)
    if at.sidebar.radio:
        at.sidebar.radio[0].set_value("📊 Dataset Profiler").run()
        assert not at.exception
