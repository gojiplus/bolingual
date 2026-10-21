from pathlib import Path

from streamlit.testing.v1 import AppTest

import bolingual.app


def test_app_renders_query_results() -> None:
    app = AppTest.from_file(Path(bolingual.app.__file__), default_timeout=30).run()
    assert not app.exception
    app.text_input[0].set_value("वॉटसन").run()
    assert not app.exception
    assert len(app.dataframe) == 3
    assert "watson" in app.dataframe[0].value["English"].tolist()[:5]
