from __future__ import annotations

from unittest.mock import MagicMock, patch

from convo_tools.__main__ import main


def test_main_no_args() -> None:
    with patch("sys.argv", ["convo-tools", "-m"]):
        assert main() == 1


def test_main_unknown_mode() -> None:
    with patch("sys.argv", ["convo-tools", "-m", "bogus"]):
        assert main() == 1


def test_main_export() -> None:
    with patch("sys.argv", ["convo-tools", "-m", "export", "--help"]):
        with patch("convo_tools.__main__.run_export") as mock_fn:
            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = MagicMock()
                mock_fn.return_value = None
                main()


def test_main_depth() -> None:
    with patch("sys.argv", ["convo-tools", "-m", "depth", "test.pkl"]):
        with patch("convo_tools.__main__.run_depth") as mock_fn:
            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = MagicMock()
                mock_fn.return_value = None
                assert main() == 0


def test_main_topics() -> None:
    with patch("sys.argv", ["convo-tools", "-m", "topics", "test.pkl"]):
        with patch("convo_tools.__main__.run_topics") as mock_fn:
            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = MagicMock()
                mock_fn.return_value = None
                assert main() == 0


def test_main_centrality() -> None:
    with patch("sys.argv", ["convo-tools", "-m", "centrality", "test.pkl"]):
        with patch("convo_tools.__main__.run_centrality") as mock_fn:
            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = MagicMock()
                mock_fn.return_value = None
                assert main() == 0


def test_main_diff() -> None:
    with patch("sys.argv", ["convo-tools", "-m", "diff", "a.pkl", "b.pkl"]):
        with patch("convo_tools.__main__.run_diff") as mock_fn:
            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = MagicMock()
                mock_fn.return_value = None
                assert main() == 0


def test_main_timeline() -> None:
    with patch("sys.argv", ["convo-tools", "-m", "timeline", "--graph", "g.pkl", "--messages", "m.pkl"]):
        with patch("convo_tools.__main__.run_timeline") as mock_fn:
            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = MagicMock()
                mock_fn.return_value = None
                assert main() == 0


def test_main_similarity() -> None:
    with patch("sys.argv", ["convo-tools", "-m", "similarity", "--graph", "g.pkl", "--messages", "m.pkl"]):
        with patch("convo_tools.__main__.run_similarity") as mock_fn:
            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = MagicMock()
                mock_fn.return_value = None
                assert main() == 0


def test_main_temporal() -> None:
    with patch("sys.argv", ["convo-tools", "-m", "temporal", "--graph", "g.pkl", "--messages", "m.pkl"]):
        with patch("convo_tools.__main__.run_temporal") as mock_fn:
            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = MagicMock()
                mock_fn.return_value = None
                assert main() == 0


def test_main_query() -> None:
    with patch("sys.argv", ["convo-tools", "-m", "query", "--graph", "g.pkl", "--query", "hello"]):
        with patch("convo_tools.__main__.run_query") as mock_fn:
            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = MagicMock()
                mock_fn.return_value = None
                assert main() == 0


def test_main_extract() -> None:
    with patch("sys.argv", ["convo-tools", "-m", "extract", "--json-dir", "/tmp/json", "-p", "out.pkl"]):
        with patch("convo_tools.__main__.run_extract") as mock_fn:
            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = MagicMock()
                mock_fn.return_value = None
                assert main() == 0


def test_main_graph() -> None:
    with patch("sys.argv", ["convo-tools", "-m", "graph", "msg.pkl"]):
        with patch("convo_tools.__main__.run_graph") as mock_fn:
            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = MagicMock()
                mock_fn.return_value = None
                assert main() == 0


def test_main_full() -> None:
    with patch("sys.argv", ["convo-tools", "-m", "full", "--json-dir", "/tmp/json", "-p", "out.pkl"]):
        with patch("convo_tools.__main__.run_extract") as mock_extract:
            with patch("convo_tools.__main__.run_graph") as mock_graph:
                with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                    mock_parse.return_value = MagicMock()
                    mock_extract.return_value = None
                    mock_graph.return_value = None
                    assert main() == 0


def test_main_serve() -> None:
    with patch("sys.argv", ["convo-tools", "-m", "serve"]):
        with patch("convo_tools.__main__.run_serve") as mock_fn:
            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = MagicMock()
                mock_fn.return_value = None
                assert main() == 0
