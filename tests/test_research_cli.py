"""Tests for the headless tradingresearch CLI."""

import pytest

pytestmark = pytest.mark.unit


def test_help_includes_required_flags(capsys):
    from cli.research import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--ticker", "--date", "--output-dir", "--token-source",
                 "--openclaw-profile-path", "--openclaw-profile-name",
                 "--deep", "--quick", "--debate-rounds", "--risk-rounds"):
        assert flag in out, f"flag {flag} missing from --help"


def test_parse_minimal_args_succeeds():
    from cli.research import build_parser

    parser = build_parser()
    ns = parser.parse_args(["--ticker", "NVDA", "--date", "2024-05-10",
                            "--output-dir", "/tmp/out"])
    assert ns.ticker == "NVDA"
    assert ns.date == "2024-05-10"
    assert ns.output_dir == "/tmp/out"
    assert ns.token_source == "keychain"  # default


def test_missing_required_arg_exits():
    from cli.research import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--ticker", "NVDA"])  # missing --date and --output-dir
