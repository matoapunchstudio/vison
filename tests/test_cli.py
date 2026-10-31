import pytest

from vision_platform.cli import main


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "vpoc" in capsys.readouterr().out or "usage" in capsys.readouterr().out.lower()


def test_unknown_subcommand_exits_two():
    with pytest.raises(SystemExit) as exc:
        main(["nope"])
    assert exc.value.code == 2


def test_known_subcommands_are_registered(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for name in ("run", "enroll", "bench", "flush"):
        assert name in out
