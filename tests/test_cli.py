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


def test_flush_missing_outbox_errors_without_creating_db(tmp_path):
    db = tmp_path / "missing.db"
    rc = main(["flush", "--outbox", str(db), "--webhook", "http://x.invalid/h"])
    assert rc == 2
    assert not db.exists()


def test_run_rejects_nonpositive_max_frames(tmp_path):
    rc = main(["run", "--pipeline", "pipelines/attendance_poc.yaml",
               "--source", "whatever.mp4", "--max-frames", "0",
               "--outbox", str(tmp_path / "o.db")])
    assert rc == 2
    assert not (tmp_path / "o.db").exists()


def test_known_subcommands_are_registered(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for name in ("run", "enroll", "bench", "flush"):
        assert name in out
