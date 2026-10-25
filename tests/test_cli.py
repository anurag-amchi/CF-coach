import time

import pytest

from cf_coach.cli import build_parser, exit_code, hours_arg, printer, short


def test_plan_command_arguments_and_defaults():
    args = build_parser().parse_args(["plan", "TitanCoder1258", "4 hours, no geometry"])
    assert (args.handle, args.request) == ("TitanCoder1258", "4 hours, no geometry")
    assert (args.no_sync, args.max_steps, args.model, args.verbose) == (False, 10, None, False)


def test_plan_options():
    args = build_parser().parse_args(
        ["plan", "me", "x", "--no-sync", "--max-steps", "3", "--model", "m", "-v"])
    assert (args.no_sync, args.max_steps, args.model, args.verbose) == (True, 3, "m", True)


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_exit_code_by_outcome():
    assert [exit_code(o) for o in ("saved", "no_plan", "max_steps", "error")] == [0, 0, 1, 1]


def test_short_cuts_long_text():
    assert short("abc", 10) == "abc"
    assert short("x" * 20, 10) == "x" * 7 + "..."


def test_printer_shows_calls_results_and_notes(capsys):
    show = printer(time.monotonic())
    show({"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "get_tag_profile", "arguments": "{}"}}]})
    show({"role": "tool", "tool_call_id": "c1", "content": '{"rating":1400}'})
    show({"role": "user", "content": "Your last tool call was rejected"})
    show({"role": "assistant", "content": "final text"})      # printed later by the CLI
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    assert lines[0].endswith('-> get_tag_profile({})')
    assert lines[1].endswith('<- {"rating":1400}')
    assert lines[2].endswith("!! Your last tool call was rejected")


def test_hours_and_repeated_avoid():
    args = build_parser().parse_args(
        ["plan", "me", "x", "--hours", "4", "--avoid", "geometry", "--avoid", "dfs and similar"])
    assert (args.hours, args.avoid) == (4, ["geometry", "dfs and similar"])
    defaults = build_parser().parse_args(["plan", "me", "x"])
    assert (defaults.hours, defaults.avoid) == (None, [])


def test_hours_out_of_range_is_a_usage_error():
    assert hours_arg("40") == 40
    for bad in ("0", "41", "abc"):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["plan", "me", "x", "--hours", bad])
