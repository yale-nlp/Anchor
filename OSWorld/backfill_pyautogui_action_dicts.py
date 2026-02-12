#!/usr/bin/env python3
"""
Backfill `action_dict` entries for replayed steps that only contain PyAutoGUI code.

This script walks through the generated branch directory, heuristically converts the
stringified PyAutoGUI commands back into the structured `action_dict` format that
downstream consumers expect, and writes the updated trajectories in-place.
"""

from __future__ import annotations

import argparse
import ast
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


# SCREEN_WIDTH = 1920
# SCREEN_HEIGHT = 1080
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 800
ACTION_WIDTH = 1280
ACTION_HEIGHT = 720
X_SCALE = ACTION_WIDTH / SCREEN_WIDTH
Y_SCALE = ACTION_HEIGHT / SCREEN_HEIGHT


PRESS_KEY_MAP = {
    "space": " ",
    "spacebar": " ",
    "comma": ",",
    "period": ".",
    "dot": ".",
    "minus": "-",
    "minus_sign": "-",
    "dash": "-",
    "slash": "/",
    "backslash": "\\",
    "semicolon": ";",
    "colon": ":",
    "quote": "'",
    "apostrophe": "'",
    "equal": "=",
    "plus": "+",
    "underscore": "_",
    # Treat Enter / Return as newline when reconstructing text from press() sequences.
    "enter": "\n",
    "return": "\n",
    # Tab is occasionally used in typing sequences.
    "tab": "\t",
}

SINGLE_KEY_DISPLAY = {
    "return": "Return",
    "enter": "Return",
    "tab": "Tab",
    "space": "Space",
    "spacebar": "Space",
    "esc": "Esc",
    "escape": "Esc",
    "backspace": "Backspace",
    "delete": "Delete",
    "home": "Home",
    "end": "End",
    "pageup": "PageUp",
    "pagedown": "PageDown",
    "pgup": "PageUp",
    "pgdn": "PageDown",
    "insert": "Insert",
    "shift": "Shift",
    "ctrl": "Ctrl",
    "alt": "Alt",
    "option": "Alt",
    "cmd": "Cmd",
    "command": "Cmd",
    "super": "Super",
    "win": "Super",
    "left": "Left",
    "right": "Right",
    "up": "Up",
    "down": "Down",
}

COMBO_NORMALIZE = {
    "return": "return",
    "enter": "return",
    "tab": "tab",
    "esc": "esc",
    "escape": "esc",
    "backspace": "backspace",
    "delete": "delete",
    "space": "space",
    "spacebar": "space",
    "shift": "shift",
    "ctrl": "ctrl",
    "control": "ctrl",
    "alt": "alt",
    "option": "alt",
    "cmd": "cmd",
    "command": "cmd",
    "super": "super",
    "win": "super",
    "pageup": "pageup",
    "pagedown": "pagedown",
    "home": "home",
    "end": "end",
}

IGNORED_PREFIXES = (
    "import pyautogui",
    "import time",
    "import pyperclip",
    "pyautogui.FAILSAFE",
    "pyautogui.PAUSE",
    "Observation:",
    "Thought:",
    "Action:",
    "Plan:",
    "Reflection:",
)


@dataclass
class ParsedCommand:
    name: str
    args: List
    kwargs: dict
    raw: str


class PyAutoGUIConverter:
    def __init__(self) -> None:
        self.failures: dict[str, int] = {}
        self.failed_samples: list[tuple[Path, int, str]] = []

    def convert(self, action: str, *, source: Path, step: int) -> Optional[dict]:
        if not action:
            self._record_failure("EMPTY", source, step, action)
            return None

        normalized_action = self._unwrap_action_container(action)
        result = self._convert_internal(normalized_action)
        stripped = normalized_action.strip()
        if result is None and stripped.upper() == "DONE":
            return None
        if result is None:
            key = stripped[:120] or "UNKNOWN"
            self._record_failure(key, source, step, action)
        return result

    def _record_failure(self, key: str, source: Path, step: int, raw: str) -> None:
        self.failures[key] = self.failures.get(key, 0) + 1
        if len(self.failed_samples) < 25:
            snippet = raw.strip().replace("\n", "\\n")[:200]
            self.failed_samples.append((source, step, snippet))

    def _unwrap_action_container(self, raw_action: str) -> str:
        """
        Some trajectories store a *stringified* dict like:

            "{'action_space': 'pyautogui', 'action': 'pyautogui.click(...)', ...}"

        Normalize these by extracting the inner ``action`` field so that the
        converter can operate on just the PyAutoGUI code.
        """
        stripped = raw_action.strip()
        if not stripped.startswith("{"):
            return raw_action

        # Try to parse as a Python-literal dict first (most common).
        try:
            obj = ast.literal_eval(stripped)
            if isinstance(obj, dict) and isinstance(obj.get("action"), str):
                return obj["action"]
        except Exception:
            pass

        # Fallback: try JSON as well, in case some trajectories use it.
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict) and isinstance(obj.get("action"), str):
                return obj["action"]
        except Exception:
            pass

        return raw_action

    def _convert_internal(self, raw_action: str) -> Optional[dict]:
        stripped = raw_action.strip()
        if not stripped:
            return None

        upper = stripped.upper()
        if upper in ("DONE", "FAIL"):
            return None
        if upper == "WAIT":
            return self._build_wait(raw_action, duration=1)

        clipboard_text = self._extract_clipboard_text(raw_action)
        commands = self._extract_commands(raw_action)

        if not commands and clipboard_text:
            return self._build_type(raw_action, clipboard_text)

        # Drag actions need to be considered before clicks
        drag_cmd = next((cmd for cmd in commands if cmd.name in {"dragTo", "dragRel"}), None)
        if drag_cmd:
            drag_result = self._convert_drag(raw_action, commands, drag_cmd)
            if drag_result:
                return drag_result

        # Scroll actions
        scroll_cmd = next((cmd for cmd in reversed(commands) if cmd.name == "scroll"), None)
        if scroll_cmd:
            scroll_result = self._convert_scroll(raw_action, scroll_cmd)
            if scroll_result:
                return scroll_result

        # Horizontal scroll actions
        hscroll_cmd = next((cmd for cmd in reversed(commands) if cmd.name == "hscroll"), None)
        if hscroll_cmd:
            hscroll_result = self._convert_hscroll(raw_action, hscroll_cmd)
            if hscroll_result:
                return hscroll_result

        # Direct typewrite
        type_cmd = next((cmd for cmd in commands if cmd.name in {"typewrite", "write"}), None)
        if type_cmd:
            text = self._extract_text_arg(type_cmd)
            if text is not None:
                return self._build_type(raw_action, text)

        # Clipboard paste
        if clipboard_text and any(cmd.name == "hotkey" and self._contains_ctrl_v(cmd) for cmd in commands):
            return self._build_type(raw_action, clipboard_text)

        if clipboard_text and self._contains_manual_ctrl_v(commands):
            return self._build_type(raw_action, clipboard_text)

        # Sequences of press commands representing text
        press_text = self._press_sequence_to_text(commands)
        if press_text is not None:
            return self._build_type(raw_action, press_text)

        # Click-like commands (last one wins)
        click_cmd = next(
            (cmd for cmd in reversed(commands) if cmd.name in {"click", "doubleClick", "tripleClick", "rightClick"}),
            None,
        )
        if click_cmd:
            click_result = self._convert_click(raw_action, click_cmd)
            if click_result:
                return click_result

        # Mouse move
        move_cmd = next((cmd for cmd in reversed(commands) if cmd.name in {"moveTo", "moveRel"}), None)
        if move_cmd and len(commands) == 1:
            move_result = self._convert_mouse_move(raw_action, move_cmd)
            if move_result:
                return move_result

        # Key style commands
        key_result = self._convert_key(raw_action, commands)
        if key_result:
            return key_result

        # Waits via sleep
        sleep_cmd = next((cmd for cmd in commands if cmd.name in {"sleep", "pause"}), None)
        if sleep_cmd:
            duration = self._extract_numeric_arg(sleep_cmd, default=1)
            return self._build_wait(raw_action, duration=max(1, int(round(duration))))

        return None

    def _extract_clipboard_text(self, raw_action: str) -> Optional[str]:
        lines = raw_action.replace(";", "\n").splitlines()
        text = None
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("pyperclip.copy"):
                name, args, _, valid = self._safe_parse(stripped.replace("pyperclip.", "pyautogui.", 1))
                if valid and args:
                    text = args[0]
        return text

    def _extract_commands(self, raw_action: str) -> List[ParsedCommand]:
        commands: list[ParsedCommand] = []
        sanitized = raw_action.replace(";", "\n")
        
        # First try to parse the entire action as a single command (handles multi-line strings)
        stripped_full = sanitized.strip()
        if stripped_full.startswith("pyautogui.") and not self._should_ignore(stripped_full):
            name, args, kwargs, valid = self._safe_parse(stripped_full)
            if valid:
                commands.append(ParsedCommand(name, args, kwargs, stripped_full))
                return commands
        
        # Fall back to line-by-line parsing
        in_block = False
        block_delimiter = None
        for line in sanitized.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("'''", '"""', "```")):
                token = stripped[:3]
                if in_block and block_delimiter == token:
                    in_block = False
                    block_delimiter = None
                else:
                    in_block = True
                    block_delimiter = token
                continue
            if in_block:
                continue
            if self._should_ignore(stripped):
                continue
            if not stripped.startswith("pyautogui."):
                continue
            name, args, kwargs, valid = self._safe_parse(stripped)
            if valid:
                commands.append(ParsedCommand(name, args, kwargs, stripped))
        return commands

    def _should_ignore(self, line: str) -> bool:
        for prefix in IGNORED_PREFIXES:
            if line.startswith(prefix):
                return True
        return False

    def _safe_parse(self, line: str) -> tuple[str, list, dict, bool]:
        try:
            node = ast.parse(line).body[0].value  # type: ignore[index]
        except Exception:
            # Try manual extraction for typewrite/write with triple-quoted strings
            if "typewrite" in line or "write" in line:
                manual_result = self._manual_extract_typewrite(line)
                if manual_result:
                    return manual_result
            return "", [], {}, False
        func = getattr(node, "func", None)
        if func is None:
            return "", [], {}, False
        name = ""
        if isinstance(func, ast.Attribute):
            name = func.attr
        elif isinstance(func, ast.Name):
            name = func.id
        args = []
        kwargs = {}
        try:
            for arg in node.args:
                args.append(ast.literal_eval(arg))
            for kw in node.keywords:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            return "", [], {}, False
        return name, args, kwargs, True

    def _manual_extract_typewrite(self, line: str) -> Optional[tuple[str, list, dict, bool]]:
        """Manually extract text from typewrite() calls with triple-quoted strings."""
        import re
        # Match pyautogui.typewrite("""...""", ...) or pyautogui.write("""...""", ...)
        match = re.match(
            r'pyautogui\.(typewrite|write)\s*\(\s*"""(.+?)"""\s*(?:,\s*(.+?))?\s*\)$',
            line,
            re.DOTALL
        )
        if match:
            func_name = match.group(1)
            text_content = match.group(2)
            kwargs_str = match.group(3)
            
            kwargs = {}
            if kwargs_str:
                # Try to parse kwargs like interval=0.01
                for kv_match in re.finditer(r'(\w+)\s*=\s*([^,]+)', kwargs_str):
                    key = kv_match.group(1)
                    val_str = kv_match.group(2).strip()
                    try:
                        kwargs[key] = ast.literal_eval(val_str)
                    except Exception:
                        pass
            
            return func_name, [text_content], kwargs, True
        return None

    def _convert_click(self, raw_action: str, cmd: ParsedCommand) -> Optional[dict]:
        button = cmd.kwargs.get("button", "left")
        clicks = cmd.kwargs.get("clicks")
        x, y = self._extract_xy(cmd)
        if x is None or y is None:
            return None
        coord = self._scale_coordinate(x, y)

        if cmd.name == "doubleClick" or clicks == 2:
            action = "double_click"
        elif cmd.name == "tripleClick" or clicks and clicks >= 3:
            action = "triple_click"
        elif cmd.name == "rightClick" or button == "right":
            action = "right_click"
        else:
            action = "left_click"

        return self._build_tool_input(raw_action, {"action": action, "coordinate": coord})

    def _convert_mouse_move(self, raw_action: str, cmd: ParsedCommand) -> Optional[dict]:
        x, y = self._extract_xy(cmd)
        if x is None or y is None:
            return None
        coord = self._scale_coordinate(x, y)
        return self._build_tool_input(raw_action, {"action": "mouse_move", "coordinate": coord})

    def _convert_drag(
        self, raw_action: str, commands: List[ParsedCommand], drag_cmd: ParsedCommand
    ) -> Optional[dict]:
        drag_index = commands.index(drag_cmd)
        start_coord = None
        for prev in reversed(commands[:drag_index]):
            if prev.name == "moveTo":
                xy = self._extract_xy(prev)
                if xy[0] is not None and xy[1] is not None:
                    start_coord = self._scale_coordinate(xy[0], xy[1])
                    break
        end_coord = None

        if drag_cmd.name == "dragRel":
            dx = drag_cmd.kwargs.get("xOffset")
            dy = drag_cmd.kwargs.get("yOffset")
            if dx is None and drag_cmd.args:
                dx = drag_cmd.args[0]
            if dy is None and len(drag_cmd.args) > 1:
                dy = drag_cmd.args[1]
            if start_coord is not None and dx is not None and dy is not None:
                end_coord = [
                    start_coord[0] + int(round(dx * X_SCALE)),
                    start_coord[1] + int(round(dy * Y_SCALE)),
                ]
        else:
            x, y = self._extract_xy(drag_cmd)
            if x is not None and y is not None:
                end_coord = self._scale_coordinate(x, y)

        if start_coord is None:
            start_coord = end_coord

        if start_coord is None or end_coord is None:
            return None

        return self._build_tool_input(
            raw_action,
            {"action": "left_click_drag", "start_coordinate": start_coord, "coordinate": end_coord},
        )

    def _convert_scroll(self, raw_action: str, cmd: ParsedCommand) -> Optional[dict]:
        amount = self._extract_numeric_arg(cmd, default=0)
        if amount == 0:
            return None
        x = cmd.kwargs.get("x")
        y = cmd.kwargs.get("y")
        if x is None and len(cmd.args) >= 2:
            x = cmd.args[1]
        if y is None and len(cmd.args) >= 3:
            y = cmd.args[2]
        if x is not None and y is not None:
            coord = self._scale_coordinate(x, y)
        else:
            coord = [ACTION_WIDTH // 2, ACTION_HEIGHT // 2]
        direction = "up" if amount > 0 else "down"
        scroll_input = {
            "action": "scroll",
            "coordinate": coord,
            "scroll_direction": direction,
            "scroll_amount": abs(int(round(amount))),
        }
        return self._build_tool_input(raw_action, scroll_input)

    def _convert_hscroll(self, raw_action: str, cmd: ParsedCommand) -> Optional[dict]:
        amount = self._extract_numeric_arg(cmd, default=0)
        if amount == 0:
            return None
        x = cmd.kwargs.get("x")
        y = cmd.kwargs.get("y")
        if x is None and len(cmd.args) >= 2:
            x = cmd.args[1]
        if y is None and len(cmd.args) >= 3:
            y = cmd.args[2]
        if x is not None and y is not None:
            coord = self._scale_coordinate(x, y)
        else:
            coord = [ACTION_WIDTH // 2, ACTION_HEIGHT // 2]
        direction = "right" if amount > 0 else "left"
        scroll_input = {
            "action": "scroll",
            "coordinate": coord,
            "scroll_direction": direction,
            "scroll_amount": abs(int(round(amount))),
        }
        return self._build_tool_input(raw_action, scroll_input)

    def _convert_key(self, raw_action: str, commands: List[ParsedCommand]) -> Optional[dict]:
        if not commands:
            return None

        hotkey_cmd = next((cmd for cmd in reversed(commands) if cmd.name == "hotkey"), None)
        if hotkey_cmd and hotkey_cmd.args:
            return self._build_key(raw_action, [str(arg) for arg in hotkey_cmd.args])

        held_combo = self._extract_held_combo(commands)
        if held_combo:
            return self._build_key(raw_action, held_combo)

        keydown_keys = [str(cmd.args[0]) for cmd in commands if cmd.name == "keyDown" and cmd.args]
        if keydown_keys:
            return self._build_key(raw_action, keydown_keys)

        press_cmds = [cmd for cmd in commands if cmd.name == "press"]
        if len(press_cmds) == 1 and press_cmds[0].args:
            return self._build_key(raw_action, [press_cmds[0].args[0]])
        if press_cmds:
            unique_keys = {str(cmd.args[0]) for cmd in press_cmds if cmd.args}
            if len(unique_keys) == 1:
                return self._build_key(raw_action, [press_cmds[0].args[0]])

        keydown_up = [cmd for cmd in commands if cmd.name in {"keyDown", "keyUp"}]
        if keydown_up:
            unique = []
            for cmd in keydown_up:
                if cmd.args:
                    key = str(cmd.args[0])
                    if key not in unique:
                        unique.append(key)
            if unique:
                return self._build_key(raw_action, unique)

        return None

    def _extract_held_combo(self, commands: List[ParsedCommand]) -> Optional[List[str]]:
        held: list[str] = []
        combo: Optional[List[str]] = None
        for cmd in commands:
            if cmd.name == "keyDown" and cmd.args:
                key = str(cmd.args[0])
                if key not in held:
                    held.append(key)
            elif cmd.name == "press" and cmd.args and held:
                combo = held + [str(cmd.args[0])]
                break
            elif cmd.name == "keyUp" and cmd.args:
                key = str(cmd.args[0])
                if key in held:
                    held.remove(key)
        return combo

    def _press_sequence_to_text(self, commands: List[ParsedCommand]) -> Optional[str]:
        if not commands or any(cmd.name != "press" for cmd in commands):
            return None
        chars = []
        for cmd in commands:
            if not cmd.args:
                return None
            key = str(cmd.args[0])
            char = self._key_to_char(key)
            if char is None:
                return None
            chars.append(char)
        if not chars:
            return None
        return "".join(chars)

    def _contains_ctrl_v(self, cmd: ParsedCommand) -> bool:
        if cmd.name != "hotkey":
            return False
        lowered = [str(arg).lower() for arg in cmd.args]
        return "ctrl" in lowered and "v" in lowered

    def _contains_manual_ctrl_v(self, commands: List[ParsedCommand]) -> bool:
        ctrl_held = False
        for cmd in commands:
            if cmd.name == "keyDown" and cmd.args:
                key = str(cmd.args[0]).lower()
                if key in {"ctrl", "control"}:
                    ctrl_held = True
                elif ctrl_held and key == "v":
                    return True
            elif cmd.name == "press" and cmd.args and ctrl_held:
                if str(cmd.args[0]).lower() == "v":
                    return True
            elif cmd.name == "keyUp" and cmd.args:
                key = str(cmd.args[0]).lower()
                if key in {"ctrl", "control"}:
                    ctrl_held = False
        return False

    def _extract_text_arg(self, cmd: ParsedCommand) -> Optional[str]:
        if cmd.args:
            first = cmd.args[0]
            if isinstance(first, str):
                return first
        return None

    def _extract_xy(self, cmd: ParsedCommand, fallback: Optional[tuple[int, int]] = None) -> tuple[Optional[float], Optional[float]]:
        x = cmd.kwargs.get("x")
        y = cmd.kwargs.get("y")
        if x is None and cmd.args:
            x = cmd.args[0]
        if y is None and len(cmd.args) > 1:
            y = cmd.args[1]
        if x is None or y is None:
            if fallback:
                return fallback
        return x, y

    def _extract_numeric_arg(self, cmd: ParsedCommand, default: float = 0) -> float:
        if cmd.args:
            try:
                return float(cmd.args[0])
            except (ValueError, TypeError):
                pass
        value = next(iter(cmd.kwargs.values()), default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _scale_coordinate(self, x: float, y: float) -> list[int]:
        return [int(round(float(x) * X_SCALE)), int(round(float(y) * Y_SCALE))]

    def _build_tool_input(self, raw_action: str, input_payload: dict) -> dict:
        return {
            "name": "computer",
            "input": input_payload,
            "id": f"backfill_{uuid.uuid4().hex}",
            "action_type": "tool_use",
            "command": raw_action if raw_action.endswith("\n") else f"{raw_action}\n",
        }

    def _build_type(self, raw_action: str, text: str) -> dict:
        return self._build_tool_input(raw_action, {"action": "type", "text": text})

    def _build_wait(self, raw_action: str, duration: int) -> dict:
        return self._build_tool_input(raw_action, {"action": "wait", "duration": duration})

    def _build_key(self, raw_action: str, keys: Iterable[str]) -> Optional[dict]:
        keys = list(keys)
        if not keys:
            return None
        if len(keys) == 1:
            key_text = self._format_single_key(keys[0])
        else:
            key_text = "+".join(self._normalize_combo_key(k) for k in keys)
        return self._build_tool_input(raw_action, {"action": "key", "text": key_text})

    def _format_single_key(self, key: str) -> str:
        lowered = key.strip().strip("'\"").lower()
        if lowered in SINGLE_KEY_DISPLAY:
            return SINGLE_KEY_DISPLAY[lowered]
        if len(lowered) == 1:
            return lowered
        return key.strip()

    def _normalize_combo_key(self, key: str) -> str:
        lowered = key.strip().strip("'\"").lower()
        if lowered in COMBO_NORMALIZE:
            return COMBO_NORMALIZE[lowered]
        return lowered

    def _key_to_char(self, key: str) -> Optional[str]:
        if not isinstance(key, str):
            return None
        if len(key) == 1:
            return key
        lowered = key.lower()
        if lowered in PRESS_KEY_MAP:
            return PRESS_KEY_MAP[lowered]
        return None


def process_trajectory(path: Path, converter: PyAutoGUIConverter, dry_run: bool) -> int:
    updated_records = []
    updated = 0

    with path.open(encoding="utf-8") as f:
        for step_index, line in enumerate(f, 1):
            record = json.loads(line)
            action_dict = record.get("action_dict")
            if not action_dict:
                converted = converter.convert(record.get("action", ""), source=path, step=step_index)
                if converted:
                    record["action_dict"] = converted
                    updated += 1
            updated_records.append(record)

    if updated and not dry_run:
        tmp_path = path.with_suffix(".jsonl.tmp")
        with tmp_path.open("w", encoding="utf-8") as tmp_file:
            for record in updated_records:
                tmp_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp_path.replace(path)

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill action_dict entries for replayed steps.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/branch_gen_winarena_half_verified"),
        help="Directory containing branch trajectories.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute conversions without rewriting files.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N trajectories.")
    args = parser.parse_args()

    converter = PyAutoGUIConverter()
    trajectory_paths = sorted(args.base_dir.rglob("trajectory.jsonl"))

    total_converted = 0
    files_touched = 0

    for idx, traj_path in enumerate(trajectory_paths):
        if args.limit is not None and idx >= args.limit:
            break
        converted = process_trajectory(traj_path, converter, args.dry_run)
        if converted:
            files_touched += 1
            total_converted += converted

    print(f"Processed {min(len(trajectory_paths), args.limit or len(trajectory_paths))} trajectory files.")
    print(f"Updated {files_touched} files and backfilled {total_converted} actions.")

    if converter.failures:
        print("Unconverted action snippets (top 10):")
        for snippet, count in list(converter.failures.items())[:10]:
            print(f"  [{count}] {snippet}")
        if converter.failed_samples:
            print("\nSample locations:")
            for path, step, snippet in converter.failed_samples:
                print(f"  {path} (step {step}): {snippet}")


if __name__ == "__main__":
    main()

