from datasets import Dataset
import re
import random
import torch
import os
from PIL import Image
from tqdm import tqdm
import traceback
import json
import copy
from typing import Optional


def create_branch_generated_dataset(
    branch_root: str,
    dual_training_types: bool = True,
    half_verified_root: Optional[str] = None,
) -> Dataset:
    """
    Create a HuggingFace Dataset from branch-generated trajectories.

    Args:
        branch_root: Root directory containing branch subdirectories
        dual_training_types: If True, creates both Type 1 and Type 2 examples for each step,
                           doubling the dataset size. If False, only creates Type 1 examples.

    Expected directory structure under `branch_root`:
        branch_root/
            <branch_id_1>/
                metadata.json
                trajectory.jsonl
                screenshots/
                    step_1.png
                    step_1_replay.png
                    step_2.png
                    step_2_replay.png
                    ...
            <branch_id_2>/
                ...

    Each dataset example corresponds to a single step in a branch trajectory and contains:
        - task_description: overall natural language description of the task
        - branch_dir: absolute path to the branch directory
        - history: concatenated reasoning strings from all previous steps in this branch
        - step: a dict describing the current step, with:
            - step: original integer step id from the trajectory (1-based)
            - is_replay: True if this step comes from the initial replay segment, False otherwise
            - reasoning: optional natural language reasoning for the step
            - action_proposal: optional short imperative description of the action
            - action_dict: the high-level action dictionary used by the agent
            - reward: scalar reward
            - done: bool flag
        - all_steps: list of dicts for all steps in this branch (same schema as `step`)
        - current_step_idx: integer index into `all_steps` for the current step
        - training_type: "type1" or "type2" indicating which training format to use
    """
    examples = []

    if not os.path.isdir(branch_root):
        raise ValueError(f"branch_root does not exist or is not a directory: {branch_root}")

    if half_verified_root is not None and not os.path.isdir(half_verified_root):
        raise ValueError(
            f"half_verified_root was provided but does not exist or is not a directory: {half_verified_root}"
        )

    # ------------------------------------------------------------------
    # First pass: collect all branches and their metadata so we can
    # determine, for LibreOffice domains, which branches should KEEP
    # replay steps and which should DROP them.
    #
    # We treat:
    #   - entries from `branch_root` as source == "main"
    #   - entries from `half_verified_root` (if provided) as source == "half_verified"
    #
    # The LibreOffice "max branch keeps replay" logic only looks at
    # branches from the main root so that adding half-verified data
    # does not change the behavior of existing training branches.
    # ------------------------------------------------------------------
    branch_infos = []

    def _collect_branch_infos(root: str, source: str) -> None:
        for dir_name in sorted(os.listdir(root)):
            dir_path = os.path.join(root, dir_name)
            if not os.path.isdir(dir_path):
                continue

            metadata_path = os.path.join(dir_path, "metadata.json")
            traj_path = os.path.join(dir_path, "trajectory.jsonl")
            screenshots_dir = os.path.join(dir_path, "screenshots")

            if not (os.path.isfile(metadata_path) and os.path.isfile(traj_path)):
                # Skip directories that do not contain both metadata and trajectory
                continue

            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"[create_branch_generated_dataset] Failed to read {metadata_path}: {e}")
                continue

            branch_infos.append(
                {
                    "dir_path": dir_path,
                    "metadata_path": metadata_path,
                    "traj_path": traj_path,
                    "screenshots_dir": screenshots_dir,
                    "metadata": metadata,
                    "source": source,
                }
            )

    # Always collect from the main root
    _collect_branch_infos(branch_root, source="main")
    # Optionally, also collect half-verified branches
    if half_verified_root is not None:
        _collect_branch_infos(half_verified_root, source="half_verified")

    # Map from branch dir to whether we should keep replay steps when
    # building the training data. Default is True (keep replays).
    include_replay_for_branch: dict[str, bool] = {}

    # Only apply the "drop replay steps for non-max branches" logic to
    # LibreOffice domains, grouped by original_task_id, and *only* for
    # branches coming from the main root. Half-verified branches are
    # always treated as "drop replay, keep only post-branch steps".
    # target_domains = {"libreoffice_calc", "libreoffice_impress", "libreoffice_writer"}
    target_domains = {}
    branches_by_original_id: dict[str, list[dict]] = {}

    for info in branch_infos:
        metadata = info["metadata"]
        if info.get("source") != "main":
            # Only main-root branches participate in the max-branch selection.
            continue
        domain = metadata.get("domain")
        if domain not in target_domains:
            continue

        original_task_id = metadata.get("original_task_id")
        if not original_task_id:
            continue

        branch_after_step_raw = metadata.get("branch_after_step")
        branch_after_step_int = int(branch_after_step_raw)

        info["branch_after_step_int"] = branch_after_step_int
        # Group branches that share the same original_task_id
        branches_by_original_id.setdefault(str(original_task_id), []).append(info)

    for _task_id, infos in branches_by_original_id.items():
        max_branch_step = max(i["branch_after_step_int"] for i in infos)
        for info in infos:
            dir_path = info["dir_path"]
            include_replay_for_branch[dir_path] = info["branch_after_step_int"] == max_branch_step

    # ------------------------------------------------------------------
    # Second pass: actually build per-step training examples, using the
    # include_replay_for_branch map to optionally drop replay steps for
    # non-max LibreOffice branches.
    # ------------------------------------------------------------------
    for info in branch_infos:
        dir_path = info["dir_path"]
        metadata = info["metadata"]
        traj_path = info["traj_path"]
        screenshots_dir = info["screenshots_dir"]

        # For LibreOffice domains that are not the max-branch for a given
        # original_task_id, we drop replay steps from training and history.
        # Look up this flag before deciding which task description to use.
        include_replay = include_replay_for_branch.get(dir_path, True)

        source = info.get("source", "main")
        domain = metadata.get("domain")

        # Half-verified branches:
        #   - always drop replay steps (use only post-branch steps)
        #   - always prefer the human-edited `new_task_description`
        if source == "half_verified":
            include_replay = False
            task_description = metadata.get(
                "new_task_description",
                metadata.get("generated_task_description_from_vllm", ""),
            )
        else:
            # For non-max LibreOffice branches (where we drop replay steps),
            # use the human-edited `new_task_description` instead of the
            # VLLM-generated one. For all other branches, keep the original
            # behavior.
            if not include_replay and domain in target_domains:
                task_description = metadata.get(
                    "new_task_description",
                    metadata.get("generated_task_description_from_vllm", ""),
                )
            else:
                # Prefer the VLLM-generated description when available; if it's missing
                # or empty, fall back to the human-edited `new_task_description`.
                task_description = metadata.get("generated_task_description_from_vllm") or metadata.get(
                    "new_task_description", ""
                )
        num_replay_steps = int(metadata.get("num_replay_steps", 0))

        # For replay steps, we may have additional metadata describing how the
        # reasoning was backfilled (e.g., "candidate_match" vs "fallback_direct").
        # We only want to create supervised training targets from replay steps
        # whose backfill method is "candidate_match", but we still want ALL
        # replay steps to appear in the trajectory/history so that later steps
        # can see their reasoning as context.
        replay_reasoning_backfill = metadata.get("replay_reasoning_backfill", {}) or {}
        replay_updated_steps = replay_reasoning_backfill.get("updated_steps", {}) or {}
        # Map from 1-based replay step index -> backfill method string
        replay_method_by_index: dict[int, str] = {}
        for k, v in replay_updated_steps.items():
            try:
                idx_int = int(k)
            except (TypeError, ValueError):
                continue
            method = v.get("method")
            if isinstance(method, str):
                replay_method_by_index[idx_int] = method

        # First, parse all steps for this branch.
        steps = []
        line_index = 0
        try:
            with open(traj_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"[create_branch_generated_dataset] JSON decode error in {traj_path}: {e}")
                        continue
                    line_index += 1

                    # Determine whether this is part of the initial replay segment.
                    # By construction, all replay steps come first in the trajectory.
                    is_replay = line_index <= num_replay_steps

                    # For non-max LibreOffice branches we skip replay steps entirely:
                    #   - they will not get supervised targets
                    #   - they will not appear in `all_steps` or textual history
                    # The collator will still use the last replay screenshot as
                    # the initial observation via its screenshot-loading logic.
                    if is_replay and not include_replay:
                        continue

                    # Normalize action_dict:
                    # - use record["action_dict"] if present
                    # - for terminal DONE steps without action_dict, synthesize one
                    # - otherwise keep an empty dict (you can fill these in later)
                    has_action_dict = "action_dict" in record
                    is_terminal_done = (
                        record.get("terminal") is True
                        and record.get("action") == "DONE"
                    )

                    if has_action_dict:
                        action_dict = record.get("action_dict") or {}
                    elif is_terminal_done:
                        action_dict = {"terminal": True, "status": "DONE"}
                    else:
                        action_dict = {}

                    step_id = record.get("step")
                    if step_id is None:
                        step_id = line_index

                    # Decide whether to skip creating training examples for this step.
                    # - We always keep the step in `steps` (so it shows up in
                    #   `all_steps` and its reasoning can contribute to history).
                    # - If skip=True, we will not create supervised training
                    #   examples for this step later on.
                    skip_flag = record.get("skip", False)
                    if is_replay:
                        # For replay steps, if the backfill method is present and
                        # not "candidate_match" (e.g., "fallback_direct"), then we
                        # only want this step for context, not as a supervised
                        # training target.
                        replay_method = replay_method_by_index.get(line_index)
                        if replay_method is not None and replay_method != "candidate_match":
                            skip_flag = True

                    step_entry = {
                        "step": step_id,
                        "is_replay": is_replay,
                        "reasoning": record.get("reasoning", ""),
                        # Short imperative description of the action, if present.
                        # This is preferred over `reasoning` when building the
                        # supervised "Action: ..." target during training.
                        "action_proposal": record.get("action_proposal", ""),
                        "action_dict": action_dict,
                        "reward": record.get("reward", 0),
                        "done": record.get("done", False),
                        # If skip is True, we still include this step in the trajectory
                        # (for context/history), but don't create training examples from it
                        "skip": skip_flag,
                    }
                    steps.append(step_entry)
        except Exception as e:
            print(f"[create_branch_generated_dataset] Failed to read {traj_path}: {e}")
            continue

        # Require at least one usable step and at least one screenshot directory
        if not steps:
            continue
        if not os.path.isdir(screenshots_dir):
            print(
                f"[create_branch_generated_dataset] No screenshots directory found for branch {dir_path}, "
                "skipping this branch."
            )
            continue

        # Flatten into one example per step, and precompute textual history for each.
        history_reasonings: list[str] = []
        abs_branch_dir = os.path.abspath(dir_path)
        num_steps = len(steps)
        for idx, step in enumerate(steps):
            history_text = "\n".join(r for r in history_reasonings if r)
            is_last_step = idx == num_steps - 1

            # For the final step in the trajectory, force a standardized
            # action_proposal so the model always sees a clear terminal message.
            if is_last_step:
                step = step.copy()
                step["action_proposal"] = "The task is completed successfully."
                steps[idx] = step

            # Skip creating training examples for steps marked with skip=True.
            # These steps remain in the trajectory for context (history and screenshots),
            # but we don't generate supervised targets from them.
            should_skip = step.get("skip", False)
            
            if not should_skip:
                # Create training examples for all non-skipped steps
                base_example = {
                    "task_description": task_description,
                    "branch_dir": abs_branch_dir,
                    "history": history_text,
                    "step": step.copy(),
                    # Provide full trajectory and index so the collator can
                    # reconstruct multi-step, multi-image chat histories.
                    "all_steps": steps,
                    "current_step_idx": idx,
                }
                
                # Create Type 1 example: predict action_proposal + action
                example_type1 = base_example.copy()
                example_type1["training_type"] = "type1"
                examples.append(example_type1)
                
                # If dual_training_types is enabled, also create Type 2 example
                if dual_training_types:
                    # Create Type 2 example: given action_proposal, predict action only
                    example_type2 = base_example.copy()
                    example_type2["training_type"] = "type2"
                    examples.append(example_type2)

            # Always add reasoning to history (even for skipped steps)
            # so that later steps have the full context
            r = step.get("reasoning", "")
            if r:
                history_reasonings.append(r)

    if not examples:
        raise ValueError(
            f"No valid branch trajectories were found under {branch_root}. "
            "Please check that the directory contains subfolders with metadata.json, "
            "trajectory.jsonl, and a screenshots/ subdirectory."
        )

    print(f"[create_branch_generated_dataset] Loaded {len(examples)} steps from branches under {branch_root}")
    return Dataset.from_list(examples)


def _parse_agentnet_code_to_action_dict(code: str) -> dict:
    """
    Best-effort conversion from AgentNet's PyAutoGUI-style code string into the
    branch-generated `action_dict` format expected by the collators.

    The AgentNet dataset (see: https://huggingface.co/datasets/xlangai/AgentNet)
    stores executable code snippets such as:
        pyautogui.click(x=0.1632, y=0.2711)
        pyautogui.moveTo(x=0.42, y=0.55)
        pyautogui.doubleClick(x=0.5, y=0.5)
        pyautogui.typewrite("hello")
        pyautogui.hotkey("ctrl", "c")
        pyautogui.scroll(-500)
        time.sleep(1.5)

    This helper maps a subset of common patterns into the high-level action
    schema used by `create_branch_generated_dataset` and the branch collators.
    If parsing fails, it returns an empty dict and the step will not have a
    structured tool-call target (only text supervision).
    """

    if not isinstance(code, str):
        return {}

    code = code.strip()
    if not code:
        return {}

    # Some snippets may contain multiple statements separated by newlines or semicolons.
    # We only look at the first statement for now.
    first_stmt = re.split(r"[;\n]", code, maxsplit=1)[0].strip()

    # time.sleep(...) → wait
    m_sleep = re.match(r"(?:time|pyautogui)\.sleep\((?P<secs>[^)]*)\)", first_stmt)
    if m_sleep:
        secs_raw = m_sleep.group("secs").strip()
        try:
            secs = float(eval(secs_raw))  # nosec - trusted dataset-only parsing
        except Exception:
            secs = 1.0
        return {
            "input": {
                "action": "wait",
                "time": secs,
            }
        }

    # Generic pyautogui.<fn>(...)
    m = re.match(r"pyautogui\.(?P<fn>\w+)\((?P<args>.*)\)", first_stmt)
    if not m:
        return {}

    fn = m.group("fn").lower()
    args_str = m.group("args").strip()

    # Parse keyword arguments into a dict in a very small, targeted way.
    kwargs: dict = {}
    if args_str:
        # Split on commas that are not inside quotes.
        parts = re.split(r",(?![^\"']*[\"'])", args_str)
        positional_args = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if "=" in p:
                k, v = p.split("=", 1)
                kwargs[k.strip()] = v.strip()
            else:
                positional_args.append(p)

    def _eval_safe(expr: str):
        try:
            return eval(expr, {"__builtins__": {}}, {})  # nosec - dataset-only
        except Exception:
            return expr

    def _parse_coord_from_kwargs():
        if "x" in kwargs and "y" in kwargs:
            x_raw, y_raw = kwargs["x"], kwargs["y"]
            x_v = _eval_safe(x_raw)
            y_v = _eval_safe(y_raw)
            try:
                x_f = float(x_v)
                y_f = float(y_v)
            except Exception:
                return None

            # AgentNet uses normalized coordinates in [0,1]; detect that and
            # map to the 1280x720 pixel space used by branch-generated data.
            if 0.0 <= x_f <= 1.0 and 0.0 <= y_f <= 1.0:
                base_w, base_h = 1280.0, 720.0
                x_f *= base_w
                y_f *= base_h

            return [x_f, y_f]
        return None

    # CLICK / DOUBLE CLICK / RIGHT CLICK
    if fn in {"click", "doubleclick", "tripleclick"}:
        coord = _parse_coord_from_kwargs()

        button = None
        if "button" in kwargs:
            button = str(_eval_safe(kwargs["button"])).lower()

        clicks = None
        for k in ("clicks", "num_clicks"):
            if k in kwargs:
                clicks = _eval_safe(kwargs[k])
                break

        if fn == "doubleclick" or clicks == 2:
            action = "double_click"
        elif fn == "tripleclick" or clicks == 3:
            action = "triple_click"
        else:
            # Single click; infer left/right from button if present.
            if button == "right":
                action = "right_click"
            elif button == "middle":
                action = "middle_click"
            else:
                action = "left_click"

        input_dict: dict = {"action": action}
        if coord is not None:
            input_dict["coordinate"] = coord
        return {"input": input_dict}

    # moveTo / moveRel → mouse_move
    if fn in {"moveto", "moverel"}:
        coord = _parse_coord_from_kwargs()
        if coord is None:
            # Some data may pass (x, y) as positional args
            if len(kwargs) == 0 and args_str:
                try:
                    # Fallback: eval as a tuple like (0.5, 0.5)
                    maybe_tuple = _eval_safe(args_str)
                    if isinstance(maybe_tuple, (list, tuple)) and len(maybe_tuple) == 2:
                        x_f = float(maybe_tuple[0])
                        y_f = float(maybe_tuple[1])
                        if 0.0 <= x_f <= 1.0 and 0.0 <= y_f <= 1.0:
                            x_f *= 1280.0
                            y_f *= 720.0
                        coord = [x_f, y_f]
                except Exception:
                    coord = None
        if coord is None:
            return {}
        return {"input": {"action": "mouse_move", "coordinate": coord}}

    # dragTo / dragRel → left_click_drag
    if fn in {"dragto", "dragrel"}:
        coord = _parse_coord_from_kwargs()
        input_dict = {"action": "left_click_drag"}
        if coord is not None:
            input_dict["coordinate"] = coord
        # Duration, if present
        if "duration" in kwargs:
            dur = _eval_safe(kwargs["duration"])
            try:
                input_dict["duration"] = float(dur)
            except Exception:
                pass
        return {"input": input_dict}

    # typewrite / write → type
    if fn in {"typewrite", "write"}:
        text = ""
        if "message" in kwargs:
            text = _eval_safe(kwargs["message"])
        elif "text" in kwargs:
            text = _eval_safe(kwargs["text"])
        elif args_str:
            text = _eval_safe(args_str)
        return {"input": {"action": "type", "text": str(text)}}

    # hotkey / keyDown / keyUp / press → key
    if fn in {"hotkey", "keydown", "keyup", "press"}:
        keys = []
        # Keys in keyword arguments (e.g., keys=['ctrl', 'c'])
        if "keys" in kwargs:
            val = _eval_safe(kwargs["keys"])
            if isinstance(val, (list, tuple)):
                keys = [str(k) for k in val]
            elif isinstance(val, str):
                keys = [val]
        elif args_str:
            # Positional string args, e.g., "ctrl", "c"
            parts = re.split(r",(?![^\"']*[\"'])", args_str)
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                v = _eval_safe(p)
                if isinstance(v, str):
                    keys.append(v)
        return {"input": {"action": "key", "keys": keys}}

    # scroll(amount, ...) → scroll
    if fn == "scroll":
        amount = 0
        if args_str:
            first_arg = re.split(r",(?![^\"']*[\"'])", args_str)[0].strip()
            amount = _eval_safe(first_arg)
        try:
            amount_int = int(amount)
        except Exception:
            amount_int = 0
        direction = "down" if amount_int < 0 else "up"
        return {
            "input": {
                "action": "scroll",
                "scroll_amount": abs(amount_int),
                "scroll_direction": direction,
            }
        }

    # Fallback: unknown action
    return {}


def create_branch_generated_dataset_from_human(
    traj_json_path: str = "/AgentNet/agentnet_ubuntu_5k.jsonl",
    image_root: str = "/AgentNet/ubuntu_images",
    dual_training_types: bool = True,
    max_tasks: Optional[int] = None,
    max_steps_per_traj: Optional[int] = 30,
) -> Dataset:
    """
    Build a HuggingFace Dataset from **local** human-annotated AgentNet trajectories,
    matching the *schema* of `create_branch_generated_dataset`.

    The local trajectory file (`traj_json_path`) is expected to be either:
        - a JSON file with a top-level list of task records, or
        - a JSONL file with one task record per line.

    Each AgentNet `task` record is treated as a "branch", and each trajectory
    step becomes a step entry with (optionally truncated to at most
    `max_steps_per_traj` steps per trajectory if that argument is not None):
        - task_description: taken from natural_language_task / instruction / actual_task
        - branch_dir: empty string (no on-disk branch directory; images are not used here)
        - history: concatenated step-level reasoning from previous steps
        - step: dict with:
            - step: integer step id (AgentNet's `index` or 1-based position)
            - is_replay: always False (AgentNet has no replay phase)
            - reasoning: AgentNet `thought`
            - action_proposal: AgentNet natural language `action`
            - action_dict: best-effort parse of AgentNet `code` into our action space
            - reward: 1 for final successful step, else 0
            - done: True for final step if task_completed, else False
            - skip: True for incorrect or redundant steps (kept in history only)
        - all_steps: list of such step dicts
        - current_step_idx: index into all_steps for this example
        - training_type: "type1" / "type2" as in `create_branch_generated_dataset`.

    Each step will also contain:
        - image: the original image filename from AgentNet (e.g. "xxxx.png")
        - image_path: absolute path to the corresponding PNG file, constructed
          as os.path.join(image_root, <step_image_filename>).
    """

    # --------------------------------------------------------------
    # Load local AgentNet trajectories from JSON / JSONL file.
    # --------------------------------------------------------------
    if not os.path.isfile(traj_json_path):
        raise ValueError(f"traj_json_path does not exist or is not a file: {traj_json_path}")

    records = []
    with open(traj_json_path, "r", encoding="utf-8") as f:
        # Peek at the first non-whitespace character to guess format.
        start_pos = f.tell()
        first_non_ws = ""
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_non_ws = ch
                break
        f.seek(start_pos)

        if first_non_ws == "[":
            # Standard JSON list
            try:
                records = json.load(f)
            except Exception as e:
                raise ValueError(f"Failed to parse JSON list from {traj_json_path}: {e}")
        else:
            # Assume JSONL: one JSON object per line
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)
                except Exception as e:
                    print(f"[create_branch_generated_dataset_from_human] Failed to parse line from {traj_json_path}: {e}")
                    continue

    if not records:
        raise ValueError(f"No records loaded from {traj_json_path}")

    examples = []

    for task_idx, record in enumerate(records):
        if max_tasks is not None and task_idx >= max_tasks:
            break

        # Task-level description
        task_description = (
            record.get("natural_language_task")
            or record.get("instruction")
            or record.get("actual_task")
            or ""
        )

        traj_full = record.get("traj") or record.get("trajectory") or []
        if not traj_full:
            continue

        if max_steps_per_traj is not None:
            traj = traj_full[: max_steps_per_traj]
        else:
            traj = traj_full

        task_completed = bool(record.get("task_completed", False))

        steps = []
        history_reasonings = []

        for idx, step_entry in enumerate(traj):
            value = step_entry.get("value", {}) or {}
            thought = value.get("thought", "") or ""
            action_nl = value.get("action", "") or ""
            code = value.get("code", "") or ""

            last_step_correct = value.get("last_step_correct", True)
            last_step_redundant = value.get("last_step_redundant", False)

            step_id = step_entry.get("index")
            if step_id is None:
                step_id = idx + 1

            is_last_step = idx == len(traj) - 1
            done_flag = bool(is_last_step and task_completed)

            action_dict = _parse_agentnet_code_to_action_dict(code)
            if done_flag and (not action_dict or not action_dict.get("terminal")):
                # Ensure we mark the final successful step as terminal.
                action_dict = {"terminal": True, "status": "DONE"}

            # Mark steps as "skip" (for context only, no supervised target)
            # if they were annotated as incorrect or redundant.
            skip_flag = False

            # Compute on-disk image path from the local image root.
            raw_image_name = step_entry.get("image")
            abs_image_root = os.path.abspath(image_root)
            abs_image_path = (
                os.path.join(abs_image_root, raw_image_name) if raw_image_name else None
            )

            step_out = {
                "step": step_id,
                "is_replay": False,
                "reasoning": thought,
                "action_proposal": action_nl,
                "action_dict": action_dict or {},
                "reward": 1 if done_flag else 0,
                "done": done_flag,
                "skip": skip_flag,
                # Keep AgentNet's original image filename and (optionally) the
                # resolved absolute image_path for downstream collators.
                "image": raw_image_name,
                "image_path": abs_image_path,
            }
            steps.append(step_out)

        if not steps:
            continue

        # Use the directory containing the trajectory JSON as a synthetic
        # "branch root", so collators can still retrieve a directory path.
        abs_branch_dir = os.path.dirname(os.path.abspath(traj_json_path))
        num_steps = len(steps)

        for idx, step in enumerate(steps):
            history_text = "\n".join(r for r in history_reasonings if r)
            is_last_step = idx == num_steps - 1

            # Mirror `create_branch_generated_dataset` behaviour for the final step.
            if is_last_step:
                step = step.copy()
                step["action_proposal"] = "The task is completed successfully."
                steps[idx] = step

            should_skip = step.get("skip", False)

            if not should_skip:
                base_example = {
                    "task_description": task_description,
                    "branch_dir": abs_branch_dir,
                    "history": history_text,
                    "step": step.copy(),
                    "all_steps": steps,
                    "current_step_idx": idx,
                }

                # Type 1 example
                example_type1 = base_example.copy()
                example_type1["training_type"] = "type1"
                examples.append(example_type1)

                # Optional Type 2 example
                if dual_training_types:
                    example_type2 = base_example.copy()
                    example_type2["training_type"] = "type2"
                    examples.append(example_type2)

            # Always extend textual history with this step's reasoning.
            r = step.get("reasoning", "")
            if r:
                history_reasonings.append(r)

    if not examples:
        raise ValueError(
            "No valid examples were built from the local AgentNet trajectories. "
            "Please check that the file is readable and has non-empty trajectories."
        )

    print(
        f"[create_branch_generated_dataset_from_human] Loaded {len(examples)} "
        f"step examples from local AgentNet file: {traj_json_path}"
    )
    return Dataset.from_list(examples)


def create_dataset_from_hf(
    hf_dataset_name: str,
    os_filter: Optional[str] = None,
    dual_training_types: bool = True,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """
    Load a trajectory dataset from HuggingFace Hub and restructure it for training.
    
    The HuggingFace dataset is expected to have per-step examples with:
        - trajectory_id: unique trajectory identifier
        - os_type: "windows" or "ubuntu"
        - task_description: task description
        - step_number: sequential step number (1, 2, 3, ...)
        - total_steps: total steps in trajectory
        - reasoning: reasoning for current step
        - action_proposal: action proposal for current step
        - action_dict: JSON string of action dict
        - image: PIL Image for this step
        - domain: task domain
    
    This function restructures the data to match the format expected by the
    training collator. It uses a memory-efficient approach by:
    1. First pass: group by trajectory_id using only indices (no image copying)
    2. Second pass: build training examples with references to original dataset
    
    Args:
        hf_dataset_name: HuggingFace dataset name (e.g., "mikeweii/ANCHOR")
        os_filter: Optional filter for OS type ("windows", "ubuntu", or None for both)
        dual_training_types: If True, creates both Type 1 and Type 2 examples
        cache_dir: Optional cache directory for HuggingFace datasets
    
    Returns:
        Dataset with restructured examples ready for training
    """
    from datasets import load_dataset
    from collections import defaultdict
    import gc
    
    print(f"[create_dataset_from_hf] Loading dataset from {hf_dataset_name}...")
    
    # Load the dataset from HuggingFace
    hf_dataset = load_dataset(hf_dataset_name, cache_dir=cache_dir, split="train")
    
    print(f"[create_dataset_from_hf] Loaded {len(hf_dataset)} raw examples")
    
    # Filter by OS if specified - use efficient column-based filtering
    # to avoid decoding images during filter
    if os_filter:
        # Get indices of matching examples efficiently using only the os_type column
        print(f"[create_dataset_from_hf] Filtering by OS '{os_filter}'...")
        os_types = hf_dataset["os_type"]  # Get just the os_type column (fast)
        matching_indices = [i for i, os_type in enumerate(os_types) if os_type == os_filter]
        hf_dataset = hf_dataset.select(matching_indices)
        print(f"[create_dataset_from_hf] After OS filter '{os_filter}': {len(hf_dataset)} examples")
    
    # First pass: group INDICES by trajectory_id (memory efficient - no image copying)
    # Use batch column access to avoid slow individual example access
    print("[create_dataset_from_hf] Grouping by trajectory (indices only)...")
    trajectory_ids = hf_dataset["trajectory_id"]  # Get column as list (fast)
    step_numbers = hf_dataset["step_number"]  # Get column as list (fast)
    
    trajectory_indices = defaultdict(list)
    for idx, (traj_id, step_number) in enumerate(zip(trajectory_ids, step_numbers)):
        trajectory_indices[traj_id].append((step_number, idx))
    
    print(f"[create_dataset_from_hf] Found {len(trajectory_indices)} unique trajectories")
    
    # Sort indices within each trajectory by step_number
    for traj_id in trajectory_indices:
        trajectory_indices[traj_id].sort(key=lambda x: x[0])  # Sort by step_number
    
    # Build a mapping from each dataset index to its trajectory's indices
    # This allows the collator to find all steps in the same trajectory
    index_to_traj_indices = {}
    for traj_id, step_list in trajectory_indices.items():
        indices_only = [idx for _, idx in step_list]
        for step_number, idx in step_list:
            # Find position of this index in the sorted trajectory
            step_idx = indices_only.index(idx)
            index_to_traj_indices[idx] = {
                "trajectory_id": traj_id,
                "all_indices": indices_only,  # All dataset indices for this trajectory
                "current_step_idx": step_idx,
            }
    
    # Clear trajectory_indices to free memory
    del trajectory_indices
    gc.collect()
    
    # Second pass: build training examples
    # Use batch column access for efficiency - only access images lazily
    print("[create_dataset_from_hf] Building training examples...")
    
    # Get all non-image columns at once (fast batch access)
    all_task_descriptions = hf_dataset["task_description"]
    all_os_types = hf_dataset["os_type"]
    all_domains = hf_dataset["domain"]
    all_reasonings = hf_dataset["reasoning"]
    all_action_proposals = hf_dataset["action_proposal"]
    all_action_dicts = hf_dataset["action_dict"]
    all_rewards = hf_dataset["reward"]
    all_dones = hf_dataset["done"]
    # Optional per-step skip flag (if present in dataset); default False.
    all_skips = None
    if "skip" in hf_dataset.column_names:
        all_skips = hf_dataset["skip"]
    # Note: We don't fetch images here - they're loaded lazily by the collator
    
    training_examples = []
    
    for idx in tqdm(range(len(hf_dataset)), desc="Building training examples"):
        traj_info = index_to_traj_indices[idx]

        # If this step is marked skip=True, keep it in the trajectory for context
        # but do not create supervised training examples from it.
        if all_skips is not None and bool(all_skips[idx]) is True:
            continue
        
        # Parse action_dict from JSON string
        action_dict_str = all_action_dicts[idx] if all_action_dicts[idx] else "{}"
        try:
            action_dict = json.loads(action_dict_str) if isinstance(action_dict_str, str) else action_dict_str
        except json.JSONDecodeError:
            action_dict = {}
        
        # Build the current step dict WITHOUT the image
        # The image will be loaded lazily by the collator when needed
        current_step = {
            "step": step_numbers[idx],
            "is_replay": False,
            "reasoning": all_reasonings[idx] if all_reasonings[idx] else "",
            "action_proposal": all_action_proposals[idx] if all_action_proposals[idx] else "",
            "action_dict": action_dict,
            "reward": all_rewards[idx] if all_rewards[idx] else 0,
            "done": all_dones[idx] if all_dones[idx] else False,
            # Store the HF dataset index for lazy image loading
            "_hf_idx": idx,
            "skip": False,
        }
        
        base_example = {
            "task_description": all_task_descriptions[idx] if all_task_descriptions[idx] else "",
            "branch_dir": "",  # Not used for HF datasets
            "step": current_step,
            # Store indices for lazy loading of all_steps in collator
            "_hf_dataset_indices": traj_info["all_indices"],
            "_hf_current_step_idx": traj_info["current_step_idx"],
            "current_step_idx": traj_info["current_step_idx"],
            "os_type": all_os_types[idx] if all_os_types[idx] else "",
            "domain": all_domains[idx] if all_domains[idx] else "",
            "trajectory_id": traj_info["trajectory_id"],
            "total_steps": len(traj_info["all_indices"]),
        }
        
        # Type 1: predict action_proposal + action
        example_type1 = base_example.copy()
        example_type1["step"] = current_step.copy()  # Shallow copy step dict
        example_type1["training_type"] = "type1"
        training_examples.append(example_type1)
        
        # Type 2: given action_proposal, predict action only
        if dual_training_types:
            example_type2 = base_example.copy()
            example_type2["step"] = current_step.copy()  # Shallow copy step dict
            example_type2["training_type"] = "type2"
            training_examples.append(example_type2)
    
    # Clear the mapping to free memory
    del index_to_traj_indices
    gc.collect()
    
    print(f"[create_dataset_from_hf] Created {len(training_examples)} training examples")
    
    # Store reference to original HF dataset for lazy loading in collator
    result_dataset = Dataset.from_list(training_examples)
    
    # Attach the original HF dataset as an attribute for the collator to use
    result_dataset._hf_source_dataset = hf_dataset
    
    return result_dataset