#!/usr/bin/env python3
"""
Backfill high-level action proposals for each step in verified branch trajectories.

For every task folder under a branch-generated root (e.g.
`/branch_generated_verified_done`), this script:

1. Reads `trajectory.jsonl`, `metadata.json`, and `screenshots/`.
2. For each step that has both a before and after screenshot:
   - Replay steps use `step_{i-1}_replay.png` -> `step_{i}_replay.png`.
   - New branch steps use:
       * Step 1: `step_{num_replay_steps}_replay.png` -> `step_1.png`
       * Step k>1: `step_{k-1}.png` -> `step_{k}.png`
3. Sends both screenshots plus the recorded low-level action and reasoning to an
   Azure OpenAI-compatible endpoint to obtain a concise natural-language
   `action_proposal` (e.g. "Click the Chrome menu button ...").
4. Stores the proposal back into each trajectory entry as `action_proposal`.

Use `--dry-run` to print the folder name, step index, and generated proposal
without modifying any files.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


LOGGER = logging.getLogger("backfill_action_propose")


# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------


def setup_logging(verbosity: int) -> None:
    level = logging.DEBUG if verbosity > 0 else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def encode_image_base64(path: Path) -> Optional[str]:
    if not path or not path.exists():
        return None
    try:
        with path.open("rb") as fp:
            return base64.b64encode(fp.read()).decode("utf-8")
    except Exception as exc:  # pragma: no cover - best-effort helper
        LOGGER.warning("Failed to read image %s: %s", path, exc)
        return None


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def dump_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_num, raw in enumerate(fp, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} in {path}") from exc
    return rows


def dump_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort extraction of a JSON object from raw model text.
    Copied (lightly) from the replay-reasoning backfill script for consistency.
    """
    if not text:
        return None
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", candidate, flags=re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
    if not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = candidate[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def build_content_block(text: str, image_b64_list: Sequence[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """
    Build a multimodal content list: start with text, then (label, image) pairs.
    """
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for label, img_b64 in image_b64_list:
        if label:
            content.append({"type": "text", "text": label})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
    return content


def determine_task_description(metadata: Dict[str, Any]) -> str:
    for key in (
        "new_task_description",
        "generated_task_description_from_vllm",
        "original_instruction",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Follow the task instructions shown on screen."


# --------------------------------------------------------------------------------------
# Core backfill logic
# --------------------------------------------------------------------------------------


@dataclass
class BackfillStats:
    processed_tasks: int = 0
    skipped_tasks: int = 0
    updated_steps: int = 0
    skipped_existing: int = 0
    skipped_no_images: int = 0


class ActionProposalBackfiller:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root_dir = Path(args.root_dir).resolve()
        self.only_task: Optional[Path] = None
        # Allow selecting a single folder either via --only-task or --task-name.
        task_selector = (getattr(args, "only_task", "") or getattr(args, "task_name", "")).strip()
        if task_selector:
            raw = Path(task_selector).expanduser()
            self.only_task = raw if raw.is_absolute() else (self.root_dir / task_selector).resolve()

        self.api_url = args.api_url.rstrip("/")
        self.api_key = args.api_key or os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            LOGGER.warning("No API key provided (use --api-key or set AZURE_OPENAI_API_KEY / OPENAI_API_KEY)")
        self.model = args.model
        self.request_timeout = args.request_timeout
        self.max_retries = args.max_retries
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens

        self.stats = BackfillStats()

    # -------------------------- public entrypoint --------------------------

    def run(self) -> BackfillStats:
        if not self.root_dir.exists():
            LOGGER.error("Root directory does not exist: %s", self.root_dir)
            return self.stats

        if self.only_task:
            task_dirs = [self.only_task] if self.only_task.exists() else []
            if not task_dirs:
                LOGGER.error("Requested task folder not found: %s", self.only_task)
        else:
            task_dirs = sorted(p for p in self.root_dir.iterdir() if p.is_dir())

        if not task_dirs:
            LOGGER.info("No task folders found under %s", self.root_dir)
            return self.stats

        for task_dir in task_dirs:
            try:
                updated = self.process_task_dir(task_dir)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("Failed to process %s: %s", task_dir, exc)
                continue

            if updated:
                self.stats.processed_tasks += 1
            else:
                self.stats.skipped_tasks += 1

        LOGGER.info(
            "Done. tasks_processed=%s, tasks_skipped=%s, steps_updated=%s, "
            "steps_skipped_existing=%s, steps_skipped_no_images=%s",
            self.stats.processed_tasks,
            self.stats.skipped_tasks,
            self.stats.updated_steps,
            self.stats.skipped_existing,
            self.stats.skipped_no_images,
        )
        return self.stats

    # -------------------------- per-task processing --------------------------

    def process_task_dir(self, task_dir: Path) -> bool:
        traj_path = task_dir / "trajectory.jsonl"
        meta_path = task_dir / "metadata.json"
        screenshots_dir = task_dir / "screenshots"

        if not traj_path.exists() or not meta_path.exists() or not screenshots_dir.exists():
            LOGGER.debug(
                "Skipping %s (missing trajectory/metadata/screenshots)", task_dir.name
            )
            return False

        trajectory = load_jsonl(traj_path)
        metadata = load_json(meta_path)
        num_replay_steps = int(metadata.get("num_replay_steps") or 0)
        num_new_steps = int(metadata.get("num_new_steps") or max(0, len(trajectory) - num_replay_steps))

        task_desc = determine_task_description(metadata)

        updated = False
        max_index = min(len(trajectory), num_replay_steps + num_new_steps)

        for idx in range(max_index):
            entry = trajectory[idx]
            action_raw = str(entry.get("action", "")).strip()
            if not action_raw or action_raw.upper() == "DONE":
                continue

            if entry.get("action_proposal") and not self.args.force:
                self.stats.skipped_existing += 1
                continue

            if idx < num_replay_steps:
                segment = "replay"
                step_in_segment = idx + 1  # 1-based replay step
                before_path = screenshots_dir / f"step_{step_in_segment - 1}_replay.png"
                after_path = screenshots_dir / f"step_{step_in_segment}_replay.png"
            else:
                segment = "new"
                step_in_segment = idx - num_replay_steps + 1  # 1-based new step index
                if step_in_segment <= 0:
                    continue
                if step_in_segment == 1:
                    before_path = screenshots_dir / f"step_{num_replay_steps}_replay.png"
                    after_path = screenshots_dir / "step_1.png"
                else:
                    before_path = screenshots_dir / f"step_{step_in_segment - 1}.png"
                    after_path = screenshots_dir / f"step_{step_in_segment}.png"

            before_b64 = encode_image_base64(before_path)
            after_b64 = encode_image_base64(after_path)
            if not before_b64 or not after_b64:
                LOGGER.debug(
                    "Skipping %s step=%s (%s) due to missing screenshots: before=%s after=%s",
                    task_dir.name,
                    step_in_segment,
                    segment,
                    before_path if before_b64 else "MISSING",
                    after_path if after_b64 else "MISSING",
                )
                self.stats.skipped_no_images += 1
                continue

            reasoning = str(entry.get("reasoning", "")).strip()

            # Special case: for the first *non-replay* (new) step, hard-code an empty proposal
            # instead of calling the LLM. This is the actual first run step, not a replayed one.
            if segment == "new" and step_in_segment == 1:
                proposal = ""
            else:
                proposal = self._generate_action_proposal(
                    task_name=task_dir.name,
                    task_description=task_desc,
                    segment=segment,
                    step_in_segment=step_in_segment,
                    action_raw=action_raw,
                    reasoning=reasoning,
                    before_b64=before_b64,
                    after_b64=after_b64,
                )

            # Treat `None` as failure, but allow empty string as a valid proposal value.
            if proposal is None:
                continue

            if self.args.dry_run:
                print(
                    f"[DRY-RUN] {task_dir.name} | segment={segment} step={step_in_segment} "
                    f"-> action_proposal={proposal}"
                )
            else:
                entry["action_proposal"] = proposal
                updated = True
                self.stats.updated_steps += 1

        if updated and not self.args.dry_run:
            dump_jsonl(traj_path, trajectory)
            LOGGER.info("Updated action proposals for %s", task_dir.name)

        return updated

    # -------------------------- LLM helpers --------------------------

    def _complete(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """
        Call Azure/OpenAI-style /chat/completions endpoint and return the text content.
        """
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        body: Dict[str, Any] = {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # Some Azure deployments require a model name even when using a deployment-specific URL.
        if self.model:
            body["model"] = self.model

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    self.api_url,
                    headers=headers,
                    json=body,
                    timeout=self.request_timeout,
                )
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    return None
                msg = choices[0].get("message") or {}
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
                    result = "".join(text_parts).strip()
                else:
                    result = str(content).strip()
                return result or None
            except Exception as exc:  # pragma: no cover - network error handling
                last_error = exc
                LOGGER.warning(
                    "LLM call failed for %s (attempt %s/%s): %s",
                    self.api_url,
                    attempt,
                    self.max_retries,
                    exc,
                )
                time.sleep(min(2 ** attempt, 8))

        LOGGER.error("Giving up after %s attempts due to error: %s", self.max_retries, last_error)
        return None

    def _generate_action_proposal(
        self,
        *,
        task_name: str,
        task_description: str,
        segment: str,
        step_in_segment: int,
        action_raw: str,
        reasoning: str,
        before_b64: str,
        after_b64: str,
    ) -> Optional[str]:
        """Ask the model to summarize the step as a single concrete action proposal."""
        seg_label = "replay" if segment == "replay" else "branch"
        text_prompt = (
            f"Task description:\n{task_description}\n\n"
            f"You are generating a concise high-level action proposal for a GUI agent.\n"
            f"This is step {step_in_segment}.\n\n"
            "Inputs you have:\n"
            "- Screenshot before the step\n"
            "- Screenshot after the step\n"
            "- The low-level action script for this step\n"
            "- The original reasoning text for this step\n\n"
            "Using ALL of this information, write ONE concrete, natural-language UI command that\n"
            "describes what the user does in this step (for example:\n"
            "\"Type the text exactly: Favorites\" or\n"
            "\"Click the Chrome menu button (three vertical dots) at the top-right corner of the browser toolbar to open the main menu.\").\n\n"
            "Rules:\n"
            "- Describe exactly one atomic action, not a multi-step plan.\n"
            "- Use imperative mood (start with a verb like 'Click', 'Type', 'Scroll', 'Select').\n"
            "- Be precise about UI targets (e.g., specify button labels, icons, menu names).\n"
            "- Do NOT include numbering, bullet points, or additional explanation.\n"
            "Respond with JSON only, in the exact format:\n"
            "{\"action_proposal\": \"...\"}"
        )

        reasoning_snip = reasoning[:1000] if reasoning else ""
        action_snip = action_raw[:1000]
        text_prompt += (
            f"\n\nLow-level action script for this step:\n{action_snip}\n\n"
            f"Original reasoning text for this step (may be empty):\n{reasoning_snip}\n"
        )

        image_blocks: List[Tuple[str, str]] = [
            ("Screenshot before the step.", before_b64),
            ("Screenshot after the step.", after_b64),
        ]
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You write precise, single-step GUI action proposals.",
                    }
                ],
            },
            {
                "role": "user",
                "content": build_content_block(text_prompt, image_blocks),
            },
        ]

        raw = self._complete(messages)
        if not raw:
            LOGGER.warning(
                "Empty response for %s step=%s (%s)", task_name, step_in_segment, segment
            )
            return None

        data = extract_json_from_response(raw)
        proposal: Optional[str] = None
        if isinstance(data, dict):
            val = data.get("action_proposal")
            if isinstance(val, str):
                proposal = val.strip()

        if not proposal:
            # Fallback: treat raw text as the proposal.
            proposal = raw.strip()

        if not proposal:
            LOGGER.warning(
                "Failed to extract proposal for %s step=%s (%s); raw=%r",
                task_name,
                step_in_segment,
                segment,
                raw[:200],
            )
            return None

        return proposal


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill high-level action_proposal strings for branch trajectories."
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="/branch_generated_verified_done",
        help="Root directory containing verified branch-generated task subdirectories.",
    )
    parser.add_argument(
        "--only-task",
        type=str,
        default="",
        help="Optional single task folder to process (name under root-dir or absolute path).",
    )
    parser.add_argument(
        "--task-name",
        "-t",
        type=str,
        default="",
        help="Alias for --only-task; folder name under root-dir or absolute path.",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        help="Azure/OpenAI chat completions endpoint URL.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("AZURE_OPENAI_API_KEY"),
        help="API key for the Azure/OpenAI endpoint (or set AZURE_OPENAI_API_KEY / OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1-chat",
        help="Model name associated with the Azure deployment (if required).",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per LLM call.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for action proposal generation.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens for each action proposal completion.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate proposals even if an entry already has action_proposal.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print folder / step / proposal but do not modify trajectory files.",
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        action="count",
        default=0,
        help="Increase log verbosity (use -v for debug).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging(args.verbosity)
    LOGGER.info("Starting action_proposal backfill with args: %s", args)
    backfiller = ActionProposalBackfiller(args)
    backfiller.run()


if __name__ == "__main__":
    main(sys.argv[1:])


