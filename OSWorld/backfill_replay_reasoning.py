#!/usr/bin/env python3
"""
Backfill missing reasoning for replay (pre-branch) steps in branch-generated trajectories.

Workflow:
1. Filter source task directories into a scratch sub-folder, keeping only tasks whose
   verification succeeded and whose final action ended with DONE.
2. For every replay step that lacks reasoning:
   a. Ask the model (with higher temperature) to propose several plausible next actions
      for the current state.
   b. Compare each candidate against the recorded action by providing the action script
      along with the before/after screenshots, and reuse the first matching candidate's
      reasoning as the replay reasoning.
   c. If no candidate matches, fall back to asking the model—conditioned on the
      before/after screenshots—to directly verbalize the reasoning.
3. Persist the updated trajectories (and optionally metadata annotations) inside the
   filtered scratch folder.

This script uses an OpenAI-compatible endpoint (e.g., a vLLM server hosting Qwen)
for all multimodal generations so it can work with either OpenAI's client or local
deployments.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI  # type: ignore[import]


LOGGER = logging.getLogger("backfill_replay_reasoning")


# --------------------------------------------------------------------------------------
# Utility functions
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
    except Exception as exc:
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def last_action_is_done(trajectory: Sequence[Dict[str, Any]]) -> bool:
    if not trajectory:
        return False
    last = trajectory[-1]
    return str(last.get("action", "")).strip().upper() == "DONE"


def extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a JSON object from raw model text."""
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


def format_history(entries: Sequence[Dict[str, Any]]) -> str:
    """Create a readable summary of prior steps for prompting."""
    if not entries:
        return "No previous actions have been taken."
    lines: List[str] = []
    for idx, entry in enumerate(entries, start=1):
        step_label = entry.get("step")
        label = step_label if isinstance(step_label, int) else idx
        action_text = str(entry.get("action", "")).strip()
        reasoning_text = str(entry.get("reasoning", "")).strip()
        if reasoning_text:
            lines.append(f"Step {label}: action -> {action_text}\nReasoning: {reasoning_text}")
        else:
            lines.append(f"Step {label}: action -> {action_text}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# LLM helpers and JSON schemas
# --------------------------------------------------------------------------------------


CANDIDATE_LIST_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "candidate_reasoning_response",
        "schema": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action_summary": {"type": "string"},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["action_summary", "reasoning"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["candidates"],
            "additionalProperties": False,
        },
    },
}

FALLBACK_REASONING_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "fallback_reasoning_response",
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
            },
            "required": ["reasoning"],
            "additionalProperties": False,
        },
    },
}

MATCH_RESULT_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "action_match_response",
        "schema": {
            "type": "object",
            "properties": {
                "match": {"type": "boolean"},
                "explanation": {"type": "string"},
            },
            "required": ["match"],
            "additionalProperties": False,
        },
    },
}


def build_content_block(text: str, image_b64_list: Sequence[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """
    Build a multimodal content list: start with text, then (label, image) pairs.

    Args:
        text: Primary text context.
        image_b64_list: Sequence of (label, b64 string) pairs.
    """
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for label, img_b64 in image_b64_list:
        if label:
            content.append({"type": "text", "text": label})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
    return content


# --------------------------------------------------------------------------------------
# Core backfill pipeline
# --------------------------------------------------------------------------------------


@dataclass
class BackfillStats:
    copied_tasks: int = 0
    processed_tasks: int = 0
    updated_steps: int = 0
    skipped_tasks: int = 0


class ReplayReasoningBackfiller:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.source_dir = Path(args.source_dir).resolve()
        self.filtered_dir = Path(args.filtered_dir).resolve()
        ensure_dir(self.filtered_dir)
        raw_only_task = args.only_task.strip() if args.only_task else ""
        self.only_task_name = ""
        self.only_task_dir: Optional[Path] = None
        if raw_only_task:
            raw_only_path = Path(raw_only_task).expanduser()
            if raw_only_path.is_absolute():
                self.only_task_dir = raw_only_path.resolve()
                self.only_task_name = self.only_task_dir.name
            else:
                self.only_task_name = raw_only_task
                self.only_task_dir = (self.filtered_dir / raw_only_task).resolve()
        if args.detail_log_path:
            detail_log = Path(args.detail_log_path).expanduser().resolve()
        else:
            detail_log = self.filtered_dir / "backfill_reasoning_details.log"
        ensure_dir(detail_log.parent)
        self.detail_log_path = detail_log
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.client = OpenAI(base_url=args.base_url, api_key=api_key)
        self.request_timeout = args.request_timeout
        self.max_retries = args.max_retries
        self.max_workers = max(1, int(getattr(args, "max_workers", 1)))
        self._stats_lock = threading.Lock()
        self._detail_lock = threading.Lock()
        self.stats = BackfillStats()

    # -------------------------- filtering --------------------------

    def filter_tasks(self) -> List[Path]:
        LOGGER.info("Filtering tasks from %s into %s", self.source_dir, self.filtered_dir)
        kept: List[Path] = []
        for task_dir in sorted(self.source_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            if self.only_task_name and task_dir.name != self.only_task_name:
                continue
            metadata_path = task_dir / "metadata.json"
            traj_path = task_dir / "trajectory.jsonl"
            if not metadata_path.exists() or not traj_path.exists():
                continue
            try:
                metadata = load_json(metadata_path)
                trajectory = load_jsonl(traj_path)
            except Exception as exc:
                LOGGER.warning("Skipping %s (failed to read): %s", task_dir.name, exc)
                continue

            verification = metadata.get("verification") or {}
            if not isinstance(verification, dict) or not verification.get("success"):
                continue
            if not last_action_is_done(trajectory):
                continue

            dest_dir = self.filtered_dir / task_dir.name
            kept.append(dest_dir)
            if dest_dir.exists():
                if self.args.overwrite_filtered:
                    shutil.rmtree(dest_dir)
                else:
                    LOGGER.debug("Filtered dir already exists, skipping copy: %s", dest_dir)
                    continue
            shutil.copytree(task_dir, dest_dir)
            with self._stats_lock:
                self.stats.copied_tasks += 1
        LOGGER.info("Copied %s tasks; %s already present", self.stats.copied_tasks, len(kept) - self.stats.copied_tasks)
        return kept

    # -------------------------- processing --------------------------

    def run(self) -> BackfillStats:
        filtered_dirs: List[Path] = []
        if not getattr(self.args, "use_existing_filtered", False):
            filtered_dirs = self.filter_tasks()
        else:
            LOGGER.info(
                "Skipping filtering/copy from source_dir; using existing contents of %s",
                self.filtered_dir,
            )
        if self.only_task_name:
            target_dir = self.only_task_dir or (self.filtered_dir / self.only_task_name)
            if not target_dir.exists():
                LOGGER.error("Requested task folder %s not found (expected path: %s)", self.only_task_name, target_dir)
                return self.stats
            target_dirs: List[Path] = [target_dir]
        else:
            target_dirs = filtered_dirs if filtered_dirs else sorted(self.filtered_dir.iterdir())
        if not target_dirs:
            LOGGER.info("No task folders found under %s", self.filtered_dir)
            return self.stats
        if self.max_workers <= 1 or len(target_dirs) == 1:
            for task_dir in target_dirs:
                if not task_dir.is_dir():
                    continue
                self._process_and_record(task_dir)
        else:
            LOGGER.info(
                "Processing %s task folders with up to %s concurrent workers",
                len(target_dirs),
                self.max_workers,
            )
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self._process_and_record, task_dir)
                    for task_dir in target_dirs
                    if task_dir.is_dir()
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        LOGGER.exception("Worker thread raised an exception: %s", exc)
        LOGGER.info("Processing finished: %s tasks updated, %s skipped, %s steps filled",
                    self.stats.processed_tasks, self.stats.skipped_tasks, self.stats.updated_steps)
        return self.stats

    def _process_and_record(self, task_dir: Path) -> None:
        try:
            updated = self.process_task_dir(task_dir)
        except Exception as exc:
            LOGGER.exception("Failed to process %s: %s", task_dir.name, exc)
            return
        if updated:
            with self._stats_lock:
                self.stats.processed_tasks += 1
            LOGGER.info("Finished task folder: %s", task_dir.name)
        else:
            with self._stats_lock:
                self.stats.skipped_tasks += 1

    def process_task_dir(self, task_dir: Path) -> bool:
        metadata_path = task_dir / "metadata.json"
        traj_path = task_dir / "trajectory.jsonl"
        screenshots_dir = task_dir / "screenshots"

        metadata = load_json(metadata_path)
        trajectory = load_jsonl(traj_path)
        num_replay_steps = int(metadata.get("num_replay_steps") or 0)
        if num_replay_steps <= 0:
            LOGGER.debug("No replay steps in %s", task_dir.name)
            return False
        if len(trajectory) < num_replay_steps:
            LOGGER.warning("Task %s has fewer trajectory entries than num_replay_steps (%s < %s)",
                           task_dir.name, len(trajectory), num_replay_steps)
            num_replay_steps = len(trajectory)

        task_description = self._determine_task_description(metadata)

        updated = False
        # Load existing updated_steps if present to preserve them
        existing_backfill = metadata.get("replay_reasoning_backfill", {})
        existing_updated_steps = existing_backfill.get("updated_steps", {}) if isinstance(existing_backfill, dict) else {}
        
        replay_backfill_meta: Dict[str, Any] = {
            "updated_steps": dict(existing_updated_steps),  # Start with existing steps
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        for idx in range(num_replay_steps):
            entry = trajectory[idx]
            if entry.get("reasoning") and not self.args.force:
                continue
            action_text = str(entry.get("action", "")).strip()
            step_index = idx + 1
            before_path = screenshots_dir / f"step_{step_index - 1}_replay.png"
            after_path = screenshots_dir / f"step_{step_index}_replay.png"
            before_b64 = encode_image_base64(before_path)
            after_b64 = encode_image_base64(after_path)
            history_summary = format_history(trajectory[:idx])
            history_images = self._gather_history_images(step_index, screenshots_dir)

            reasoning, method = self._infer_reasoning(
                task_description=task_description,
                action_text=action_text,
                after_b64=after_b64,
                before_b64=before_b64,
                history_summary=history_summary,
                history_images=history_images,
                task_name=task_dir.name,
                step_index=step_index,
            )

            if not reasoning:
                LOGGER.warning("No reasoning generated for %s step %s", task_dir.name, step_index)
                continue

            entry["reasoning"] = reasoning
            step_backfill_meta = {
                "method": method,
                "action_text": action_text,
            }
            replay_backfill_meta["updated_steps"][str(step_index)] = step_backfill_meta
            updated = True
            with self._stats_lock:
                self.stats.updated_steps += 1
            self._log_detail(
                task_dir.name,
                f"Step {step_index}: reasoning stored using method '{method}'",
            )

            if self.args.sleep_between_steps > 0:
                time.sleep(self.args.sleep_between_steps)

        if updated and not self.args.dry_run:
            dump_jsonl(traj_path, trajectory)
            metadata.setdefault("replay_reasoning_backfill", {}).update(replay_backfill_meta)
            dump_json(metadata_path, metadata)
        return updated

    # -------------------------- step helpers --------------------------

    def _complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                extra_args: Dict[str, Any] = {}
                if response_format:
                    extra_args["response_format"] = response_format
                resp = self.client.chat.completions.create(
                    model=self.args.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=self.request_timeout,
                    **extra_args,
                )
                return resp.choices[0].message.content if resp.choices else None
            except Exception as exc:
                last_error = exc
                LOGGER.warning("LLM call failed (attempt %s/%s): %s", attempt, self.max_retries, exc)
                time.sleep(min(2 ** attempt, 8))
        LOGGER.error("Giving up after %s attempts: %s", self.max_retries, last_error)
        return None

    def _log_detail(self, task_name: str, message: str) -> None:
        line = f"[{task_name}] {message}"
        LOGGER.info(line)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        try:
            with self._detail_lock:
                with self.detail_log_path.open("a", encoding="utf-8") as fp:
                    fp.write(f"{timestamp} {line}\n")
        except Exception as exc:
            LOGGER.warning("Failed to write detail log (%s): %s", self.detail_log_path, exc)

    def _determine_task_description(self, metadata: Dict[str, Any]) -> str:
        for key in (
            "generated_task_description_from_vllm",
        ):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return "Follow the task instructions shown on screen."

    def _infer_reasoning(
        self,
        task_description: str,
        action_text: str,
        after_b64: Optional[str],
        before_b64: Optional[str],
        history_summary: str,
        history_images: Sequence[Tuple[str, str]],
        task_name: str,
        step_index: int,
    ) -> Tuple[Optional[str], str]:
        candidates = self._sample_next_actions(
            task_description=task_description,
            history_summary=history_summary,
            history_images=history_images,
            current_state_b64=after_b64,
            task_name=task_name,
            step_index=step_index,
        )
        best_reasoning: Optional[str] = None
        method = "candidate_match"
        for idx, candidate in enumerate(candidates, start=1):
            summary = candidate.get("action_summary", "")
            reasoning = candidate.get("reasoning", "")
            matched = self._actions_match_via_llm(
                action_text=action_text,
                before_b64=before_b64,
                after_b64=after_b64,
                candidate_action=summary,
                candidate_index=idx,
                task_name=task_name,
                step_index=step_index,
            )
            if matched:
                best_reasoning = reasoning.strip()
                self._log_detail(
                    task_name,
                    f"Step {step_index}: selected sampled action #{idx} -> '{summary}'",
                )
                break
        if best_reasoning:
            return best_reasoning, method

        fallback_reasoning = self._fallback_reasoning(task_description, action_text, before_b64, after_b64)
        if fallback_reasoning:
            self._log_detail(task_name, f"Step {step_index}: using fallback reasoning")
        return fallback_reasoning, "fallback_direct" if fallback_reasoning else method

    def _sample_next_actions(
        self,
        task_description: str,
        history_summary: str,
        history_images: Sequence[Tuple[str, str]],
        current_state_b64: Optional[str],
        task_name: str,
        step_index: int,
    ) -> List[Dict[str, str]]:
        if not current_state_b64:
            self._log_detail(task_name, f"Step {step_index}: missing current-state screenshot; skipping candidate sampling")
            return []
        text_prompt = (
            f"Task description:\n{task_description}\n\n"
            "History of completed replay steps:\n"
            f"{history_summary}\n\n"
            f"Based on the COMPLETE trajectory above and the UI STATE change to the current state (screenshots provided), list {self.args.num_candidates} distinct, "
            "plausible next-step actions that would progress the task. "
            "Respond in JSON as {\"candidates\": [{\"action_summary\": \"...\", \"reasoning\": \"...\"}, ...]}.\n"
            "Each action_summary must be a concrete UI command (e.g., 'Click the Privacy and security tab')."
            "The action_summary must be a single action, like copy ..., or click ..., or select ..., or type ..., or etc. It cannot be a compound action like 'Copy the email address and then click the 'Send' button'."
            "Each reasoning must be a 1-2 sentence explanation of what the user is thinking and the action to take. "
            "The reasoning should be a natural-language explanation of what have been seen/done, follow by the next action to take concretely. You should write in first-person perspective (e.g. I, we, etc.)."
            "Reasoning Example: 'The current screen shows the settings categories but not the detailed options. Selecting the 'Privacy and security' tab to reveal the controls the task needs.'"
            "Reasoning Example: 'I'm now in the \"Font Effects\" tab. I can see at the bottom right there's a \"Shadow\" checkbox in the \"Effects\" section. Let me check that box to enable the shadow effect on the LAUNCH text.'"
            "Reasoning Example: 'I can see the context menu. I'll click on \"Format Cells...\" to open the formatting dialog:'"
        )
        image_blocks = list(history_images)
        image_blocks.append(("Current state screenshot.", current_state_b64))
        messages = [
            {"role": "system", "content": "Generate diverse next-step hypotheses for a GUI agent."},
            {"role": "user", "content": build_content_block(text_prompt, image_blocks)},
        ]
        response = self._complete(
            messages=messages,
            max_tokens=2048,
            temperature=self.args.candidate_temperature,
            top_p=0.95,
            response_format=CANDIDATE_LIST_RESPONSE_FORMAT,
        )
        print(response)
        print("--------------------------------")
        data = extract_json_from_response(response or "")
        candidates = []
        if isinstance(data, dict):
            raw_candidates = data.get("candidates")
            if isinstance(raw_candidates, list):
                for item in raw_candidates:
                    if isinstance(item, dict):
                        summary = item.get("action_summary")
                        reasoning = item.get("reasoning")
                        if isinstance(summary, str) and isinstance(reasoning, str):
                            candidates.append(
                                {
                                    "action_summary": summary.strip(),
                                    "reasoning": reasoning.strip(),
                                }
                            )
        if candidates:
            for idx, candidate in enumerate(candidates, start=1):
                self._log_detail(
                    task_name,
                    f"Step {step_index}: sampled action #{idx} -> action='{candidate['action_summary']}' | reasoning='{candidate['reasoning']}'",
                )
        else:
            self._log_detail(task_name, f"Step {step_index}: no candidates returned by model")
        return candidates

    def _fallback_reasoning(
        self,
        task_description: str,
        action_text: str,
        before_b64: Optional[str],
        after_b64: Optional[str],
    ) -> Optional[str]:
        if not after_b64:
            return None
        text_prompt = (
            f"Task description:\n{task_description}\n\n"
            "Using the before/after screenshots, write in 1 - 2 sentences to mimic the thinking process about why the user is taking the action and why it take the action, you should write in first-person perspective (e.g. The current screen shows the settings categories but not the detailed options. Selecting the 'Privacy and security' tab to reveal the controls the task needs.)"
            f"Action script (for reference):\n{action_text}\n"
            "Respond with JSON: {\"reasoning\": \"...\"} focusing on intent, not low-level code."
            "Each reasoning must be a 1-2 sentence explanation of what the user is thinking and the action to take. "
            "The reasoning should be a natural-language explanation of what have been seen/done, follow by the next action to take concretely. You should write in first-person perspective (e.g. I, we, etc.)."
            "Reasoning Example: 'The current screen shows the settings categories but not the detailed options. Selecting the 'Privacy and security' tab to reveal the controls the task needs.'"
            "Reasoning Example: 'I'm now in the \"Font Effects\" tab. I can see at the bottom right there's a \"Shadow\" checkbox in the \"Effects\" section. Let me check that box to enable the shadow effect on the LAUNCH text.'"
            "Reasoning Example: 'I can see the context menu. I'll click on \"Format Cells...\" to open the formatting dialog:'"
        )
        image_blocks: List[Tuple[str, str]] = []
        if before_b64:
            image_blocks.append(("State before the action.", before_b64))
        if after_b64:
            image_blocks.append(("State after the action.", after_b64))
        messages = [
            {"role": "system", "content": "Explain why a GUI change is helpful for the task at hand."},
            {"role": "user", "content": build_content_block(text_prompt, image_blocks)},
        ]
        response = self._complete(
            messages=messages,
            max_tokens=256,
            temperature=self.args.fallback_temperature,
            top_p=0.9,
            response_format=FALLBACK_REASONING_RESPONSE_FORMAT,
        )
        data = extract_json_from_response(response or "")
        reasoning = (data or {}).get("reasoning")
        if isinstance(reasoning, str):
            return reasoning.strip()
        return None

    def _gather_history_images(self, step_index: int, screenshots_dir: Path) -> List[Tuple[str, str]]:
        """
        Collect replay screenshots up to (and including) the state immediately before the current step.
        step_index is 1-based for the step being processed.
        """
        images: List[Tuple[str, str]] = []
        for hist_step in range(step_index):
            path = screenshots_dir / f"step_{hist_step}_replay.png"
            label = "Initial replay state" if hist_step == 0 else f"State after replay step {hist_step}"
            b64 = encode_image_base64(path)
            if b64:
                images.append((label, b64))
        return images

    def _actions_match_via_llm(
        self,
        action_text: str,
        before_b64: Optional[str],
        after_b64: Optional[str],
        candidate_action: str,
        *,
        candidate_index: int,
        task_name: str,
        step_index: int,
    ) -> bool:
        if not candidate_action or not after_b64:
            return False
        text_prompt = (
            "Determine whether the proposed candidate action matches the recorded replay action. "
            "Use the action script plus the visual change between the before/after screenshots. "
            "Respond ONLY with JSON: {\"match\": true|false, \"explanation\": \"...\"}. "
            "Only answer true if the candidate accurately describes the change you observe."
        )
        image_blocks: List[Tuple[str, str]] = []
        if before_b64:
            image_blocks.append(("Screenshot before the action.", before_b64))
        image_blocks.append(("Screenshot after the action.", after_b64))
        text_prompt += (
            f"\n\nActual replay action script:\n{action_text or 'N/A'}\n\n"
            f"Proposed candidate action:\n{candidate_action}"
        )
        messages = [
            {"role": "system", "content": "Judge GUI actions by comparing script intent with observed screenshot changes."},
            {"role": "user", "content": build_content_block(text_prompt, image_blocks)},
        ]
        response = self._complete(
            messages=messages,
            max_tokens=200,
            temperature=0.0,
            top_p=0.8,
            response_format=MATCH_RESULT_RESPONSE_FORMAT,
        )
        data = extract_json_from_response(response or "") or {}
        match_val = data.get("match")
        explanation_val = data.get("explanation", "")
        if isinstance(match_val, bool):
            self._log_detail(
                task_name,
                f"Step {step_index}: candidate #{candidate_index} match={match_val} explanation='{explanation_val}'",
            )
            return match_val
        self._log_detail(
            task_name,
            f"Step {step_index}: candidate #{candidate_index} match undetermined (response missing boolean)",
        )
        return False


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill replay reasoning for branch-generated tasks.")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="/branch_gen_winarena_filtered_done_or_verified",
        help="Directory with original branch-generated tasks.",
    )
    parser.add_argument(
        "--filtered-dir",
        type=str,
        default="/branch_gen_winarena_filtered_done_or_verified",
        help="Output directory for filtered tasks (will be created if missing).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible base URL serving the Qwen model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-32B-Instruct",
        help="Model identifier exposed by the OpenAI-compatible server.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key if required by the server (optional for most self-hosted setups).",
    )
    parser.add_argument("--request-timeout", type=int, default=180, help="Per-request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries per Qwen call.")
    parser.add_argument("--num-candidates", type=int, default=10, help="Number of candidate next steps to request.")
    parser.add_argument("--summary-temperature", type=float, default=0.0, help="Temperature for action summarization.")
    parser.add_argument("--candidate-temperature", type=float, default=0.9, help="Temperature for candidate sampling.")
    parser.add_argument("--fallback-temperature", type=float, default=0.2, help="Temperature for fallback reasoning.")
    parser.add_argument("--overwrite-filtered", action="store_true", help="Re-copy filtered tasks even if they already exist.")
    parser.add_argument(
        "--use-existing-filtered",
        action="store_true",
        help="Do not copy/filter from source_dir; instead, process trajectories already present in filtered_dir.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate reasoning even if a replay step already has one.")
    parser.add_argument("--dry-run", action="store_true", help="Do not overwrite files; just log what would change.")
    parser.add_argument("--verbosity", "-v", action="count", default=0, help="Increase log verbosity.")
    parser.add_argument("--sleep-between-steps", type=float, default=0.0, help="Optional pause (seconds) between replay steps.")
    parser.add_argument(
        "--detail-log-path",
        type=str,
        default="",
        help="Path to a file where per-step sampling details will be appended. Defaults to filtered_dir/backfill_reasoning_details.log",
    )
    parser.add_argument(
        "--only-task",
        type=str,
        default="",
        help="Process only the specified task folder (name relative to filtered_dir or absolute path).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=13,
        help="Number of branch folders to process concurrently.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging(args.verbosity)
    LOGGER.info("Starting replay reasoning backfill with args: %s", args)
    backfiller = ReplayReasoningBackfiller(args)
    stats = backfiller.run()
    LOGGER.info("Done. %s", stats)


if __name__ == "__main__":
    main(sys.argv[1:])

