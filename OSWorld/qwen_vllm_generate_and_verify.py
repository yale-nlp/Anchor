#!/usr/bin/env python3
import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    # OpenAI-compatible client for vLLM
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("The 'openai' package is required. Install via: pip install openai>=1.0.0") from e


def encode_image_base64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_trajectory(trajectory_path: Path) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    with trajectory_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                steps.append(json.loads(line))
            except Exception:
                # Skip malformed lines
                continue
    return steps


# def build_action_summary(trajectory: List[Dict[str, Any]]) -> str:
#     parts: List[str] = []
#     for idx, step in enumerate(trajectory, start=1):
#         action_text = str(step.get("action", "")).strip()
#         reasoning_text = str(step.get("reasoning", "")).strip()
#         if len(action_text) > 1200:
#             action_text = action_text[:1200] + "..."
#         if len(reasoning_text) > 1200:
#             reasoning_text = reasoning_text[:1200] + "..."
#         if reasoning_text:
#             parts.append(f"Step {idx}:\nAction:\n{action_text}\nReasoning:\n{reasoning_text}")
#         else:
#             parts.append(f"Step {idx}:\nAction:\n{action_text}")
#     return "\n\n".join(parts)

def build_action_summary(trajectory: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for idx, step in enumerate(trajectory, start=1):
        action_text = str(step.get("action", "")).strip()
        if len(action_text) > 1200:
            action_text = action_text[:1200] + "..."
        parts.append(f"Step {idx}:\nAction:\n{action_text}")
    return "\n\n".join(parts)

def collect_screenshots(screens_dir: Path, max_images: int) -> List[Path]:
    if not screens_dir.exists():
        return []
    candidates = [p for p in screens_dir.glob("*.png") if p.is_file()]
    # Order rule:
    # 1) Files ending with "_replay.png" first
    # 2) Within each group, sort by the numeric value after "step_"
    def sort_key(p: Path) -> Tuple[int, int, str]:
        name = p.name
        is_replay = 1 if not name.endswith("_replay.png") else 0  # replay first -> smaller key
        m = re.search(r"step_(\d+)(?:_replay)?\.png$", name)
        step_num = int(m.group(1)) if m else 10**9
        return (is_replay, step_num, name)
    images = sorted(candidates, key=sort_key)
    if max_images > 0 and len(images) > max_images:
        # Downsample uniformly to at most max_images
        stride = max(1, len(images) // max_images)
        images = images[::stride][:max_images]
    return images

def collect_last_screenshots(screens_dir: Path, max_images: int) -> List[Path]:
    if not screens_dir.exists():
        return []
    candidates = [p for p in screens_dir.glob("*.png") if p.is_file()]
    # Sort by step number ascending; if both replay and non-replay exist for the same step,
    # treat replay as earlier than non-replay so the "last" reflects the most recent state.
    def sort_key(p: Path) -> Tuple[int, int, str]:
        name = p.name
        m = re.search(r"step_(\d+)(?:_replay)?\.png$", name)
        step_num = int(m.group(1)) if m else 10**9
        is_replay = 0 if name.endswith("_replay.png") else 1  # non-replay should come later within same step
        return (step_num, is_replay, name)
    images = sorted(candidates, key=sort_key)
    if max_images > 0 and len(images) > max_images:
        images = images[-max_images:]
    return images


def create_client(base_url: str, api_key: Optional[str]) -> OpenAI:
    # vLLM typically ignores API key; provide a placeholder if not set
    return OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"))


def call_chat_completion(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def generate_unified_task_description(
    client: OpenAI,
    model: str,
    app_name: str,
    trajectory: List[Dict[str, Any]],
    screenshots: List[Path],
    metadata: Dict[str, Any],
) -> str:
    action_summary = build_action_summary(trajectory)
    provided_task_descriptions = metadata.get("new_task_description", "")
    # System + User messages with multimodal content
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                '''You are a task generation expert. Given a list of actions and screenshots, 
                produce a single concise task description that would be accomplished by performing 
                these actions in order on the computer.
                1. You should propose tasks that are clear and specific, it should not be too general.
                2. The task description should provide all the necessary information to complete the task.
                3. The task should be feasible to complete by a real user and should not require any additional
                information that is not specified in this input.
                4. The task description should not be foucs on the details of the actions, but should be a overall task description.
                Example: "Increase the brightness and contrast of the image on Slide 2 to make its details more visible."
                    "Calculate and display the maximum Revenue value in a new cell below the data."
                    "Set the Word Wrap Column value to 120 characters to allow longer lines."
                    "Configure Google Chrome to open a specific website (for example, your favorite news site) as the homepage and startup page whenever the browser is launched."
                '''
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        '''
                        Given a list of actions performed on the computer app {app_name} and the corresponding screenshots in order.
                        List of actions:
                        {action_summary}
                        Here is the provided task description:
                        {provided_task_descriptions}
                        Your task: It the task description match the action and the progression of the screenshot, you can return it.
                        Sometimes the descriptions may not match the entire action trajectory but part of it. 
                        In this case, you should come up with a single task description that will be accomplished by performing 
                        these actions in the given sequence on the computer. Respond with ONLY the task description text, don't include any other text.
                        '''
                    ),
                },
            ],
        },
    ]

    # Attach screenshots as images to the user message
    for img_path in screenshots:
        try:
            b64 = encode_image_base64(img_path)
            messages[1]["content"].append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            )
        except Exception:
            # Skip unreadable image
            continue

    response_text = call_chat_completion(client, model, messages, max_tokens=512, temperature=0.2)
    return response_text.strip()


def parse_verification_result(text: str) -> Tuple[Optional[bool], Optional[str]]:
    # Prefer JSON if the model returns it; fall back to pattern-based parse
    text_stripped = text.strip()

    # If response is inside code fences (``` or ```json), extract inner content
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text_stripped, flags=re.IGNORECASE)
    candidate = fenced.group(1).strip() if fenced else text_stripped

    # If still not pure JSON, try to slice between first { and last }
    if not candidate.lstrip().startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = candidate[start : end + 1]

    # Try JSON block
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            raw_success = parsed.get("success")
            explanation = parsed.get("explanation")
            if isinstance(raw_success, bool) and isinstance(explanation, str):
                return raw_success, explanation.strip()
    except Exception:
        pass

    # Try pattern matches: SUCCESS: true/false, EXPLANATION: ...
    success_match = re.search(r"SUCCESS:\s*(true|false)", text_stripped, flags=re.IGNORECASE)
    explanation_match = re.search(r"EXPLANATION:\s*(.+)", text_stripped, flags=re.DOTALL | re.IGNORECASE)
    if success_match:
        success_val = success_match.group(1).lower() == "true"
        explanation_val = explanation_match.group(1).strip() if explanation_match else ""
        return success_val, explanation_val or None

    return None, None


def verify_task_success(
    client: OpenAI,
    model: str,
    task_description: str,
    trajectory: List[Dict[str, Any]],
    screenshots: List[Path],
) -> Tuple[bool, str]:
    action_summary = build_action_summary(trajectory)
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a precise evaluator. Given a task description, the full action trajectory, and screenshots, "
                "determine whether the task was successfully completed."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "# Task Description\n"
                        f"{task_description}\n\n"
                        "# Action Trajectory\n"
                        f"{action_summary}\n\n"
                        "# Instruction\n"
                        "Assess whether the actions and final states shown in the screenshots complete the task.\n"
                        "Respond in strict JSON ONLY (no code fences, no extra text) with the following structure:\n"
                        '{\n  "success": true|false,\n  "explanation": "short explanation"\n}\n'
                    ),
                },
            ],
        },
    ]

    for img_path in screenshots:
        try:
            b64 = encode_image_base64(img_path)
            messages[1]["content"].append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            )
        except Exception:
            continue

    response_text = call_chat_completion(client, model, messages, max_tokens=600, temperature=0.0)
    success_val, explanation_val = parse_verification_result(response_text)
    if success_val is None or explanation_val is None:
        # Fallback: treat as failure with raw response
        return False, f"Unparseable verification response: {response_text}"
    return success_val, explanation_val


def detect_app_name_from_metadata(metadata: Dict[str, Any]) -> str:
    # Attempt to infer an app/domain for the summarization prompt
    domain = metadata.get("domain")
    if isinstance(domain, str) and domain.strip():
        return domain.strip()
    return "unknown"


def process_task_dir(
    client: OpenAI,
    model: str,
    task_dir: Path,
    max_images: int,
    dry_run: bool,
) -> None:
    metadata_path = task_dir / "metadata.json"
    trajectory_path = task_dir / "trajectory.jsonl"
    screenshots_dir = task_dir / "screenshots"

    if not metadata_path.exists() or not trajectory_path.exists():
        return

    try:
        with metadata_path.open("r") as f:
            metadata = json.load(f)
    except Exception:
        return
    
    # Filter out mismatched state replays
    if isinstance(metadata, dict):
        sv = metadata.get("state_verification")
        if isinstance(sv, dict) and sv.get("match") is False:
            expl = str(sv.get("explanation", "")).strip()
            print(f"[SKIP-MISMATCH] {task_dir.name} -> {expl}")
            return
    
    # Skip if verification already exists
    if isinstance(metadata, dict) and "verification" in metadata:
        print(f"[SKIP] {task_dir.name} -> verification already present")
        return

    trajectory = load_trajectory(trajectory_path)
    # Determine screenshot selection strategy based on total count
    all_candidates = [p for p in screenshots_dir.glob("*.png") if p.is_file()] if screenshots_dir.exists() else []
    has_too_many = max_images > 0 and len(all_candidates) > max_images
    if has_too_many:
        # If more than max, skip generating a new task description;
        # use the new_task_description from metadata and select the last max screenshots.
        selected_screenshots = collect_last_screenshots(screenshots_dir, max_images=max_images)
        task_description_to_verify = str(metadata.get("new_task_description", "")).strip()
    else:
        # Otherwise, downsample (uniform) and generate a unified task description from vLLM.
        selected_screenshots = collect_screenshots(screenshots_dir, max_images=max_images)
        task_description_to_verify = generate_unified_task_description(
            client=client,
            model=model,
            app_name=detect_app_name_from_metadata(metadata),
            trajectory=trajectory,
            screenshots=selected_screenshots,
            metadata=metadata,
        )
        # Expose generated description for reference
        metadata["generated_task_description_from_vllm"] = task_description_to_verify

    # Verify success using the chosen task description
    success, explanation = verify_task_success(
        client=client,
        model=model,
        task_description=task_description_to_verify,
        trajectory=trajectory,
        screenshots=selected_screenshots,
    )

    if dry_run:
        # In dry-run mode, do not write back to disk; just show what would happen.
        print(f"[DRY-RUN] {task_dir.name}")
        print(f"  task_description: {task_description_to_verify}")
        print(f"  success: {success}")
        print(f"  explanation: {explanation[:500]}")
    else:
        # Save verification results into metadata.json (overwrite existing verification)
        metadata["verification"] = {
            "success": success,
            "explanation": explanation,
        }
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"[UPDATED] {task_dir.name} -> success={success}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and verify tasks using Qwen via vLLM server.")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="/branch_gen_winarena_filtered",
        help="Root directory containing task subdirectories (each with metadata.json, trajectory.jsonl, screenshots/).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible base URL for the vLLM server (e.g., http://127.0.0.1:8000/v1).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-32B-Instruct",
        help="Model name as served by vLLM.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key if required by the server (vLLM often ignores; default EMPTY).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="Maximum number of screenshots to attach per task (uniformly downsampled if more).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, do not write back to metadata.json; just print results.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    client = create_client(base_url=args.base_url, api_key=args.api_key)

    # Iterate task directories
    for task_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        # Expect required files
        if not (task_dir / "trajectory.jsonl").exists() or not (task_dir / "metadata.json").exists():
            continue
        try:
            process_task_dir(
                client=client,
                model=args.model,
                task_dir=task_dir,
                max_images=args.max_images,
                dry_run=bool(args.dry_run),
            )
        except Exception as e:
            print(f"[ERROR] {task_dir.name}: {e}")



if __name__ == "__main__":
    main()


