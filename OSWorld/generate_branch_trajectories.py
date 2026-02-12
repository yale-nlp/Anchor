"""
Generate new tasks and gold trajectories from existing branch points.

This script:
1. Reads branch point analysis JSONs
2. For each branch point:
   - Replays trajectory up to that point
   - Collects previous 4 states (screenshots)
   - Uses GPT-5 to generate new tasks
   - Uses Claude 4.5 to execute the generated task
   - Refines task description during execution
   - Verifies success with VLLM
   - Saves successful trajectories
"""

import argparse
import base64
import datetime
import io
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Set, MutableMapping
from multiprocessing import Process, Manager, current_process

from PIL import Image

from dotenv import load_dotenv
from openai import AzureOpenAI
# from openai import OpenAI
from desktop_env.desktop_env import DesktopEnv
from mm_agents.qwen3vl_agent import Qwen3VLAgent
from mm_agents.anthropic import AnthropicAgent
# Load environment variables from .env file
load_dotenv()


# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

os.makedirs("logs", exist_ok=True)
datetime_str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

file_handler = logging.FileHandler(
    os.path.join("logs", f"branch-generation-{datetime_str}.log"), encoding="utf-8"
)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    "[%(asctime)s %(levelname)s %(module)s/%(lineno)d] %(message)s"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Initialize GPT-5 client (will be set after loading .env)
gpt5_client = None


def initialize_gpt5_client():
    """Initialize GPT-5 client from environment variables."""
    global gpt5_client
    
    # Azure API
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    
    if not api_key:
        raise ValueError(
            "AZURE_OPENAI_API_KEY not found in environment variables. "
            "Please set it in your .env file."
        )
    
    gpt5_client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key
    )
    
    # OpenAI API
    # api_key = os.getenv("OPENAI_API_KEY")
    
    # if not api_key:
    #     raise ValueError(
    #         "OPENAI_API_KEY not found in environment variables. "
    #         "Please set it in your .env file."
    #     )
    
    # gpt5_client = OpenAI(
    #     api_key=api_key
    # )
    logger.info("✓ GPT-5 client initialized successfully")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate branch trajectories")
    parser.add_argument("--branch_analysis_dir", type=str, 
                       default="branch_point_selected",
                       help="Directory containing branch point analysis JSONs")
    parser.add_argument("--processed_task_ids_path", type=str,
                       default=None,
                       help="Path to JSON file listing task_ids that are already fully processed and should be skipped")
    parser.add_argument("--task_ids", type=str, default=None,
                       help="Comma-separated task IDs to process (filters branch analysis files by task_id)")
    parser.add_argument("--trajectory_base_dir", type=str,
                       default="trajectory_data",
                       help="Base directory containing trajectory data")
    parser.add_argument("--output_dir", type=str,
                       default="branch_generated",
                       help="Output directory for generated trajectories")
    parser.add_argument("--config_base_dir", type=str,
                       default="evaluation_examples",
                       help="Base directory for task configs")
    parser.add_argument("--provider_name", type=str, default="aws",
                       help="Virtualization provider")
    parser.add_argument("--path_to_vm", type=str, default=None,
                       help="Path to VM configuration")
    parser.add_argument("--os_type", type=str, default="Ubuntu",
                       help="Guest OS type")
    parser.add_argument("--headless", action="store_true",
                       help="Run VM in headless mode")
    parser.add_argument("--pause", type=float, default=2.0,
                       help="Pause between actions when replaying")
    parser.add_argument("--region", type=str, default="us-east-1",
                       help="AWS region for VM")
    parser.add_argument("--screen_width", type=int, default=1920,
                       help="Screen width")
    parser.add_argument("--screen_height", type=int, default=1080,
                       help="Screen height")
    parser.add_argument("--client_password", type=str, default="",
                       help="Client password")
    parser.add_argument("--enable_proxy", action="store_true",
                       help="Enable proxy")
    parser.add_argument("--aws_profile", type=str, default=None,
                       help="Named AWS profile to use (overrides env credentials for this run)")
    parser.add_argument("--max_continuation_steps", type=int, default=30,
                       help="Max steps for Claude to continue task")
    parser.add_argument("--task_refinement_interval", type=int, default=1,
                       help="Refine task every N steps")
    parser.add_argument("--limit_tasks", type=int, default=None,
                       help="Limit number of tasks to process (for testing)")
    parser.add_argument("--limit_branches_per_task", type=int, default=None,
                       help="Limit branch points per task (for testing)")
    parser.add_argument("--num_new_tasks", type=int, default=2,
                       help="Default tasks per branch if JSON lacks 'num_tasks' (deprecated)")
    parser.add_argument("--num_envs", type=int, default=1,
                       help="Number of parallel processes (each handles one branch point at a time)")
    # Executor config
    parser.add_argument("--executor_agent", type=str, default="claude",
                       choices=["claude", "qwen3vl"],
                       help="Agent to execute generated tasks")
    # Qwen3VL config
    parser.add_argument("--qwen_model", type=str, default="qwen3-vl-flash-2025-10-15",
                       help="Qwen3-VL model name")
    parser.add_argument("--qwen_api_backend", type=str, default="dashscope",
                       choices=["dashscope", "openai"],
                       help="Backend for Qwen3-VL API")
    parser.add_argument("--qwen_enable_thinking", action="store_true",
                       help="Enable Qwen3-VL thinking mode (DashScope)")
    parser.add_argument("--qwen_thinking_budget", type=int, default=32768,
                       help="Thinking token budget for Qwen3-VL")
    parser.add_argument("--qwen_max_tokens", type=int, default=32768,
                       help="Max tokens for Qwen3-VL generation")
    parser.add_argument("--qwen_temperature", type=float, default=0.0,
                       help="Temperature for Qwen3-VL")
    parser.add_argument("--qwen_top_p", type=float, default=0.9,
                       help="Top-p for Qwen3-VL")
    parser.add_argument("--qwen_coord", type=str, default="relative",
                       choices=["absolute", "relative"],
                       help="Coordinate type for Qwen3-VL")
    parser.add_argument("--qwen_history_n", type=int, default=4,
                       help="Number of past steps Qwen3-VL conditions on")
    parser.add_argument("--qwen_add_thought_prefix", action="store_true",
                       help="Add thought prefix in Qwen3-VL response parsing")
    return parser.parse_args()


def iter_trajectory(path: Path) -> Iterator[Dict[str, Any]]:
    """Iterate through trajectory JSONL file."""
    with path.open("r", encoding="utf-8") as fp:
        for line_num, line in enumerate(fp, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON on line {line_num} of {path}") from exc


def load_task_config(task_id: str, domain: str, config_base_dir: Path) -> Optional[Dict[str, Any]]:
    """Load task configuration."""
    candidate_path = config_base_dir / "examples" / domain / f"{task_id}.json"
    if not candidate_path.exists():
        logger.warning(f"Task config not found: {candidate_path}")
        return None
    
    with candidate_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def find_trajectory_file(task_id: str, model: str, trajectory_base_dir: Path) -> Optional[Path]:
    """Find the trajectory file for a given task and model."""
    # Map model names to directory patterns
    model_dir_map = {
        "Claude-4.5": "results_claude-sonnet-4-5-20250929_100steps/claude_computer_use/screenshot/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "AGI0": "results_agi0_50steps",
        "UI-TARS": "results_UI-TARS-2-2509_100steps/pyautogui/screenshot/ep-20250926152455-74xj7",
        "UI-TARS-2-2509": "results_UI-TARS-2-2509_100steps/pyautogui/screenshot/ep-20250926152455-74xj7",
    }
    
    model_dir = model_dir_map.get(model)
    if not model_dir:
        logger.error(f"Unknown model: {model}")
        return None
    
    # Search for trajectory file
    base_path = trajectory_base_dir / model_dir
    if not base_path.exists():
        logger.error(f"Model directory not found: {base_path}")
        return None
    
    # Search through directories to find task_id
    for traj_file in base_path.rglob("*/traj.jsonl"):
        if task_id in str(traj_file):
            return traj_file
    
    logger.warning(f"Trajectory file not found for task {task_id} in {base_path}")
    return None


def load_existing_task_ids_from_output(output_dir: Path) -> Dict[Tuple[str, int], int]:
    """
    Load existing task outputs and count them by (task_id, branch_after_step).
    
    Returns:
        Dict mapping (task_id, branch_after_step) -> count of existing outputs
    """
    existing: Dict[Tuple[str, int], int] = {}
    try:
        if not output_dir.exists():
            return existing
        for sub in output_dir.iterdir():
            if not sub.is_dir():
                continue
            metadata_path = sub / "metadata.json"
            if not metadata_path.exists():
                continue
            try:
                with metadata_path.open("r", encoding="utf-8") as fp:
                    meta = json.load(fp)
                task_id = meta.get("original_task_id")
                branch_after_step = meta.get("branch_after_step")
                if isinstance(task_id, str) and task_id.strip() and isinstance(branch_after_step, int):
                    key = (task_id.strip(), branch_after_step)
                    existing[key] = existing.get(key, 0) + 1
            except Exception:
                # Ignore unreadable metadata
                continue
    except Exception:
        # If any unexpected error occurs, return best-effort collected set
        pass
    return existing


def normalize_task_description(desc: str) -> str:
    """
    Normalize a task description string for duplicate detection.
    
    Lowercases, strips leading/trailing whitespace, and collapses internal
    whitespace so that trivially rephrased copies map to the same key.
    """
    if not isinstance(desc, str):
        desc = str(desc)
    desc = desc.strip().lower()
    # Collapse all whitespace runs to a single space
    desc = re.sub(r"\s+", " ", desc)
    return desc


def add_task_to_seen_store(
    seen_tasks_store: Optional[MutableMapping[str, str]],
    description: str,
    lock: Any = None,
) -> bool:
    """
    Add a task description to the shared seen-tasks store.
    
    Args:
        seen_tasks_store: Mapping from normalized description -> original description.
        description: The new task description.
        lock: Optional multiprocessing lock for atomic check-and-set.
    
    Returns:
        True if this description was newly added (i.e., not seen before),
        False if it was already present.
    """
    if seen_tasks_store is None:
        # No global store configured, treat as always-new.
        return True
    norm = normalize_task_description(description)
    if not norm:
        return True
    if lock is not None:
        with lock:
            if norm in seen_tasks_store:
                return False
            seen_tasks_store[norm] = description
            return True
    # Single-process or non-shared mapping
    if norm in seen_tasks_store:
        return False
    seen_tasks_store[norm] = description
    return True


def load_seen_tasks_from_disk(store_path: Path, seen_tasks_store: MutableMapping[str, str]) -> None:
    """
    Load previously proposed tasks from disk into the shared store.
    
    The on-disk format is a JSON object:
        { "tasks": [ { "description": "..." }, ... ] }
    """
    if not store_path.exists():
        return
    try:
        with store_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        tasks = data.get("tasks", [])
        loaded = 0
        for item in tasks:
            if not isinstance(item, dict):
                continue
            desc = item.get("description")
            if not desc:
                continue
            norm = normalize_task_description(desc)
            if norm not in seen_tasks_store:
                seen_tasks_store[norm] = desc
                loaded += 1
        logger.info(f"Preloaded {loaded} previously proposed tasks from {store_path}")
    except Exception as e:
        logger.error(f"Failed to load seen tasks from {store_path}: {e}", exc_info=True)


def save_seen_tasks_to_disk(store_path: Path, seen_tasks_store: MutableMapping[str, str]) -> None:
    """
    Persist all seen tasks to a JSON file on disk.
    
    The file is written atomically via a temporary file + rename to avoid
    corruption if the process is interrupted mid-write.
    """
    try:
        # Use values() to get original descriptions; dedupe while preserving insertion order.
        unique_descs: List[str] = list(dict.fromkeys(seen_tasks_store.values()))
        payload = {"tasks": [{"description": d} for d in unique_descs]}
        tmp_path = store_path.with_suffix(store_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)
        tmp_path.replace(store_path)
        logger.info(f"Saved {len(unique_descs)} seen tasks to {store_path}")
    except Exception as e:
        logger.error(f"Failed to save seen tasks to {store_path}: {e}", exc_info=True)


def get_action_space_for_model(model: str) -> str:
    """Get the action space for a given model."""
    model_action_map = {
        "Claude-4.5": "claude_computer_use",
        "AGI0": "pyautogui",
        "UI-TARS": "pyautogui",
        "UI-TARS-2-2509": "pyautogui",
        "qwen3-vl": "pyautogui",
        "Qwen3-VL": "pyautogui",
    }
    return model_action_map.get(model, "pyautogui")


def extract_action_string(record_action: Any, action_space: str) -> Optional[str]:
    """Extract action command from trajectory record."""
    if record_action is None:
        return None
    
    if isinstance(record_action, str):
        return record_action
    
    if isinstance(record_action, dict):
        # Claude format
        if action_space == "claude_computer_use":
            command = record_action.get("command")
            if command and isinstance(command, str):
                return command.strip()
        # PyAutoGUI format
        else:
            return str(record_action)
    
    return None


def encode_image_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64."""
    return base64.b64encode(image_bytes).decode('utf-8')


def resize_for_claude(
    image_bytes: bytes,
    target_size: Tuple[int, int] = (1280, 720),
) -> bytes:
    """
    Resize screenshot bytes to Claude's expected display resolution.

    If anything goes wrong, fall back to the original bytes.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return image_bytes


# def refine_task_description_gpt5(
#     task_description: str,
#     current_screenshot: bytes,
#     actions_so_far: List[Dict[str, Any]],
#     step_num: int
# ) -> str:
#     """
#     Use GPT-5 to refine task description based on current progress.
    
#     Returns:
#         Refined task description or original if refinement fails
#     """
#     logger.info(f"Refining task description at step {step_num}")
    
#     action_summary = []
#     for i, action in enumerate(actions_so_far[-5:], start=max(1, len(actions_so_far) - 4)):
#         action_str = action.get('action', 'N/A')
#         if len(str(action_str)) > 100:
#             action_str = str(action_str)[:100] + "..."
#         action_summary.append(f"Step {i}: {action_str}")
    
#     messages = [
#         {
#             "role": "system",
#             "content": """You are verifying whether the current task description is still achievable.
# Do NOT rewrite or adjust the task unless it has clearly become impossible to complete based on the current state.
# Only modify the task description if The original task can no longer be completed from the current state.


# """
#         },
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": f"""# Current Task Description
# {task_description}

# # Actions Taken So Far:
# {chr(10).join(action_summary)}

# # Current Screenshot
# See attached screenshot showing current state.

# # Your Task
# Review the task description and current progress. If the task given is unachievable from the current state, modify the task description to be an achievable
# task from the current state and provide the modified version. 
# Otherwise, return the original description unchanged.
# So in most cases, you should keep the original description unchanged, change it only when it is clearly unachievable from the current state.

# Return ONLY the task description (modified or original), no extra text or formatting."""
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/png;base64,{encode_image_base64(current_screenshot)}"
#                     }
#                 }
#             ]
#         }
#     ]
    
#     try:
#         response = gpt5_client.chat.completions.create(
#             model="gpt-4o",  # Changed from "gpt-5-chat" to OpenAI model
#             messages=messages,
#             max_tokens=500,
#             temperature=0.5
#         )
        
#         refined_description = response.choices[0].message.content.strip()
#         logger.info(f"Refined task: {refined_description}")
#         return refined_description
    
#     except Exception as e:
#         logger.error(f"Error refining task description: {e}")
#         return task_description


def generate_new_tasks_gpt5(
    task_id: str,
    original_instruction: str,
    branch_point: Dict[str, Any],
    action_trajectory: List[Dict[str, Any]],
    screenshots: List[bytes],
    num_tasks: int = 1,
    existing_task_descriptions: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Use GPT-5 to generate new tasks that can be completed from this branch point.
    
    New logic:
    1) Summarize (≤3 sentences) what the agent has done so far using the original instruction,
       the actions and reasoning from prior steps, and the current state screenshot.
    2) Use that concise summary + current state screenshot to propose follow-up task descriptions.
    
    Args:
        task_id: Original task ID
        original_instruction: Original task instruction
        branch_point: Branch point info with 'after_step', 'reason', 'new_task_examples'
        action_trajectory: List of actions taken up to branch point (may include 'reasoning')
        screenshots: List of screenshot bytes (all screenshots from initial state to branch point)
        num_tasks: Number of tasks to generate
    
    Returns:
        List of task dicts with 'description' field.
    """
    logger.info(f"Generating {num_tasks} new tasks for branch point after step {branch_point['after_step']}")
    
    # Build compact but comprehensive prior steps digest including action + reasoning (all steps)
    steps_digest_lines: List[str] = []
    for rec in action_trajectory:
        step_no = rec.get("step")
        action_str = rec.get("action")
        if isinstance(action_str, dict):
            action_str = action_str.get("command", str(action_str))
        reasoning_str = rec.get("reasoning")
        # Truncate noisy content to keep token usage reasonable
        action_str = str(action_str) if action_str is not None else "N/A"
        if len(action_str) > 200:
            action_str = action_str[:200] + "..."
        if reasoning_str:
            reasoning_str = str(reasoning_str)
            if len(reasoning_str) > 400:
                reasoning_str = reasoning_str[:400] + "..."
        steps_digest_lines.append(
            f"- Step {step_no}: Action: {action_str}" + (f" | Reasoning: {reasoning_str}" if reasoning_str else "")
        )
    steps_digest = chr(10).join(steps_digest_lines)
    
    # Current screenshot (most recent)
    current_image_bytes = screenshots[-1] if screenshots else None
    
    # ---- Stage 1: Summarize progress in ≤3 sentences ----
    progress_summary: Optional[str] = None
    if action_trajectory:
        summarize_messages = [
            {
                "role": "system",
                "content": """You are an analyst. Given the original task, prior steps (action + reasoning), and the current UI state image,
                write a chronological summary (within 3 sentences) describing what the agent has done so far up to this point. Return only the summary text.
                The summary should focus on the trajectory about what the agent has done about the UI interface.
                """
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""#Task
                            Instruction: {original_instruction}
                            
                            # Branch Point
                            After step: {branch_point['after_step']}
                            
                            # Prior Steps (Action + Reasoning)
                            {steps_digest}
                            
                            # Current State
                            See image."""
                    }
                ]
            }
        ]
        if current_image_bytes:
            summarize_messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image_base64(current_image_bytes)}"
                }
            })
        try:
            logger.info("Requesting progress summary from GPT-5...")
            response_summary = gpt5_client.chat.completions.create(
                model="gpt-5.1-chat",  # Changed from "gpt-5.1-chat" to OpenAI model, use gpt-5.1 for now
                messages=summarize_messages,
                max_tokens=400,
                # max_completion_tokens=600,
                temperature=0.3
            )
            raw_content = response_summary.choices[0].message.content
            logger.info(f"Raw summary response: {raw_content}")
            progress_summary = (raw_content or "").strip()
            if progress_summary:
                logger.info(f"Progress summary (<=3 sentences): {progress_summary[:300]}{'...' if len(progress_summary)>300 else ''}")
            else:
                logger.warning("Progress summary is empty after API call succeeded")
        except Exception as e:
            logger.error(f"Error summarizing progress with GPT-5: {e}")
            progress_summary = None
    else:
        # Branch at step 0 (or no prior actions): there is nothing to summarize.
        progress_summary = (
            "The agent has not taken any actions yet; this branch starts from the "
            "initial state of the task environment."
        )
    
    # ---- Stage 2: Propose follow-up tasks using the summary + current state image ----
    # Include branch analysis context: reason this is a good branch point and example new tasks
    reason_text = branch_point.get("reason") or ""
    new_task_examples = branch_point.get("new_task_examples") or []
    if isinstance(new_task_examples, (list, tuple)):
        examples_formatted = "\n".join(f"- {ex}" for ex in new_task_examples)
    else:
        examples_formatted = str(new_task_examples) if new_task_examples else ""
    # Include a deduplicated, truncated list of previously proposed tasks so GPT-5
    # can avoid generating near-duplicates.
    existing_task_descriptions = existing_task_descriptions or []
    # Deduplicate while preserving order
    existing_task_descriptions = list(dict.fromkeys(existing_task_descriptions))
    max_prior_for_prompt = 50
    if existing_task_descriptions:
        limited_prior = existing_task_descriptions[:max_prior_for_prompt]
        existing_formatted = "\n".join(f"- {d}" for d in limited_prior)
        logger.info(f"Passing {len(limited_prior)} prior tasks to GPT-5 to discourage duplicates")
    else:
        existing_formatted = "N/A"
    gen_messages = [
        {
            "role": "system",
            "content": """You propose natural, feasible follow-up tasks based on the current UI state.
                
                When generating a task, describe it as a complete task that an expert could perform from start to finish, 
                not just what could be done from the current state. The task should reflect an overall meaningful general objective 
                that would be accomplished across a full trajectory.
                
                Requirements for each proposed task:
                - Be specific and clearly defined, but you should not be too specific about the steps to complete the task.
                - Must have verifiable success criteria.
                - The task should feel like a natural extension of what has been done so far.
                - Should be completable within 5–15 steps.
                - Be different from the original task.
                - It doesn't require any authentication or login.
                - Avoid proposing tasks that are redundant with or too similar to the list of previously proposed tasks provided.
                
                Return only the task descriptions."""
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""# Agent Progress Summary till current state
                                {(progress_summary or 'No summary available.')}
                                
                                # Original Task Description
                                {original_instruction}
                                
                                # Branch Point Reason (you can refer to this if you think it helps)
                                {reason_text or 'N/A'}
                                
                                # Example New Tasks from Branch Analysis (you can refer to this if you think it helps, but no need to be too dependent on it)
                                {examples_formatted or 'N/A'}
                                
                                # Previously Proposed Tasks (avoid duplicating these; propose clearly new tasks)
                                {existing_formatted}
                                
                                # Current State
                                See image.
                                
                                # Your Task
                                # Generate {num_tasks} specific, feasible tasks that can be completed from this current state.
                                The {num_tasks} tasks should be different from each other to increase the diversity of the tasks, and should be different with the previously proposed tasks.
                                Provide a clear general task description (what does the whole task overall do).
                                Return your response as a JSON object with a single key \"tasks\" whose value is a JSON array, e.g.:
                                {{
                                  "tasks": [
                                    {{
                                      "description": "General task description"
                                    }}
                                  ]
                                }}"""
                }
            ]
        }
    ]
    if current_image_bytes:
        gen_messages[1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image_base64(current_image_bytes)}"
            }
        })
    
    try:
        response = gpt5_client.chat.completions.create(
            model="gpt-5.1-chat",  # Changed from "gpt-5.1-chat" to OpenAI model
            messages=gen_messages,
            max_tokens=1200,
            # max_completion_tokens=1200,
            temperature=0.7,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "branch_tasks",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "tasks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "description": {
                                            "type": "string",
                                            "description": "General task description for a new branch task starting from the current state."
                                        }
                                    },
                                    "required": ["description"],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["tasks"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        )
        response_text = response.choices[0].message.content
        logger.info(f"GPT-5 tasks response: {response_text[:500]}...")
        # With response_format json_schema, the content should already be strict JSON.
        # We keep a small fallback for older-style responses.
        try:
            parsed = json.loads(response_text)
        except Exception:
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                parsed = []
        if isinstance(parsed, dict) and "tasks" in parsed:
            tasks_json = parsed["tasks"]
        else:
            tasks_json = parsed
        # Attach progress_summary to each task item if possible
        try:
            if isinstance(tasks_json, list):
                enriched_tasks = []
                for item in tasks_json:
                    if isinstance(item, dict):
                        item.setdefault("description", item.get("description", ""))
                        item["progress_summary"] = progress_summary
                        enriched_tasks.append(item)
                    else:
                        enriched_tasks.append({"description": str(item), "progress_summary": progress_summary})
                tasks_json = enriched_tasks
        except Exception:
            pass
        logger.info(f"Generated {len(tasks_json)} tasks")
        return tasks_json
    except Exception as e:
        logger.error(f"Error generating tasks with GPT-5: {e}")
        return []



def verify_task_success_vllm(
    task_description: str,
    final_screenshot: bytes,
    trajectory: List[Dict[str, Any]]
) -> Tuple[bool, str]:
    """
    Use VLLM to verify if the task was completed successfully.
    
    For now, we'll use GPT-5 as a stand-in for VLLM verification.
    You can replace this with actual VLLM API call.
    
    Returns:
        Tuple of (success: bool, explanation: str)
    """
    logger.info("Verifying task success with VLLM")
    
    # Prepare action summary
    action_summary = []
    for i, action in enumerate(trajectory[-10:], start=max(1, len(trajectory) - 9)):
        action_str = action.get('action', 'N/A')
        if len(str(action_str)) > 100:
            action_str = str(action_str)[:100] + "..."
        action_summary.append(f"Step {i}: {action_str}")
    
    # Collect the last up to 3 screenshots (newest last)
    recent_screenshots: List[bytes] = []
    try:
        traj_screens = [a.get('screenshot') for a in trajectory if isinstance(a.get('screenshot'), (bytes, bytearray))]
    except Exception:
        traj_screens = []
    if traj_screens:
        recent_screens = traj_screens[-3:]
    # Ensure the provided final_screenshot is included and is the newest
    if isinstance(final_screenshot, (bytes, bytearray)):
        if not recent_screens or recent_screens[-1] != final_screenshot:
            print(f"Final screenshot strange error")
            recent_screens.append(final_screenshot)
        # Keep only the last 3 with the final image as the newest
        recent_screens = recent_screens[-3:]
    
    messages = [
        {
            "role": "system",
            "content": """You are a task verification expert. Given a task description, 
            and the last 3 state screenshots, determine if the task was completed successfully.
            Be strict but fair in your assessment."""
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""# Task Description
{task_description}

# Recent Screenshots (newest last)
See the last 3 screenshots showing the final states.

# Your Task
Determine if the task was completed successfully.

Return your response in this exact format:
SUCCESS: true/false
CERTAINTY: The certainty of your answer, from 0 to 100.
EXPLANATION: Your detailed explanation of why the task succeeded or failed."""
                }
            ]
        }
    ]
    
    # Append images to the user message content
    try:
        if recent_screenshots:
            for img_bytes in recent_screenshots:
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image_base64(img_bytes)}"
                    }
                })
        else:
            # Fallback to the single final screenshot if no trajectory screenshots available
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image_base64(final_screenshot)}"
                }
            })
    except Exception:
        # Best-effort: still try to include the final screenshot
        try:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image_base64(final_screenshot)}"
                }
            })
        except Exception:
            pass
    
    try:
        response = gpt5_client.chat.completions.create(
            model="gpt-5.1-chat",  # Changed from "gpt-5-chat" to OpenAI model
            messages=messages,
            max_tokens=500,
            # max_completion_tokens=500,
            temperature=0.1
        )
        
        response_text = response.choices[0].message.content
        logger.info(f"VLLM verification response: {response_text}")
        
        # Parse response
        success_match = re.search(r'SUCCESS:\s*(true|false)', response_text, re.IGNORECASE)
        explanation_match = re.search(r'EXPLANATION:\s*(.+)', response_text, re.DOTALL)
        certainty_match = re.search(r'CERTAINTY:\s*(\d+)', response_text, re.IGNORECASE)
        
        success = success_match and success_match.group(1).lower() == 'true'
        explanation = explanation_match.group(1).strip() if explanation_match else response_text
        certainty = int(certainty_match.group(1)) if certainty_match else 0
        return success, explanation, certainty
    
    except Exception as e:
        logger.error(f"Error verifying task: {e}")
        return False, f"Verification error: {e}"


def verify_state_match_with_gpt(
    screenshot_a: bytes,
    screenshot_b: bytes
) -> Tuple[bool, str]:
    """
    Ask GPT-5 to judge whether two screenshots show the same UI state, using a strict JSON schema.
    
    Returns:
        Tuple of (match: bool, explanation: str)
    """
    logger.info("Verifying state match between replay and reference screenshots via GPT-5")
    messages = [
        {
            "role": "system",
            "content": (
                "You are a visual comparator. You are given two screenshots from a computer UI. "
                "Determine if they depict the SAME general UI state (layout, window, panels opened, etc.). "
                "Ignore small differences like cursor position, text color, tiny selection mismatch if the overall state and structure is identical. "
                "Respond following the enforced JSON schema with a boolean `match` and a short string `explanation`."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Compare these two screenshots and judge if they show the overall same UI state."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image_base64(screenshot_a)}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image_base64(screenshot_b)}"
                    }
                }
            ],
        },
    ]
    try:
        response = gpt5_client.chat.completions.create(
            model="gpt-5.1-chat",  # Changed from "gpt-5-chat" to OpenAI model
            messages=messages,
            max_tokens=300,
            # max_completion_tokens=600,
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "state_match",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "match": {
                                "type": "boolean",
                                "description": "True if the two screenshots depict the same overall UI state; false otherwise."
                            },
                            "explanation": {
                                "type": "string",
                                "description": "A short natural-language explanation of why the states do or do not match."
                            },
                        },
                        "required": ["match", "explanation"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        )
        text = (response.choices[0].message.content or "").strip()
        try:
            result = json.loads(text)
            match = bool(result.get("match"))
            explanation = str(result.get("explanation") or "")
            return match, explanation
        except Exception:
            # Fallback: be conservative and treat as mismatch
            return False, f"Unparseable response: {text}"
    except Exception as e:
        logger.error(f"Error during state match verification: {e}", exc_info=True)
        return False, f"Verification error: {e}"


def save_state_mismatch_record(
    output_dir: Path,
    task_id: str,
    domain: str,
    branch_after_step: int,
    task_description: str,
    explanation: str,
    reference_image: Optional[bytes],
    replay_image: Optional[bytes]
) -> None:
    """
    Persist a record indicating the replay state does not match the expected reference state.
    Saves a small metadata.json and both images for debugging.
    """
    branch_id = f"{task_id}_branch{branch_after_step}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    task_output_dir = output_dir / branch_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    # Write metadata
    metadata = {
        "original_task_id": task_id,
        "domain": domain,
        "branch_after_step": branch_after_step,
        "new_task_description": task_description,
        "state_verification": {
            "match": False,
            "explanation": explanation,
        },
        "generation_timestamp": datetime.datetime.now().isoformat(),
    }
    with open(task_output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    # Save images if available
    screenshots_dir = task_output_dir / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)
    try:
        if reference_image:
            with open(screenshots_dir / f"step_{branch_after_step}_reference.png", "wb") as fp:
                fp.write(reference_image)
    except Exception:
        pass
    try:
        if replay_image:
            with open(screenshots_dir / f"step_{branch_after_step}_replay.png", "wb") as fp:
                fp.write(replay_image)
    except Exception:
        pass
    logger.info(f"Saved state mismatch record to {task_output_dir}")


def replay_to_branch_point(
    env: DesktopEnv,
    traj_path: Path,
    branch_after_step: int,
    action_space: str,
    pause: float
) -> Tuple[List[Dict[str, Any]], List[bytes]]:
    """
    Replay trajectory up to the branch point.
    
    Returns:
        Tuple of (actions, screenshots) where screenshots includes initial state + all screenshots
    """
    logger.info(f"Replaying trajectory to step {branch_after_step}")
    
    actions = []
    screenshots = []
    
    # Capture initial screenshot before any actions
    initial_screenshot = env.controller.get_screenshot()
    screenshots.append(initial_screenshot)
    logger.info("Captured initial screenshot (before any actions)")
    
    step_idx = 0
    for record in iter_trajectory(traj_path):
        if step_idx >= branch_after_step:
            break
        
        action_field = record.get("action")
        action_str = extract_action_string(action_field, action_space)
        
        if action_str is None or (isinstance(action_str, str) and action_str.strip().upper() == "DONE"):
            step_idx += 1
            continue
        
        logger.info(f"Replaying step {step_idx + 1}: {str(action_str)[:100]}")
        
        try:
            obs, reward, done, info = env.step(action_str, pause=pause)
            
            # Save action and screenshot
            actions.append({
                'step': step_idx + 1,
                'action': action_str,
                'reward': reward,
                'done': done,
                'screenshot': obs['screenshot']
            })
            
            # Keep all screenshots for full context
            screenshots.append(obs['screenshot'])
            
            if done:
                logger.warning(f"Trajectory ended early at step {step_idx + 1}")
                break
        
        except Exception as e:
            logger.error(f"Error replaying step {step_idx + 1}: {e}")
            break
        
        step_idx += 1
    
    logger.info(f"Replayed {len(actions)} steps, collected {len(screenshots)} screenshots")
    return actions, screenshots


def load_context_from_disk(
    traj_path: Path,
    branch_after_step: int,
    action_space: str
) -> Tuple[List[Dict[str, Any]], List[bytes]]:
    """
    Load actions and screenshots up to the branch point directly from stored files,
    without replaying in an environment.
    
    This expects the trajectory JSONL and corresponding PNG screenshots to live in the
    same directory. It will try to map each step to a screenshot using common fields
    like 'screenshot_file' or 'screenshot_path'. If those are missing, it will fallback
    to searching for files that include the step number (e.g., 'step_{n}_*.png').
    """
    logger.info(f"Loading context from disk up to step {branch_after_step}")
    actions: List[Dict[str, Any]] = []
    screenshots: List[bytes] = []
    traj_dir = traj_path.parent
    
    def _try_read_image(image_path: Path) -> Optional[bytes]:
        try:
            with image_path.open("rb") as fp:
                return fp.read()
        except Exception:
            return None
    
    step_idx = 0
    for record in iter_trajectory(traj_path):
        if step_idx >= branch_after_step:
            break
        
        action_field = record.get("action")
        action_str = extract_action_string(action_field, action_space)
        
        # Skip empty/DONE actions in alignment with replay logic
        if action_str is None or (isinstance(action_str, str) and action_str.strip().upper() == "DONE"):
            step_idx += 1
            continue
        
        step_number = record.get("step_num", step_idx + 1)
        
        # Attempt to locate screenshot bytes
        screenshot_bytes: Optional[bytes] = None
        # 1) Direct field with filename or path
        screenshot_name = record.get("screenshot_file") or record.get("screenshot_path")
        if isinstance(screenshot_name, str):
            candidate = (traj_dir / screenshot_name)
            if not candidate.exists():
                # Some logs may store absolute-ish names; try basename in current dir
                candidate = traj_dir / Path(screenshot_name).name
            screenshot_bytes = _try_read_image(candidate)
        
        # 2) Fallback: glob by step number convention e.g., step_3_*.png
        if screenshot_bytes is None:
            try:
                candidates = sorted(traj_dir.glob(f"step_{step_number}_*.png"))
            except Exception:
                candidates = []
            if not candidates:
                # try a looser pattern that still includes the step number
                try:
                    candidates = sorted([p for p in traj_dir.glob("*.png") if f"step_{step_number}" in p.name])
                except Exception:
                    candidates = []
            if candidates:
                screenshot_bytes = _try_read_image(candidates[0])
        
        # Append action (include screenshot if we found it)
        action_record: Dict[str, Any] = {
            "step": step_number,
            "action": action_str,
        }
        # Try to include reasoning if present in the trajectory record
        try:
            reasoning_text = record.get("response")
            if not reasoning_text:
                act = record.get("action")
                if isinstance(act, dict):
                    reasoning_text = act.get("raw_response") or act.get("text")
            if reasoning_text:
                # Trim very long raw responses
                reasoning_trimmed = str(reasoning_text)
                if len(reasoning_trimmed) > 2000:
                    reasoning_trimmed = reasoning_trimmed[:2000] + "..."
                action_record["reasoning"] = reasoning_trimmed
        except Exception:
            pass
        if screenshot_bytes is not None:
            action_record["screenshot"] = screenshot_bytes
        actions.append(action_record)
        
        # Keep a parallel screenshots list for GPT-5 context
        if screenshot_bytes is not None:
            screenshots.append(screenshot_bytes)
        
        step_idx += 1
    
    logger.info(f"Loaded {len(actions)} actions and {len(screenshots)} screenshots from disk")
    return actions, screenshots


def populate_agent_history(
    agent: AnthropicAgent,
    prior_actions: List[Dict],
    prior_screenshots: List[bytes],
    task_description: str
):
    """
    Pre-populate the agent's message history with prior trajectory context.
    
    This provides Claude with visual context of how the system reached the current state,
    showing the progression of screenshots and actions without pretending Claude made those actions.
    
    Args:
        agent: The AnthropicAgent instance
        prior_actions: List of actions from before the branch point
        prior_screenshots: List of screenshots from before the branch point
    """
    if not prior_actions or not prior_screenshots:
        logger.info("No prior context to populate")
        return
    
    logger.info(f"Populating agent history with {len(prior_actions)} prior steps")
    
    # Add initial screenshot with context explanation
    if len(prior_screenshots) > 0:
        # Ensure screenshot resolution matches Claude's 1280x720 expectations
        init_resized = resize_for_claude(prior_screenshots[0])
        init_screenshot_base64 = base64.b64encode(init_resized).decode('utf-8')
        agent.messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": init_screenshot_base64,
                    },
                },
                {
                    "type": "text",
                    "text": task_description
                }
            ]
        })
    
    # Add each prior action as assistant+user pair (tool_use + tool_result)
    # This ensures images are treated as tool_result images and subject to filtering
    for idx, action_dict in enumerate(prior_actions):
        action_str = action_dict.get('action', '')
        
        # Truncate very long actions for readability
        if len(str(action_str)) > 200:
            action_str = str(action_str)[:200] + "..."
        
        # Add assistant message with tool use (simulating prior action)
        tool_use_id = f"context_step_{idx}"
        agent.messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"[Context - prior step {idx + 1}]"
                },
                {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": "computer",
                    "input": {"action": "key", "text": "context"}  # Dummy action
                }
            ]
        })
        
        # Add tool result with screenshot (will be subject to image filtering)
        screenshot_idx = idx + 1
        if screenshot_idx < len(prior_screenshots):
            resized_bytes = resize_for_claude(prior_screenshots[screenshot_idx])
            screenshot_base64 = base64.b64encode(resized_bytes).decode('utf-8')
            agent.messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": [
                            {"type": "text", "text": f"[Prior action: {action_str}]"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": screenshot_base64
                                }
                            }
                        ]
                    }
                ]
            })
    
    logger.info(f"Agent history populated with {len(prior_actions)} prior steps as tool_result format")


def populate_qwen_history(
    agent: Qwen3VLAgent,
    prior_actions: List[Dict],
    prior_screenshots: List[bytes],
):
    """
    Pre-populate Qwen3-VL agent internal history so its message builder
    (OpenAI/DashScope-style image_url + text) includes prior context.
    """
    if not prior_actions or not prior_screenshots:
        logger.info("No prior context to populate for Qwen3-VL")
        return
    try:
        # Use the last N steps based on agent.history_n
        history_n = getattr(agent, "history_n", 4) or 4
        count = min(history_n, len(prior_actions), len(prior_screenshots))
        # Take the last 'count' items to form recent history
        actions_hist = prior_actions[-count:]
        screenshots_hist = prior_screenshots[-count:]
        # Build responses from reasoning if available, else from action text
        responses_hist: List[str] = []
        actions_text_hist: List[str] = []
        screenshots_b64_hist: List[str] = []
        for a, img_bytes in zip(actions_hist, screenshots_hist):
            action_str = a.get("action", "")
            reasoning_str = a.get("reasoning")
            if reasoning_str:
                text = str(reasoning_str)
            else:
                text = f"Action: {str(action_str)}"
            if len(text) > 2000:
                text = text[:2000] + "..."
            responses_hist.append(text)
            actions_text = str(action_str)
            if len(actions_text) > 400:
                actions_text = actions_text[:400] + "..."
            actions_text_hist.append(actions_text)
            try:
                screenshots_b64_hist.append(base64.b64encode(img_bytes).decode("utf-8"))
            except Exception:
                # Skip image if any encoding error
                pass
        # Assign to agent internals; Qwen3VLAgent.predict will consume these
        agent.responses = responses_hist
        agent.actions = actions_text_hist
        agent.screenshots = screenshots_b64_hist
        logger.info(f"Qwen3-VL history populated with {len(responses_hist)} prior steps")
    except Exception as e:
        logger.warning(f"Failed to populate Qwen3-VL history: {e}", exc_info=False)

def execute_task_with_claude(
    env: DesktopEnv,
    task_description: str,
    max_steps: int,
    refinement_interval: int,
    screen_size: Tuple[int, int],
    provider_name: str = "aws",
    prior_actions: List[Dict] = None,
    prior_screenshots: List[bytes] = None
) -> Tuple[bool, List[Dict[str, Any]], str, Dict[int, str]]:
    """
    Execute a task using Claude 4.5 agent.
    
    Args:
        env: Desktop environment
        task_description: The branch task description
        max_steps: Maximum steps to execute
        refinement_interval: Interval for task refinement
        screen_size: Screen dimensions
        provider_name: Cloud provider name
        original_task: The original task instruction (for context)
        prior_actions: Actions taken before the branch point
        prior_screenshots: Screenshots from before the branch point
    
    Returns:
        Tuple of (completed: bool, trajectory: List, final_task_description: str)
    """
    logger.info(f"Executing task with Claude: {task_description}")
    
    # Initialize Claude agent. We explicitly pass screen_size so that
    # AnthropicAgent can compute the correct resize_factor for mapping
    # 1280x720 tool coordinates back to the real display resolution.
    agent = AnthropicAgent(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        action_space="claude_computer_use",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        screen_size=screen_size,
    )
    
    # Pre-populate agent's message history with prior context if provided
    if prior_actions and prior_screenshots:
        populate_agent_history(agent, prior_actions, prior_screenshots, task_description)
        
        # Add the new task instruction with current screenshot
        # This is necessary because predict() only adds initial message if messages is empty
        current_screenshot = env.controller.get_screenshot()
        # Resize to 1280x720 so what Claude sees matches the tool display spec
        resized_current = resize_for_claude(current_screenshot)
        screenshot_base64 = base64.b64encode(resized_current).decode('utf-8')
        agent.messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": screenshot_base64,
                    },
                },
                {
                    "type": "text",
                    "text": task_description
                }
            ]
        })
        logger.info("Added new task instruction to agent history")
    
    trajectory = []
    current_task_description = task_description
    # Track refined task descriptions by step (include initial at step 0)
    refined_descriptions_by_step: Dict[int, str] = {0: task_description}
    
    step = 0
    done = False
    
    while step < max_steps and not done:
        step += 1
        logger.info(f"Claude step {step}/{max_steps}")
        
        # Task refinement disabled
        # if step > 1 and step % refinement_interval == 0:
        #     current_screenshot = env.controller.get_screenshot()
        #     current_task_description = refine_task_description_gpt5(
        #         current_task_description,
        #         current_screenshot,
        #         trajectory,
        #         step
        #     )
        #     # Record refinement tied to this step
        #     refined_descriptions_by_step[step] = current_task_description
        
        try:
            # Get current observation
            obs = {
                'screenshot': env.controller.get_screenshot()
            }
            
            # Prepare instruction for Claude
            instruction = current_task_description  # Continue with current task
            
            # Get action from Claude
            # Claude.predict() returns (reasonings: str, actions: List[Dict])
            reasonings, actions = agent.predict(
                task_instruction=instruction,
                obs=obs
            )
            
            logger.info(f"Claude reasoning: {str(reasonings)[:200]}")
            logger.info(f"Claude actions: {actions}")
            
            # Check for terminal actions
            if isinstance(actions, list) and len(actions) > 0:
                if isinstance(actions[0], str) and actions[0].strip().upper() in ["DONE", "FAIL"]:
                    done = True
                    trajectory.append({
                        'step': step,
                        'action': actions[0],
                        'reasoning': reasonings,
                        'terminal': True
                    })
                    break
                
                # Extract action command
                action_dict = actions[0]
                if isinstance(action_dict, dict):
                    action_str = action_dict.get('command', str(action_dict))
                else:
                    action_str = str(action_dict)
            else:
                # No actions returned, treat as DONE
                done = True
                trajectory.append({
                    'step': step,
                    'action': 'DONE',
                    'reasoning': reasonings,
                    'terminal': True
                })
                break
            
            logger.info(f"Executing action: {str(action_str)[:150]}")
            
            # Execute action
            obs_result, reward, done, info = env.step(action_str, pause=2.0)
            
            # Record trajectory
            trajectory.append({
                'step': step,
                'action': action_str,
                'action_dict': action_dict if isinstance(action_dict, dict) else None,
                'reasoning': reasonings,
                'reward': reward,
                'done': done,
                'info': info,
                'screenshot': obs_result['screenshot']
            })
            
        except Exception as e:
            logger.error(f"Error during Claude execution at step {step}: {e}", exc_info=True)
            trajectory.append({
                'step': step,
                'error': str(e)
            })
            break
    
    completed = done or step >= max_steps
    logger.info(f"Claude execution {'completed' if completed else 'incomplete'} after {step} steps")
    
    return completed, trajectory, current_task_description, refined_descriptions_by_step


def execute_task_with_qwen3vl(
    env: DesktopEnv,
    task_description: str,
    max_steps: int,
    refinement_interval: int,
    screen_size: Tuple[int, int],
    qwen_args: argparse.Namespace,
    prior_actions: List[Dict] = None,
    prior_screenshots: List[bytes] = None
) -> Tuple[bool, List[Dict[str, Any]], str, Dict[int, str]]:
    """
    Execute a task using Qwen3-VL agent.
    """
    logger.info(f"Executing task with Qwen3-VL: {task_description}")
    agent = Qwen3VLAgent(
        model=qwen_args.qwen_model,
        max_tokens=qwen_args.qwen_max_tokens,
        top_p=qwen_args.qwen_top_p,
        temperature=qwen_args.qwen_temperature,
        action_space="pyautogui",
        observation_type="screenshot",
        history_n=qwen_args.qwen_history_n,
        add_thought_prefix=qwen_args.qwen_add_thought_prefix,
        coordinate_type=qwen_args.qwen_coord,
        api_backend=qwen_args.qwen_api_backend,
        enable_thinking=qwen_args.qwen_enable_thinking,
        thinking_budget=qwen_args.qwen_thinking_budget,
    )
    try:
        agent.reset(logger)
    except Exception:
        agent.reset()

    # Pre-populate Qwen3-VL internal history with prior context if provided
    if prior_actions and prior_screenshots:
        populate_qwen_history(agent, prior_actions, prior_screenshots)

    trajectory: List[Dict[str, Any]] = []
    current_task_description = task_description
    refined_descriptions_by_step: Dict[int, str] = {0: task_description}

    step = 0
    done = False
    while step < max_steps and not done:
        step += 1
        logger.info(f"Qwen step {step}/{max_steps}")

        # Task refinement disabled
        # if step > 1 and refinement_interval > 0 and step % refinement_interval == 0:
        #     try:
        #         current_screenshot = env.controller.get_screenshot()
        #         current_task_description = refine_task_description_gpt5(
        #             current_task_description,
        #             current_screenshot,
        #             trajectory,
        #             step
        #         )
        #         refined_descriptions_by_step[step] = current_task_description
        #     except Exception as e:
        #         logger.warning(f"Qwen refinement failed at step {step}: {e}")

        try:
            obs = {'screenshot': env.controller.get_screenshot()}
            response, actions = agent.predict(
                instruction=current_task_description,
                obs=obs
            )
            logger.info(f"Qwen response: {str(response)[:200]}")
            logger.info(f"Qwen actions: {actions}")

            if not actions:
                # No actions returned, treat as DONE
                done = True
                trajectory.append({
                    'step': step,
                    'action': 'DONE',
                    'reasoning': response,
                    'terminal': True
                })
                break

            for action_str in actions:
                logger.info(f"Executing action: {str(action_str)[:150]}")
                obs_result, reward, done, info = env.step(action_str, pause=2.0)
                trajectory.append({
                    'step': step,
                    'action': action_str,
                    'reasoning': response,
                    'reward': reward,
                    'done': done,
                    'info': info,
                    'screenshot': obs_result.get('screenshot')
                })
                if done:
                    break

        except Exception as e:
            logger.error(f"Error during Qwen execution at step {step}: {e}", exc_info=True)
            trajectory.append({
                'step': step,
                'error': str(e)
            })
            break

    completed = done or step >= max_steps
    logger.info(f"Qwen execution {'completed' if completed else 'incomplete'} after {step} steps")
    return completed, trajectory, current_task_description, refined_descriptions_by_step


def save_successful_trajectory(
    output_dir: Path,
    task_id: str,
    domain: str,
    branch_after_step: int,
    original_task: Dict[str, Any],
    new_task_description: str,
    progress_summary: Optional[str],
    replay_actions: List[Dict[str, Any]],
    new_actions: List[Dict[str, Any]],
    refined_descriptions_by_step: Dict[int, str],
    initial_screenshot: Optional[bytes] = None,
    # verification_result: Dict[str, Any]
):
    """Save a trajectory to disk (both successful and failed)."""
    # Create output directory
    branch_id = f"{task_id}_branch{branch_after_step}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    task_output_dir = output_dir / branch_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        "original_task_id": task_id,
        "domain": domain,
        "branch_after_step": branch_after_step,
        "original_instruction": original_task.get("instruction", ""),
        "new_task_description": new_task_description,
        "refined_task_descriptions_by_step": {str(k): v for k, v in (refined_descriptions_by_step or {}).items()},
        "progress_summary": progress_summary,
        "generation_timestamp": datetime.datetime.now().isoformat(),
        "num_replay_steps": len(replay_actions),
        "num_new_steps": len(new_actions),
        "total_steps": len(replay_actions) + len(new_actions),
        # "verification": verification_result
    }
    
    with open(task_output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save screenshots from both replay and new actions
    screenshot_dir = task_output_dir / "screenshots"
    screenshot_dir.mkdir(exist_ok=True)

    # Save initial environment screenshot as step_0_replay.png if provided
    if isinstance(initial_screenshot, (bytes, bytearray)):
        step0_path = screenshot_dir / "step_0_replay.png"
        try:
            with step0_path.open("wb") as f:
                f.write(initial_screenshot)
        except Exception as e:
            logger.warning(f"Failed to save step_0_replay screenshot to {step0_path}: {e}")
    
    # Save replay screenshots
    for action in replay_actions:
        if 'screenshot' in action:
            screenshot_path = screenshot_dir / f"step_{action['step']}_replay.png"
            with open(screenshot_path, "wb") as f:
                f.write(action['screenshot'])
    
    # Save new action screenshots
    for action in new_actions:
        if 'screenshot' in action:
            screenshot_path = screenshot_dir / f"step_{action['step']}.png"
            with open(screenshot_path, "wb") as f:
                f.write(action['screenshot'])
    
    # Save full trajectory (replay + new actions) without screenshots
    # Remove screenshot fields before JSON serialization
    with open(task_output_dir / "trajectory.jsonl", "w") as f:
        for action in replay_actions + new_actions:
            action_copy = action.copy()
            action_copy.pop('screenshot', None)  # Remove screenshot bytes
            f.write(json.dumps(action_copy))
            f.write("\n")
    
    logger.info(f"Saved successful trajectory to {task_output_dir}")


def process_branch_point(
    branch_analysis: Dict[str, Any],
    branch_point: Dict[str, Any],
    args: argparse.Namespace,
    remaining_count: Optional[int] = None,
    seen_tasks_store: Optional[MutableMapping[str, str]] = None,
    seen_tasks_lock: Any = None,
) -> int:
    """
    Process a single branch point.
    
    Args:
        branch_analysis: Branch analysis data for this task
        branch_point: Specific branch point to process
        args: Command-line arguments
        remaining_count: Number of tasks to generate (overrides num_tasks from branch_point/args)
    
    Returns:
        Number of successful trajectories generated
    """
    task_id = branch_analysis["task_id"]
    model = branch_analysis["model"]
    domain = branch_analysis["app_type"]
    branch_after_step = branch_point["after_step"]
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing: {task_id}, Branch after step {branch_after_step}")
    logger.info(f"{'='*80}")
    
    # Find trajectory file
    traj_path = find_trajectory_file(task_id, model, Path(args.trajectory_base_dir))
    if not traj_path:
        logger.error(f"Could not find trajectory file for {task_id}")
        return 0
    
    # Load task config
    task_config = load_task_config(task_id, domain, Path(args.config_base_dir))
    if not task_config:
        logger.error(f"Could not load task config for {task_id}")
        return 0
    
    action_space = get_action_space_for_model(model)
    
    # Initialize environment
    env_kwargs = {
        "provider_name": args.provider_name,
        "path_to_vm": args.path_to_vm,
        "os_type": args.os_type,
        "action_space": action_space,
        "headless": args.headless,
    }
    
    if args.provider_name == "aws":
        from desktop_env.providers.aws.manager import IMAGE_ID_MAP
        screen_size = (args.screen_width, args.screen_height)
        ami_id = IMAGE_ID_MAP[args.region].get(screen_size, IMAGE_ID_MAP[args.region][(1920, 1080)])
        
        env_kwargs.update({
            "region": args.region,
            "snapshot_name": ami_id,
            "screen_size": screen_size,
            "enable_proxy": args.enable_proxy,
            "client_password": args.client_password,
        })
    
    successful_count = 0
    
    try:
        env = DesktopEnv(**env_kwargs)
        logger.info("Environment initialized")
        
        try:
            # Reset environment with task config
            env.reset(task_config=task_config)
            logger.info("Environment reset with task config")

            # Load prior context up to the branch point directly from stored data (no replay)
            replay_actions, screenshots = load_context_from_disk(
                traj_path, branch_after_step, action_space
            )

            # Special handling for branch point at step 0:
            # there are no prior actions by definition, but we still want
            # the current UI state screenshot as context for GPT-5.
            if not replay_actions:
                if branch_after_step == 0:
                    logger.info(
                        "Branch point after_step=0 detected; "
                        "using initial environment screenshot as prior context"
                    )
                    try:
                        initial_screen = env.controller.get_screenshot()
                        if isinstance(initial_screen, (bytes, bytearray)):
                            screenshots = [initial_screen]
                        else:
                            screenshots = []
                            logger.warning(
                                "Initial screenshot for branch_after_step=0 "
                                "was not bytes; continuing without screenshot context"
                            )
                    except Exception as e:
                        screenshots = []
                        logger.warning(
                            "Failed to capture initial screenshot for branch_after_step=0: %s",
                            e,
                        )
                else:
                    logger.error("Failed to load prior context from disk")
                    return 0
            
            # Determine number of tasks for this branch point
            # Priority: remaining_count > branch_point.num_tasks > args.num_new_tasks
            if remaining_count is not None:
                num_tasks_for_branch = remaining_count
            else:
                num_tasks_for_branch = branch_point.get("num_tasks", args.num_new_tasks)
                try:
                    num_tasks_for_branch = int(num_tasks_for_branch)
                except Exception:
                    num_tasks_for_branch = args.num_new_tasks

            # Prepare a list of previously proposed task descriptions (if any)
            # to pass into GPT-5 so it can avoid generating duplicates.
            existing_for_prompt: Optional[List[str]] = None
            if seen_tasks_store is not None:
                try:
                    # Use the original descriptions stored as values
                    existing_for_prompt = list(seen_tasks_store.values())
                except Exception:
                    existing_for_prompt = None

            # Generate new tasks with GPT-5
            new_tasks = generate_new_tasks_gpt5(
                task_id=task_id,
                original_instruction=task_config.get("instruction", ""),
                branch_point=branch_point,
                action_trajectory=replay_actions,
                screenshots=screenshots,
                num_tasks=num_tasks_for_branch,
                existing_task_descriptions=existing_for_prompt,
            )
            
            if not new_tasks:
                logger.warning("No new tasks generated")
                return 0

            # Filter out any tasks whose descriptions were already seen globally.
            # This protects against generating similar tasks at different branch points
            # or in different worker processes.
            if seen_tasks_store is not None:
                filtered_tasks: List[Dict[str, Any]] = []
                for t in new_tasks:
                    desc = t.get("description", "")
                    if not desc:
                        continue
                    is_new = add_task_to_seen_store(seen_tasks_store, desc, seen_tasks_lock)
                    if not is_new:
                        logger.info(
                            f"Skipping already-seen generated task for {task_id} "
                            f"branch_after_step={branch_after_step}: {desc[:200]}"
                        )
                        continue
                    filtered_tasks.append(t)
                new_tasks = filtered_tasks
                if not new_tasks:
                    logger.warning(
                        "All GPT-5-generated tasks for this branch were duplicates of previously "
                        "seen tasks; nothing to execute for this branch point."
                    )
                    return 0
            
            # Process each generated task
            for task_idx, new_task in enumerate(new_tasks, 1):
                logger.info(f"\n--- Processing generated task {task_idx}/{len(new_tasks)} ---")
                logger.info(f"Task: {new_task['description']}")
                
                try:
                    # Reset environment to branch point state
                    try:
                        reset_obs = env.reset(task_config=task_config)
                    except Exception as e:
                        logger.error(f"Failed to reset environment for task {task_id}: {e}", exc_info=True)
                        continue

                    # Capture an explicit initial screenshot after reset.
                    # We poll every 5 seconds (up to 60 seconds total) until
                    # we successfully obtain valid screenshot bytes.
                    initial_screenshot = None
                    max_wait_seconds = 60.0
                    poll_interval = 5.0
                    start_time = time.time()
                    while time.time() - start_time < max_wait_seconds:
                        try:
                            candidate = env.controller.get_screenshot()
                            if isinstance(candidate, (bytes, bytearray)):
                                initial_screenshot = candidate
                                break
                            else:
                                logger.warning(
                                    "Got non-bytes initial screenshot after reset; retrying in %.1fs",
                                    poll_interval,
                                )
                        except Exception as e:
                            logger.warning(
                                "Error capturing initial step_0_replay screenshot after reset: %s. Retrying in %.1fs",
                                e,
                                poll_interval,
                            )
                        time.sleep(poll_interval)
                    if initial_screenshot is None:
                        logger.error(
                            "Failed to capture a valid initial step_0_replay screenshot within %.1f seconds.",
                            max_wait_seconds,
                        )

                    replay_actions_retry, replay_screenshots_retry = replay_to_branch_point(
                        env, traj_path, branch_after_step, action_space, args.pause
                    )
                    
                    # Verify the replayed state matches reference disk state before proceeding
                    try:
                        reference_last = screenshots[-1] if screenshots else None  # from load_context_from_disk above
                        replay_last = replay_screenshots_retry[-1] if replay_screenshots_retry else None
                    except Exception:
                        reference_last, replay_last = None, None
                    if reference_last and replay_last:
                        match, match_expl = verify_state_match_with_gpt(replay_last, reference_last)
                        if not match:
                            logger.error(f"State mismatch at branch_after_step={branch_after_step}: {match_expl}")
                            # Save a mismatch record and skip executing this task
                            save_state_mismatch_record(
                                output_dir=Path(args.output_dir),
                                task_id=task_id,
                                domain=domain,
                                branch_after_step=branch_after_step,
                                task_description=new_task.get("description", ""),
                                explanation=match_expl,
                                reference_image=reference_last,
                                replay_image=replay_last,
                            )
                            continue
                    else:
                        logger.warning("Skipping state verification due to missing reference or replay screenshot.")
                    
                    # Execute task with selected agent, providing prior context
                    if args.executor_agent == "qwen3vl":
                        completed, new_trajectory, final_task_desc, refined_map = execute_task_with_qwen3vl(
                            env=env,
                            task_description=new_task["description"],
                            max_steps=args.max_continuation_steps,
                            refinement_interval=args.task_refinement_interval,
                            screen_size=(args.screen_width, args.screen_height),
                            qwen_args=args,
                            prior_actions=replay_actions_retry,
                            prior_screenshots=replay_screenshots_retry
                        )
                    else:
                        completed, new_trajectory, final_task_desc, refined_map = execute_task_with_claude(
                            env=env,
                            task_description=new_task["description"],
                            max_steps=args.max_continuation_steps,
                            refinement_interval=args.task_refinement_interval,
                            screen_size=(args.screen_width, args.screen_height),
                            provider_name=args.provider_name,
                            prior_actions=replay_actions_retry,
                            prior_screenshots=replay_screenshots_retry
                        )
                    
                    if not completed or not new_trajectory:
                        logger.warning(f"Task {task_idx} did not complete")
                        continue
                    
                    # Get final screenshot
                    final_screenshot = new_trajectory[-1].get('screenshot')
                    if not final_screenshot:
                        final_screenshot = env.controller.get_screenshot()
                    
                    # Verify success with VLLM
                    # success, explanation, certainty = verify_task_success_vllm(
                    #     task_description=final_task_desc,
                    #     final_screenshot=final_screenshot,
                    #     trajectory=new_trajectory
                    # )
                    # Temporarily disable verification
                    # success = True
                    # explanation = "Verification skipped"
                    # certainty = 0
                    
                    # logger.info(f"Verification result: {'SUCCESS' if success else 'FAILED'}")
                    # logger.info(f"Explanation: {explanation}")
                    # logger.info(f"Certainty: {certainty}")
                    
                    # Save trajectory regardless of verification result
                    save_successful_trajectory(
                        output_dir=Path(args.output_dir),
                        task_id=task_id,
                        domain=domain,
                        branch_after_step=branch_after_step,
                        original_task=task_config,
                        new_task_description=final_task_desc,
                        progress_summary=new_task.get("progress_summary"),
                        replay_actions=replay_actions_retry,
                        new_actions=new_trajectory,
                        refined_descriptions_by_step=refined_map,
                        initial_screenshot=initial_screenshot,
                        # verification_result={
                        #     "success": success,
                        #     "explanation": explanation,
                        #     "certainty": certainty
                        # }
                    )
                    
                    # Count only completed trajectories
                    if completed:
                        successful_count += 1
                
                except Exception as e:
                    logger.error(f"Error processing task {task_idx}: {e}", exc_info=True)
                    continue
        
        finally:
            env.close()
            logger.info("Environment closed")
    
    except Exception as e:
        logger.error(f"Error processing branch point: {e}", exc_info=True)
    
    return successful_count


def _worker_process(task_queue, result_list, seen_tasks_store, seen_tasks_lock, args: argparse.Namespace) -> None:
    """Worker process that pulls branch-point tasks and processes them."""
    process_name = current_process().name
    logger.info(f"Process {process_name} started.")
    # Ensure GPT-5 client is initialized in each spawned process
    try:
        initialize_gpt5_client()
    except Exception as e:
        logger.error(f"[{process_name}] Failed to initialize GPT-5 client: {e}", exc_info=True)
        # If GPT-5 client init fails, drain queue items as failed to avoid stalling
        while True:
            try:
                task_queue.get(timeout=3)
            except Exception:
                break
        return
    # Optionally set AWS profile per worker if provided
    if getattr(args, "aws_profile", None):
        os.environ["AWS_PROFILE"] = args.aws_profile
    while True:
        try:
            item = task_queue.get(timeout=5)
        except Exception:
            logger.info(f"Process {process_name} finished (no more tasks).")
            break
        branch_analysis, branch_point, remaining_count = item
        try:
            succ = process_branch_point(
                branch_analysis,
                branch_point,
                args,
                remaining_count,
                seen_tasks_store=seen_tasks_store,
                seen_tasks_lock=seen_tasks_lock,
            )
            result_list.append(int(succ))
            logger.info(f"[{process_name}] ✓ Completed branch after_step={branch_point.get('after_step')} with {succ} successes")
        except Exception as e:
            logger.error(f"[{process_name}] ✗ Error on branch after_step={branch_point.get('after_step')}: {e}", exc_info=True)
            result_list.append(0)


def main():
    args = parse_args()
    
    logger.info("Starting branch trajectory generation")
    logger.info(f"Args: {vars(args)}")
    
    # Initialize GPT-5 client from .env
    try:
        initialize_gpt5_client()
    except ValueError as e:
        logger.error(f"Failed to initialize GPT-5 client: {e}")
        logger.error("Please check your .env file and ensure OPENAI_API_KEY is set.")
        return
    
    # Load branch analysis files
    branch_analysis_dir = Path(args.branch_analysis_dir)
    if not branch_analysis_dir.exists():
        logger.error(f"Branch analysis directory not found: {branch_analysis_dir}")
        return

    # Optionally load a list of task_ids that should be fully skipped
    processed_task_ids: Set[str] = set()
    if getattr(args, "processed_task_ids_path", None):
        processed_path = Path(args.processed_task_ids_path)
        if not processed_path.exists():
            logger.warning(f"Processed task ids file not found: {processed_path} (continuing without it)")
        else:
            try:
                with processed_path.open("r", encoding="utf-8") as fp:
                    data = json.load(fp)
                if isinstance(data, dict):
                    # Prefer a clear key if present, otherwise use all string values
                    if "processed_task_ids" in data and isinstance(data["processed_task_ids"], list):
                        processed_task_ids = {str(t).strip() for t in data["processed_task_ids"] if str(t).strip()}
                    else:
                        processed_task_ids = {str(v).strip() for v in data.values() if isinstance(v, str) and v.strip()}
                elif isinstance(data, list):
                    processed_task_ids = {str(t).strip() for t in data if str(t).strip()}
                else:
                    logger.warning(f"Unrecognized format in processed_task_ids file: {processed_path}; expected list or dict")
                if processed_task_ids:
                    logger.info(f"Loaded {len(processed_task_ids)} processed task_ids from {processed_path}")
            except Exception as e:
                logger.error(f"Failed to load processed_task_ids from {processed_path}: {e}", exc_info=True)
    
    # Get all JSON files
    json_files = sorted(branch_analysis_dir.glob("*.json"))
    # Skip _usage_summary.json
    json_files = [f for f in json_files if not f.name.startswith("_")]
    
    # Optionally filter by task_ids
    if args.task_ids:
        wanted_task_ids = {tid.strip() for tid in str(args.task_ids).split(",") if tid.strip()}
        filtered_files = []
        for f in json_files:
            try:
                with open(f, "r") as fp:
                    data = json.load(fp)
                if data.get("task_id") in wanted_task_ids:
                    filtered_files.append(f)
            except Exception as e:
                logger.error(f"Error reading {f} while filtering by task_ids: {e}", exc_info=True)
        json_files = filtered_files

    if args.limit_tasks:
        json_files = json_files[:args.limit_tasks]
    
    logger.info(f"Found {len(json_files)} branch analysis files")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect already generated outputs by (task_id, branch_after_step)
    existing_outputs = load_existing_task_ids_from_output(output_dir)
    if existing_outputs:
        logger.info(f"Detected {len(existing_outputs)} unique (task_id, branch_point) combinations with existing outputs")
        logger.info(f"Total existing output directories: {sum(existing_outputs.values())}")
    
    # Build all branch-point tasks across files, checking if we need to catch up
    all_tasks: List[Tuple[Dict[str, Any], Dict[str, Any], int]] = []  # (branch_analysis, branch_point, remaining_count)
    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                branch_analysis = json.load(f)
            task_id_in_file = branch_analysis.get("task_id")
            if not isinstance(task_id_in_file, str):
                logger.warning(f"Skipping {json_file}: no valid task_id")
                continue

            # If this task_id is already marked as processed, skip it entirely
            if processed_task_ids and task_id_in_file in processed_task_ids:
                logger.info(f"Task {task_id_in_file} is in processed_task_ids; skipping all its branch points from {json_file}")
                continue
            
            branch_points = branch_analysis.get("analysis", {}).get("branch_points", [])
            if args.limit_branches_per_task:
                branch_points = branch_points[-args.limit_branches_per_task:]
            
            for branch_point in branch_points:
                # Determine expected number of tasks for this branch point
                num_tasks_expected = branch_point.get("num_tasks", args.num_new_tasks)
                try:
                    num_tasks_expected = int(num_tasks_expected)
                except Exception:
                    num_tasks_expected = args.num_new_tasks
                
                # Check how many outputs already exist for this (task_id, branch_after_step)
                branch_after_step = branch_point.get("after_step")
                if not isinstance(branch_after_step, int):
                    logger.warning(f"Skipping branch point in {json_file}: invalid after_step")
                    continue
                
                key = (task_id_in_file, branch_after_step)
                existing_count = existing_outputs.get(key, 0)
                
                # Calculate how many more tasks we need to generate
                remaining_count = num_tasks_expected - existing_count
                
                if remaining_count > 0:
                    logger.info(f"Task {task_id_in_file} branch {branch_after_step}: {existing_count}/{num_tasks_expected} exist, generating {remaining_count} more")
                    all_tasks.append((branch_analysis, branch_point, remaining_count))
                else:
                    logger.info(f"Task {task_id_in_file} branch {branch_after_step}: {existing_count}/{num_tasks_expected} exist, skipping (complete)")
                    
        except Exception as e:
            logger.error(f"Error preparing tasks from {json_file}: {e}", exc_info=True)
            continue
    
    total_branch_points = len(all_tasks)
    total_successful = 0
    
    if total_branch_points == 0:
        logger.info("No branch points found to process.")
    elif args.num_envs > 1:
        logger.info(f"Running {total_branch_points} branch-point tasks with {args.num_envs} parallel processes")
        with Manager() as manager:
            task_queue = manager.Queue()
            result_list = manager.list()
            seen_tasks_store = manager.dict()
            seen_tasks_lock = manager.Lock()
            # Preload any previously proposed tasks from disk so we avoid regenerating them.
            seen_tasks_path = output_dir / "proposed_tasks.json"
            load_seen_tasks_from_disk(seen_tasks_path, seen_tasks_store)
            # enqueue tasks
            for item in all_tasks:
                task_queue.put(item)
            # start workers
            processes: List[Process] = []
            for i in range(args.num_envs):
                p = Process(
                    target=_worker_process,
                    args=(task_queue, result_list, seen_tasks_store, seen_tasks_lock, args),
                    name=f"Worker-{i+1}",
                )
                p.start()
                processes.append(p)
                logger.info(f"Started process {p.name} with PID {p.pid}")
            # wait for workers to finish
            for p in processes:
                p.join()
            # aggregate results
            total_successful = sum(int(x) for x in list(result_list))
            logger.info("All processes completed.")
            # Persist the union of all seen tasks from this run
            save_seen_tasks_to_disk(seen_tasks_path, seen_tasks_store)
    else:
        logger.info("Running in single-process mode")
        # Initialize GPT-5 client for single-process mode
        try:
            initialize_gpt5_client()
        except Exception as e:
            logger.error(f"Failed to initialize GPT-5 client: {e}")
            return
        # In single-process mode, maintain a local seen-tasks mapping.
        seen_tasks_path = output_dir / "proposed_tasks.json"
        seen_tasks_store: Dict[str, str] = {}
        load_seen_tasks_from_disk(seen_tasks_path, seen_tasks_store)
        for idx, (branch_analysis, branch_point, remaining_count) in enumerate(all_tasks, 1):
            logger.info(f"\nBranch point {idx}/{total_branch_points} (after_step={branch_point.get('after_step')}, generating {remaining_count} tasks)")
            succ = process_branch_point(
                branch_analysis,
                branch_point,
                args,
                remaining_count,
                seen_tasks_store=seen_tasks_store,
                seen_tasks_lock=None,
            )
            total_successful += succ
        # Persist any newly seen tasks from this single-process run
        save_seen_tasks_to_disk(seen_tasks_path, seen_tasks_store)
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total branch analysis files scanned: {len(json_files)}")
    logger.info(f"Total branch points processed: {total_branch_points}")
    logger.info(f"Total successful trajectories generated: {total_successful}")
    logger.info(f"Output directory: {output_dir}")
    
    # Save summary
    summary = {
        "generation_timestamp": datetime.datetime.now().isoformat(),
        "total_files": len(json_files),
        "total_branch_points": total_branch_points,
        "total_successful": total_successful,
        "args": vars(args)
    }
    
    with open(output_dir / "generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

