import argparse
import json
import os
import random
from pathlib import Path

import torch
from accelerate import Accelerator
import transformers
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

from PIL import Image
import re
from train_utils import create_branch_generated_dataset, create_dataset_from_hf

random.seed(123937)

# suggested deepspeed config (same as Qwen/LLaMA scripts)
DS_CONFIG_DICT = {
    "zero_optimization": {
        "stage": 3,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True,
        "round_robin_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        # Enable ZeRO-3 parameter CPU offloading to reduce GPU memory usage
        # while keeping the optimizer on GPU.
        # "offload_param": {
        #     "device": "cpu",
        #     "pin_memory": True,
        # },
    },
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "bf16": {"enabled": "auto"},
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
}


def create_model(model_name_or_path, use_flash_attention=False, cache_dir=None):
    """
    Create a GLM-4.1V model for training.

    Args:
        model_name_or_path: Path to the pretrained model or HuggingFace model ID
            (e.g., zai-org/GLM-4.1V-9B-Base)
        use_flash_attention: Whether to use Flash Attention 2
        cache_dir: Directory to cache downloaded models
    """

    # First, try the standard AutoModelForCausalLM path. On sufficiently
    # recent versions of `transformers`, GLM-4.1V will be wired into this
    # auto-mapping and this will Just Work.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if use_flash_attention else None,
        )
        return model
    except ValueError as e:
        # On some cluster installs, GLM-4.1V's config (`Glm4vConfig`) may be
        # present but not yet registered with AutoModelForCausalLM, which
        # triggers a ValueError like the one you saw. In that case, fall back
        # to locating the dedicated glm4v model class directly.
        if "Glm4vConfig" not in str(e):
            raise

        # Best-effort fallback: introspect `transformers.models.glm4v` to find
        # a *ForCausalLM / *ForConditionalGeneration style class and use it.
        try:
            glm4v_mod = transformers.models.glm4v.modeling_glm4v  # type: ignore[attr-defined]
        except Exception as inner_exc:
            raise RuntimeError(
                "Your installed `transformers` appears to know about Glm4vConfig "
                "but does not expose a glm4v modeling module compatible with "
                "AutoModelForCausalLM. Consider upgrading `transformers` to a "
                "newer version."
            ) from inner_exc

        candidate_cls = None
        for name in dir(glm4v_mod):
            lower = name.lower()
            if lower.startswith("glm4vfor") and (
                "causallm" in lower or "conditionalgeneration" in lower
            ):
                cls = getattr(glm4v_mod, name)
                if hasattr(cls, "from_pretrained"):
                    candidate_cls = cls
                    break

        if candidate_cls is None:
            raise RuntimeError(
                "Could not locate a GLM-4.1V generation class (e.g., Glm4vForCausalLM) "
                "inside `transformers.models.glm4v`. Please upgrade `transformers`."
            ) from e

        model = candidate_cls.from_pretrained(  # type: ignore[call-arg]
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            attn_implementation="flash_attention_2" if use_flash_attention else None,
        )
        return model


def build_system_prompt(coordinate_type="relative", processed_width=1000, processed_height=1000):
    """
    Build the system prompt exactly as qwen3vl_agent.py does, including the tools_def JSON.
    """
    description_prompt_lines = [
        "Use a mouse and keyboard to interact with a computer, and take screenshots.",
        "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
        "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.",
        (
            f"* The screen's resolution is {processed_width}x{processed_height}."
            if coordinate_type == "absolute"
            else "* The screen's resolution is 1000x1000."
        ),
        "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
        "* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
        "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
    ]
    description_prompt = "\n".join(description_prompt_lines)

    action_description_prompt = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor from a specified start (x, y) coordinate to a target (x, y) coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action).
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question.
        """

    tools_def = {
        "type": "function",
        "function": {
            "name_for_human": "computer_use",
            "name": "computer_use",
            "description": description_prompt,
            "parameters": {
                "properties": {
                    "action": {
                        "description": action_description_prompt,
                        "enum": [
                            "key",
                            "type",
                            "mouse_move",
                            "left_click",
                            "left_click_drag",
                            "right_click",
                            "middle_click",
                            "double_click",
                            "scroll",
                            "wait",
                            "terminate",
                        ],
                        "type": "string",
                    },
                    "keys": {"description": "Required only by `action=key`.", "type": "array"},
                    "text": {"description": "Required only by `action=type`.", "type": "string"},
                    "coordinate": {"description": "The x,y target coordinates for mouse actions.", "type": "array"},
                    "start_coordinate": {"description": "The x,y starting coordinates for drag actions.", "type": "array"},
                    "pixels": {"description": "The amount of scrolling.", "type": "number"},
                    "time": {"description": "The seconds to wait.", "type": "number"},
                    "status": {
                        "description": "The status of the task.",
                        "type": "string",
                        "enum": ["success", "failure"],
                    },
                },
                "required": ["action"],
                "type": "object",
            },
            "args_format": "Format the arguments as a JSON object.",
        },
    }

    system_prompt = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
""" + json.dumps(tools_def) + """
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Reasoning: a short reasoning describe the thinking and the action to take in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Reasoning, <tool_call>.
- Be brief: one sentence for Reasoning.
- Do not output anything else outside those parts.
- If finishing, use action=terminate in the tool call."""

    return system_prompt


class BranchGeneratedGLMCollator:
    """
    Data collator for training GLM-4.1V models on branch-generated
    trajectories, one target step per example, using multi-step, multi-image chat histories.

    Expects each dataset example to have the following fields (as created by
    `create_branch_generated_dataset`):
        - task_description: overall task description for the branch
        - branch_dir: absolute path to the branch directory
        - history: concatenated reasoning strings from all previous steps in this branch
        - step: a dict describing the current step, with:
            - step: integer step id (1-based)
            - is_replay: whether this is a replay step
            - reasoning: optional natural language reasoning
            - action_proposal: optional short imperative description of the action
            - action_dict: high-level action dictionary
        - all_steps: list of dicts for all steps in this branch (same schema as `step`)
        - current_step_idx: index into `all_steps` for the current step
    """

    def __init__(self, args, processor, max_steps=1, dual_training_types=True):
        self.max_steps = max_steps
        self.processor = processor
        self.args = args
        self.dual_training_types = dual_training_types  # Enable both training types
        # Counter to control how often we print input/output for debugging
        self._call_count = 0

    def _build_tool_call_from_action_dict(self, step):
        """
        Convert a high-level action_dict from the dataset into a tool-call JSON
        of the form:
            {"name": "computer_use", "arguments": {...}}

        Note: Training data coordinates are in 1280x720 pixel space. At runtime,
        the agent with coordinate_type="relative" expects coordinates on a
        0..999 grid in both x and y, which it then scales to the actual screen
        size. Here we convert 1280x720 pixel coordinates into this 0..999
        relative grid so that training and inference use the same convention.
        """
        action_dict = step.get("action_dict") or {}

        # Terminal / DONE steps
        if action_dict.get("terminal") is True:
            status_raw = str(action_dict.get("status", "")).lower()
            if status_raw in ("done", "success"):
                status = "success"
            elif status_raw in ("fail", "failure"):
                status = "failure"
            else:
                status = "success"
            return {
                "name": "computer_use",
                "arguments": {
                    "action": "terminate",
                    "status": status,
                },
            }

        input_dict = action_dict.get("input") or {}
        raw_action_type = input_dict.get("action")
        if not raw_action_type:
            return None

        # Map triple_click → double_click
        if raw_action_type == "triple_click":
            action_type = "double_click"
        else:
            action_type = raw_action_type

        arguments = {"action": action_type}

        # Helper function to scale coordinates from 1280x720 absolute pixels
        # into a 0..999 relative integer grid.
        def scale_coordinate_to_relative(coord):
            """
            Scale coordinate from 1280x720 pixel space into 0..999 relative
            integer space.
            """
            if isinstance(coord, (list, tuple)) and len(coord) == 2:
                base_w, base_h = 1280.0, 720.0
                try:
                    x = float(coord[0])
                    y = float(coord[1])
                except Exception:
                    return coord

                # Clamp to the original screen bounds
                x = max(0.0, min(base_w, x))
                y = max(0.0, min(base_h, y))

                # Scale to 0..999 and round to nearest integer
                x_rel = x / base_w * 999.0
                y_rel = y / base_h * 999.0

                x_int = int(round(x_rel))
                y_int = int(round(y_rel))

                # Ensure final coordinates are in-bounds 0..999
                x_int = max(0, min(999, x_int))
                y_int = max(0, min(999, y_int))

                return [x_int, y_int]

            return coord

        # Mouse-based actions with coordinates
        if action_type in (
            "left_click",
            "right_click",
            "middle_click",
            "double_click",
            "mouse_move",
            "left_click_drag",
        ):
            if action_type == "left_click_drag":
                start_coord = input_dict.get("start_coordinate")
                end_coord = input_dict.get("coordinate")

                if isinstance(start_coord, (list, tuple)) and len(start_coord) == 2:
                    arguments["start_coordinate"] = scale_coordinate_to_relative(start_coord)

                # If no explicit end_coord is provided, fall back to start_coord
                if isinstance(end_coord, (list, tuple)) and len(end_coord) == 2:
                    arguments["coordinate"] = scale_coordinate_to_relative(end_coord)
                else:
                    print("No end coordinate provided!!!!!!!!!")
            else:
                coord = input_dict.get("coordinate")
                if coord is None:
                    # Some trajectories may only store start_coordinate for
                    # certain mouse actions; fall back to that if present.
                    coord = input_dict.get("start_coordinate")
                if isinstance(coord, (list, tuple)) and len(coord) == 2:
                    arguments["coordinate"] = scale_coordinate_to_relative(coord)

            if action_type == "left_click_drag":
                duration = input_dict.get("duration", 0.5)
                if duration is None:
                    duration = 0.5
                try:
                    arguments["duration"] = float(duration)
                except Exception:
                    arguments["duration"] = 0.5

        elif action_type == "type":
            text = input_dict.get("text", "")
            arguments["text"] = str(text)

        elif action_type == "key":
            keys = input_dict.get("keys")
            if not keys:
                # Many trajectories encode key combos as a single string like "ctrl+c"
                key_text = input_dict.get("text", "")
                if isinstance(key_text, str) and key_text:
                    keys = [k.strip() for k in key_text.split("+") if k.strip()]
            if isinstance(keys, str):
                keys = [keys]
            keys = keys or []
            arguments["keys"] = [str(k) for k in keys]

        elif action_type == "scroll":
            amount = input_dict.get("scroll_amount")
            if amount is None:
                amount = input_dict.get("pixels", 0)
            try:
                amount = int(amount)
            except Exception:
                amount = 0
            direction = str(input_dict.get("scroll_direction", "")).lower()
            if direction == "down":
                amount = -abs(amount)
            elif direction == "up":
                amount = abs(amount)
            arguments["pixels"] = amount

        elif action_type == "wait":
            time_val = input_dict.get("time", None)
            if time_val is not None:
                try:
                    arguments["time"] = float(time_val)
                except Exception:
                    pass

        return {
            "name": "computer_use",
            "arguments": arguments,
        }

    def __call__(self, data):
        # Increment call counter for optional debug printing
        self._call_count += 1

        assert len(data) == 1, f"BranchGeneratedGLMCollator only supports batch_size == 1, got {len(data)}"
        example = data[0]

        overall_task = example["task_description"]
        all_steps = example.get("all_steps", None)
        current_step_idx = example.get("current_step_idx", None)

        # For HuggingFace datasets, lazily reconstruct all_steps from the HF source dataset
        # so max_past_screenshots continues to work without storing all images in memory.
        hf_indices = example.get("_hf_dataset_indices", None)
        if hf_indices is not None and all_steps is None:
            hf_source = getattr(self, "_hf_source_dataset", None)
            if hf_source is not None:
                all_steps = []
                for hf_idx in hf_indices:
                    hf_ex = hf_source[hf_idx]
                    action_dict_str = hf_ex.get("action_dict", "{}")
                    try:
                        action_dict = json.loads(action_dict_str) if isinstance(action_dict_str, str) else action_dict_str
                    except json.JSONDecodeError:
                        action_dict = {}

                    all_steps.append(
                        {
                            "step": hf_ex["step_number"],
                            "is_replay": False,
                            "reasoning": hf_ex.get("reasoning", ""),
                            "action_proposal": hf_ex.get("action_proposal", ""),
                            "action_dict": action_dict,
                            "reward": hf_ex.get("reward", 0),
                            "done": hf_ex.get("done", False),
                            # store HF index for lazy image loading
                            "_hf_idx": hf_idx,
                        }
                    )

        # Build system prompt (same content as Qwen/LLaMA agents)
        system_prompt_text = build_system_prompt(
            coordinate_type="relative",
            processed_width=1920,
            processed_height=1080,
        )

        system_message = {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt_text},
            ],
        }

        branch_dir = example["branch_dir"]

        # Helper to load a screenshot for a given step entry.
        # To predict a step's action, we need the screenshot from the PREVIOUS step,
        # since it shows the state AFTER the previous action was executed.
        def load_step_image(step_entry):
            # Priority 1: HF dataset lazy image loading
            hf_idx = step_entry.get("_hf_idx")
            if hf_idx is not None:
                hf_source = getattr(self, "_hf_source_dataset", None)
                if hf_source is not None:
                    img = hf_source[hf_idx].get("image")
                    if img is not None:
                        return img, f"hf_idx_{hf_idx}"

            # Priority 2: embedded image directly on step_entry
            embedded = step_entry.get("image")
            if embedded is not None:
                return embedded, "embedded"

            # Priority 3: explicit image_path (human/AgentNet style)
            image_path = step_entry.get("image_path")
            if image_path:
                image_local = Image.open(image_path)
                return image_local, image_path

            # If we are in HF mode (branch_dir is empty) and we couldn't get an image,
            # do NOT fall back to local screenshots.
            if not branch_dir:
                raise FileNotFoundError(
                    "HF example is missing image access (no _hf_source_dataset / _hf_idx)."
                )

            step_id_local = step_entry.get("step")
            is_replay_local = step_entry.get("is_replay", False)

            if is_replay_local:
                # For replay step N, use screenshot from replay step N-1
                prev_step_id = step_id_local - 1
                img_path_local = os.path.join(
                    branch_dir, "screenshots", f"step_{prev_step_id}_replay.png"
                )
            else:
                # For non-replay step N, use screenshot from step N-1.
                # If step N is 1, fall back to last replay step.
                prev_step_id = step_id_local - 1
                if prev_step_id < 1:
                    replay_screenshots = []
                    screenshots_dir_local = os.path.join(branch_dir, "screenshots")
                    if os.path.exists(screenshots_dir_local):
                        for filename in os.listdir(screenshots_dir_local):
                            if filename.endswith("_replay.png"):
                                match = re.match(r"step_(\d+)_replay\.png", filename)
                                if match:
                                    replay_screenshots.append(int(match.group(1)))
                    if replay_screenshots:
                        last_replay_step = max(replay_screenshots)
                        img_path_local = os.path.join(
                            branch_dir, "screenshots", f"step_{last_replay_step}_replay.png"
                        )
                    else:
                        img_path_local = os.path.join(
                            branch_dir, "screenshots", "step_0_replay.png"
                        )
                else:
                    img_path_local = os.path.join(
                        branch_dir, "screenshots", f"step_{prev_step_id}.png"
                    )

            image_local = Image.open(img_path_local)
            # Use original 1920x1080 resolution without resizing
            return image_local, img_path_local

        messages = [system_message]
        images = []

        # Build multi-turn chat history if full trajectory information is available.
        use_full_history = all_steps is not None and current_step_idx is not None

        if use_full_history:
            max_past = getattr(self.args, "max_past_screenshots", None)
            if max_past is None:
                max_past = 0
            max_past = max(0, int(max_past))

            # Window of past steps ending just before the current step.
            start_idx = max(0, current_step_idx - max_past)

            for idx in range(start_idx, current_step_idx):
                prev_step = all_steps[idx]
                try:
                    prev_image, _ = load_step_image(prev_step)
                except FileNotFoundError:
                    # Skip steps without screenshots.
                    continue

                images.append(prev_image)

                if idx == start_idx:
                    # Instruction + previous actions string
                    prev_actions_for_this_step = []
                    for i in range(idx):
                        if i < len(all_steps):
                            prev_step_i = all_steps[i]
                            prev_desc_i = (
                                prev_step_i.get("reasoning", "")
                                or prev_step_i.get("action_proposal")
                            )
                            prev_actions_for_this_step.append(
                                f"Step {i+1} reasoning: {prev_desc_i}"
                            )
                    previous_actions_str = (
                        "\n".join(prev_actions_for_this_step)
                        if prev_actions_for_this_step
                        else "None"
                    )

                    instruction_prompt_hist = f"""
Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction: {overall_task}

Previous actions:
{previous_actions_str}"""

                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": instruction_prompt_hist},
                            ],
                        }
                    )
                else:
                    # Subsequent history steps: just the image
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                            ],
                        }
                    )

                # Assistant response for this past step
                reasoning = prev_step.get("reasoning", "") or prev_step.get(
                    "action_proposal", ""
                )
                tool_call_json = self._build_tool_call_from_action_dict(prev_step)

                if tool_call_json:
                    action_line = (
                        f"Step {idx+1} reasoning: {reasoning}"
                        if reasoning
                        else f"Step {idx+1} reasoning: Perform action"
                    )
                    tool_call_text = (
                        "<tool_call>\n"
                        + json.dumps(tool_call_json, ensure_ascii=False)
                        + "\n</tool_call>"
                    )
                    assistant_text = f"{action_line}\n{tool_call_text}"
                else:
                    assistant_text = (
                        f"Step {idx+1} reasoning: {reasoning}"
                        if reasoning
                        else f"Step {idx+1} reasoning: Continue"
                    )

                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": assistant_text},
                        ],
                    }
                )

        # Each dataset example corresponds to exactly one (current) step.
        step = example["step"]

        # Load the screenshot for the current step (previous state's image).
        try:
            current_image, image_path = load_step_image(step)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found for current step: {branch_dir}")

        images.append(current_image)

        # Build previous_actions_str for the current step
        previous_actions = []
        if use_full_history and current_step_idx is not None:
            for i in range(current_step_idx):
                if i < len(all_steps):
                    prev_step_i = all_steps[i]
                    prev_desc_i = (
                        prev_step_i.get("reasoning", "")
                        or prev_step_i.get("action_proposal", "")
                    )
                    previous_actions.append(f"Step {i+1} reasoning: {prev_desc_i}")
        previous_actions_str = (
            "\n".join(previous_actions) if previous_actions else "None"
        )

        # Instruction prompt for the current step
        instruction_prompt = f"""
Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction: {overall_task}

Previous actions:
{previous_actions_str}"""

        # Final user turn
        if use_full_history and current_step_idx is not None and current_step_idx > 0:
            # Instruction already in first history message → only image
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                    ],
                }
            )
        else:
            # No history or first step: include both image and instruction text
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": instruction_prompt},
                    ],
                }
            )

        # Training types:
        #   Type 1: predict (action_proposal + action)
        #   Type 2: given action_proposal, predict action only
        training_type = example.get("training_type", "type1")
        use_type_2 = training_type == "type2"

        # For Type 2, append action_proposal to the user input
        if use_type_2:
            action_text_in = step.get("reasoning", "") or step.get(
                "action_proposal", ""
            )
            if action_text_in:
                action_proposal_prompt = f"\nReasoning: {action_text_in}\n"
                messages[-1]["content"].append(
                    {"type": "text", "text": action_proposal_prompt}
                )

        # Let the GLM processor build the multimodal prompt
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        batch = self.processor(
            text=[prompt],
            images=[images],
            padding=True,
            return_tensors="pt",
        )

        # Start with prompt tokens; target labels are -100 for the prompt part
        input_ids_list = [batch["input_ids"]]
        labels_list = [
            torch.full(
                (1, batch["input_ids"].shape[1]),
                -100,
                dtype=torch.long,
            )
        ]

        # Build supervised target for this step
        action_text = step.get("reasoning", "") or step.get("action_proposal", "")
        tool_call = self._build_tool_call_from_action_dict(step)

        lines = []

        if use_type_2:
            # Type 2: Only output the tool_call (action_proposal already in input)
            if tool_call and isinstance(tool_call, dict):
                lines.append("<tool_call>")
                lines.append(json.dumps(tool_call, ensure_ascii=False))
                lines.append("</tool_call>")
        else:
            # Type 1: Reasoning + tool_call
            if action_text:
                lines.append(f"Reasoning: {action_text}")
            else:
                if tool_call and isinstance(tool_call, dict):
                    args = tool_call.get("arguments", {}) or {}
                    act = args.get("action", "unknown")
                    lines.append(f"Reasoning: Perform {act} action")
                else:
                    lines.append(
                        "Reasoning: Decide the next action based on the screenshot."
                    )

            if tool_call and isinstance(tool_call, dict):
                lines.append("<tool_call>")
                lines.append(json.dumps(tool_call, ensure_ascii=False))
                lines.append("</tool_call>")

        # Append the EOS token so the model learns when to stop generating.
        answer = "\n".join(lines) + self.processor.tokenizer.eos_token

        answer_input_ids = self.processor.tokenizer(
            answer,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]

        input_ids_list.append(answer_input_ids)
        labels_list.append(answer_input_ids)

        assert "pixel_values" in batch, f"Image not found: {image_path}!!!\n"

        input_ids = torch.cat(input_ids_list, dim=1)
        labels = torch.cat(labels_list, dim=1)

        attention_mask = torch.ones_like(input_ids)

        # Update batch with new text fields; keep any image-related keys as-is.
        batch["input_ids"] = input_ids
        batch["labels"] = labels
        batch["attention_mask"] = attention_mask

        # ------------------------------------------------------------------
        # Optional debug printing of input/output for a few batches.
        # ------------------------------------------------------------------
        if self._call_count <= 5 or self._call_count % 100 == 0:
            try:
                training_type_str = (
                    "Type 2 (with action_proposal)"
                    if use_type_2
                    else "Type 1 (predict action_proposal + action)"
                )

                step_id = step.get("step", "?")
                is_replay = step.get("is_replay", False)
                step_type = "Replay" if is_replay else "Regular"
                num_images_used = len(images)

                print("\n" + "╔" + "=" * 98 + "╗")
                print(
                    f"║  TRAINING EXAMPLE #{self._call_count:04d} - {training_type_str:^60s}  ║"
                )
                print("╠" + "=" * 98 + "╣")
                print(
                    f"║  Step: {step_id} ({step_type}) | Images: {num_images_used} | Task: {overall_task[:45]:45s}  ║"
                )
                print("╠" + "=" * 98 + "╣")

                print("║  MESSAGE STRUCTURE:")
                for i, msg in enumerate(messages):
                    role = msg["role"]
                    content_items = msg["content"]
                    content_types = [item["type"] for item in content_items]
                    print(f"║    [{i}] {role:10s}: {', '.join(content_types)}")
                print("╠" + "=" * 98 + "╣")

                print("║  MODEL INPUT PROMPT:")
                print("╠" + "-" * 98 + "╣")
                prompt_lines = prompt.split("\n")
                for line in prompt_lines[:100]:
                    if len(line) > 96:
                        print(f"║  {line[:93]}...")
                    else:
                        print(f"║  {line:96s}║")
                if len(prompt_lines) > 100:
                    print(
                        f"║  ... [{len(prompt_lines) - 100} more lines omitted] ..."
                    )

                print("╠" + "=" * 98 + "╣")

                print("║  MODEL TARGET OUTPUT:")
                print("╠" + "-" * 98 + "╣")
                answer_lines = answer.split("\n")
                for line in answer_lines:
                    if len(line) > 96:
                        print(f"║  {line[:93]}...")
                    else:
                        print(f"║  {line:96s}║")

                print("╠" + "=" * 98 + "╣")

                num_input_tokens = input_ids.shape[1]
                num_label_tokens = (labels[0] != -100).sum().item()
                print(
                    f"║  TOKENS: Input={num_input_tokens:5d} | Labels={num_label_tokens:5d} | Images={num_images_used:2d}"
                    + " " * 38
                    + "║"
                )

                print("╚" + "=" * 98 + "╝\n")

            except Exception as e:
                print(f"[Collator Debug] Failed to print input/output: {e}")
                import traceback

                traceback.print_exc()

        return batch


class SafeTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_token_counts = []  # Track tokens per GPU for current step

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training_step to handle OOM errors in a distributed-safe manner.

        With DeepSpeed ZeRO-3, all ranks must execute the same forward pass because
        parameters are partitioned. If one rank hits OOM and returns early while
        others continue, DeepSpeed will detect a rank disagreement and crash.

        Solution: Use torch.distributed to synchronize OOM status across all ranks.
        If ANY rank hits OOM, ALL ranks skip the step together.
        """
        import torch.distributed as dist

        # Track token usage
        if "input_ids" in inputs:
            num_input_tokens = inputs["input_ids"].shape[1]
            num_labels = (inputs["labels"] != -100).sum().item()
            self.step_token_counts.append(num_input_tokens)
        else:
            num_input_tokens = 0
            num_labels = 0

        # Flag to track if this rank hit OOM
        local_oom = torch.tensor(
            [0],
            device=inputs["input_ids"].device
            if "input_ids" in inputs
            else "cuda",
        )
        loss = None

        try:
            loss = super().training_step(model, inputs, num_items_in_batch)
        except RuntimeError as e:
            if "out of memory" in str(e):
                local_oom[0] = 1
                print(
                    f"[OOM ERROR] Rank {dist.get_rank() if dist.is_initialized() else 0} "
                    f"failed on {num_input_tokens} input tokens, {num_labels} label tokens. "
                    f"Step token history: {self.step_token_counts}"
                )
                torch.cuda.empty_cache()
            else:
                raise e

        # Synchronize OOM status across all ranks
        if dist.is_initialized():
            dist.all_reduce(local_oom, op=dist.ReduceOp.MAX)

        if local_oom[0] > 0:
            if dist.is_initialized() and dist.get_rank() == 0:
                print("[OOM SYNC] At least one rank hit OOM, all ranks skipping this step")
            self.step_token_counts = []
            torch.cuda.empty_cache()
            return torch.tensor(
                0.0,
                device=inputs["input_ids"].device
                if "input_ids" in inputs
                else "cuda",
                requires_grad=True,
            )

        return loss


def main():
    """
    Train GLM-4.1V-9B-Base on branch-generated trajectories.

    Example usage:
        torchrun --nproc_per_node=4 train_glm41v.py \\
            --model_name_or_path zai-org/GLM-4.1V-9B-Base \\
            --use_flash_attention \\
            --batch_size 16 \\
            --output_dir /path/to/output
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="zai-org/GLM-4.1V-9B-Base",
        help=(
            "Model name or path to load from "
            "(e.g., zai-org/GLM-4.1V-9B-Base or a local checkpoint directory)"
        ),
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default="/hf_cache",
        help="Directory to use for Hugging Face cache (models, tokenizers, etc.)",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Use Flash Attention",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use BF16 (default True for GLM-4.1V)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/glm-4.1v-9b-train",
        help="Output directory",
    )
    parser.add_argument(
        "--save-strategy",
        type=str,
        default="steps",
        help="Save strategy",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (global, across GPUs)",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--no-tqdm",
        dest="tqdm",
        action="store_false",
        help="Disable tqdm",
    )
    parser.add_argument(
        "--tensorboard-logging",
        action="store_true",
        help="log to tensorboard",
    )
    parser.add_argument(
        "--use-google-search",
        action="store_true",
        help="(unused here) keep parity with other scripts",
    )
    parser.add_argument(
        "--use-nogoto-gs-format",
        action="store_true",
        help="(unused here) keep parity with other scripts",
    )
    parser.add_argument(
        "--branch_generated_root",
        type=str,
        default="/branch_gen_winarena_filtered_done_or_verified",
        help=(
            "Root directory containing branch-generated trajectories "
            "(each subdirectory should contain metadata.json, trajectory.jsonl, and screenshots/). "
            "If provided, this will be used to build the training dataset."
        ),
    )
    parser.add_argument(
        "--branch_generated_half_verified_root",
        type=str,
        default="/branch_gen_winarena_half_verified",
        help=(
            "Optional root directory containing half-verified branch-generated trajectories. "
            "If provided, these branches will be added to the training dataset in addition to "
            "--branch_generated_root. For these tasks, only post-branch (non-replay) steps are "
            "used, and the task description is taken from `new_task_description`."
        ),
    )
    parser.add_argument(
        "--include_half_verified",
        action="store_true",
        default=False,
        help=(
            "If set, include trajectories from --branch_generated_half_verified_root "
            "in the training data. If not set, only data from --branch_generated_root "
            "is used."
        ),
    )
    parser.add_argument(
        "--use_hf_dataset",
        action="store_true",
        default=False,
        help=(
            "If set, train on a HuggingFace dataset (default: yale-nlp/Anchor) instead "
            "of local branch-generated trajectories."
        ),
    )
    parser.add_argument(
        "--hf_dataset_name",
        type=str,
        default="yale-nlp/Anchor",
        help="HuggingFace dataset name to load (default: yale-nlp/Anchor).",
    )
    parser.add_argument(
        "--hf_os_filter",
        type=str,
        default=None,
        choices=["windows", "ubuntu"],
        help="Optional OS filter when using HuggingFace dataset.",
    )
    parser.add_argument(
        "--max_past_screenshots",
        type=int,
        default=2,
        help=(
            "Maximum number of past screenshots (steps) to include in the prompt, "
            "in addition to the current step. Default is 2. "
            "Set to 0 to only use the current screenshot."
        ),
    )
    parser.add_argument(
        "--dual_training_types",
        action="store_true",
        default=True,
        help=(
            "Enable dual training types: "
            "Type 1: instruction + history → predict (action_proposal + action), "
            "Type 2: instruction + history + action_proposal → predict action only. "
            "When enabled, creates BOTH training examples for each step (doubles the dataset size). "
            "Default is True."
        ),
    )
    parser.add_argument(
        "--no_dual_training_types",
        dest="dual_training_types",
        action="store_false",
        help="Disable dual training types and only use Type 1 (original behavior).",
    )

    args = parser.parse_args()

    accelerator = Accelerator()

    # Ensure Hugging Face uses the requested cache directory.
    if args.hf_cache_dir:
        os.makedirs(args.hf_cache_dir, exist_ok=True)
        os.environ.setdefault("HF_HOME", args.hf_cache_dir)
        os.environ.setdefault("HF_HUB_CACHE", args.hf_cache_dir)

    with accelerator.local_main_process_first():
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            cache_dir=args.hf_cache_dir,
        )
        model = create_model(
            args.model_name_or_path,
            use_flash_attention=args.use_flash_attention,
            cache_dir=args.hf_cache_dir,
        )

    # Build training dataset
    if args.use_hf_dataset:
        print(f"Loading training data from HuggingFace dataset: {args.hf_dataset_name}")
        train_dataset = create_dataset_from_hf(
            hf_dataset_name=args.hf_dataset_name,
            os_filter=args.hf_os_filter,
            dual_training_types=args.dual_training_types,
            cache_dir=args.hf_cache_dir,
        )
    else:
        # Build training dataset from branch-generated trajectories
        if not args.branch_generated_root:
            raise ValueError(
                "You must provide --branch_generated_root pointing to the branch_generated directory."
            )
        if args.include_half_verified:
            half_verified_root = args.branch_generated_half_verified_root
        else:
            half_verified_root = None

        train_dataset = create_branch_generated_dataset(
            args.branch_generated_root,
            dual_training_types=args.dual_training_types,
            half_verified_root=half_verified_root,
        )

    # Preserve HF source dataset across shuffle (shuffle drops custom attributes)
    hf_source_dataset = getattr(train_dataset, "_hf_source_dataset", None)

    # Shuffle the dataset so Type 1 and Type 2 examples of the same step
    # are unlikely to be adjacent.
    train_dataset = train_dataset.shuffle(seed=42)

    if hf_source_dataset is not None:
        train_dataset._hf_source_dataset = hf_source_dataset

    print("train_dataset:", train_dataset)
    print("len(train_dataset):", len(train_dataset))
    print("Dataset shuffled to randomize Type 1 and Type 2 examples.")

    import time

    time.sleep(3)

    num_gpus = accelerator.num_processes
    print(f"training on {num_gpus} GPUs")
    assert (
        args.batch_size % num_gpus == 0
    ), "Batch size must be divisible by the number of GPUs"
    gradient_accumulation_steps = args.batch_size // num_gpus

    if args.bf16:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False

    training_args = TrainingArguments(
        ddp_find_unused_parameters=False,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,  # Branch collator assumes batch_size == 1
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=args.learning_rate,
        weight_decay=args.wd,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        warmup_steps=50,
        logging_steps=10,
        output_dir=args.output_dir,
        save_strategy=args.save_strategy,
        save_steps=400,
        save_total_limit=5 if args.save_strategy == "steps" else None,
        save_only_model=True,
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        report_to="tensorboard" if args.tensorboard_logging else "none",
        deepspeed=DS_CONFIG_DICT,
        disable_tqdm=not args.tqdm,
        dataloader_num_workers=1,
        dataloader_prefetch_factor=1,
    )

    data_collator = BranchGeneratedGLMCollator(
        args,
        processor,
        dual_training_types=args.dual_training_types,
    )

    if hf_source_dataset is not None:
        data_collator._hf_source_dataset = hf_source_dataset

    # Save a "checkpoint-0" copy of the original (pre-finetune) model & processor
    out_path = Path(training_args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        pre_ft_ckpt_dir = out_path / "checkpoint-0"
        pre_ft_ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(pre_ft_ckpt_dir)
        processor.save_pretrained(pre_ft_ckpt_dir)
    accelerator.wait_for_everyone()

    trainer = SafeTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()

