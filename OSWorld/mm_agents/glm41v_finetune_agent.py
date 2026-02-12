import base64
import json
import logging
import os
from io import BytesIO
from typing import Dict, List, Tuple

from PIL import Image
import backoff
import openai
from openai import OpenAI


logger = logging.getLogger("desktopenv.glm41v_finetune_agent")


def process_image(image_bytes, max_pixels=1920*1080):
    """
    Process an image for GLM-4.1V models.
    Keeps the original resolution if under max_pixels, otherwise resizes proportionally.
    """
    image = Image.open(BytesIO(image_bytes))
    width, height = image.size
    
    # Calculate current pixel count
    current_pixels = width * height
    
    # If within limit, keep original
    if current_pixels <= max_pixels:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        processed_bytes = buffer.getvalue()
        return base64.b64encode(processed_bytes).decode("utf-8")
    
    # Otherwise, resize proportionally
    scale_factor = (max_pixels / current_pixels) ** 0.5
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    processed_bytes = buffer.getvalue()
    
    return base64.b64encode(processed_bytes).decode("utf-8")


def _build_system_prompt(
    coordinate_type: str = "relative",
    processed_width: int = 1000,
    processed_height: int = 1000,
) -> str:
    """
    Build the system prompt to match the format used during finetuning in
    `Explorer/train/train_glm41v.py`.
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
                    "keys": {
                        "description": "Required only by `action=key`.",
                        "type": "array",
                    },
                    "text": {
                        "description": "Required only by `action=type`.",
                        "type": "string",
                    },
                    "coordinate": {
                        "description": "The x,y target coordinates for mouse actions.",
                        "type": "array",
                    },
                    "start_coordinate": {
                        "description": "The x,y starting coordinates for drag actions.",
                        "type": "array",
                    },
                    "pixels": {
                        "description": "The amount of scrolling.",
                        "type": "number",
                    },
                    "time": {
                        "description": "The seconds to wait.",
                        "type": "number",
                    },
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


class GLM41VFinetuneAgent:
    """
    Agent for finetuned GLM-4.1V models via vLLM OpenAI-compatible API.
    
    Uses the same prompt format as the training script (train_glm41v.py).
    """

    def __init__(
        self,
        platform: str = "ubuntu",
        model: str = "zai-org/GLM-4.1V-9B-Base",
        max_tokens: int = 4096,
        top_p: float = 0.9,
        temperature: float = 0.0,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        max_trajectory_length: int = 3,
        coordinate_type: str = "relative",
        add_thought_prefix: bool = False,
        api_base: str = "http://localhost:8000/v1",
    ):
        self.platform = platform
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.coordinate_type = coordinate_type
        self.add_thought_prefix = add_thought_prefix
        self.api_base = api_base

        assert action_space in ["pyautogui"], "Invalid action space"
        assert observation_type in ["screenshot"], "Invalid observation type"

        # Initialize OpenAI client pointing to vLLM server
        self.client = OpenAI(
            api_key="EMPTY",  # vLLM doesn't require a real API key
            base_url=self.api_base,
        )

        self.thoughts = []
        self.actions = []
        self.observations = []
        self.responses = []
        self.screenshots = []

        # Build the system prompt matching training format
        self.system_prompt = _build_system_prompt(
            coordinate_type=coordinate_type,
            processed_width=1920,
            processed_height=1080,
        )

    def reset(self):
        """Reset agent state for a new task."""
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.responses = []
        self.screenshots = []

    @backoff.on_exception(
        backoff.expo,
        (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        ),
        max_tries=5,
    )
    def predict(self, instruction: str, obs: Dict) -> Tuple[str, List[str]]:
        """
        Generate next action given current observation.
        
        Args:
            instruction: Task instruction
            obs: Observation dict containing screenshot bytes
            
        Returns:
            Tuple of (full_response, list_of_action_strings)
        """
        # Process screenshot
        screenshot_bytes = obs.get("screenshot", b"")
        if not screenshot_bytes:
            logger.error("No screenshot in observation")
            return "FAIL", "No screenshot available"
        
        screenshot_b64 = process_image(screenshot_bytes)
        self.screenshots.append(screenshot_b64)

        # Build message history
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

        # Build previous actions string
        previous_actions = []
        for i, (thought, action) in enumerate(zip(self.thoughts, self.actions)):
            step_num = i + 1
            reasoning_text = thought if thought else "Perform action"
            previous_actions.append(f"Step {step_num} reasoning: {reasoning_text}")
        
        previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"

        # Build user message with history
        # Include up to max_trajectory_length past screenshots
        start_idx = max(0, len(self.screenshots) - self.max_trajectory_length - 1)
        
        for idx in range(start_idx, len(self.screenshots)):
            screenshot_img_b64 = self.screenshots[idx]
            
            if idx == start_idx:
                # First message includes instruction and previous actions
                instruction_prompt = f"""Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction: {instruction}

Previous actions:
{previous_actions_str}"""
                
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot_img_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": instruction_prompt
                        }
                    ]
                })
            elif idx < len(self.screenshots) - 1:
                # Past actions with screenshots and responses
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot_img_b64}"
                            }
                        }
                    ]
                })
                # Add corresponding assistant response
                hist_idx = idx - start_idx
                if hist_idx < len(self.responses):
                    messages.append({
                        "role": "assistant",
                        "content": self.responses[hist_idx]
                    })
            else:
                # Current screenshot
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot_img_b64}"
                            }
                        }
                    ]
                })

        # Call vLLM API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            
            response_text = response.choices[0].message.content
            self.responses.append(response_text)
            
            logger.info(f"Model response: {response_text}")
            
            # Parse the response to extract action
            action = self._parse_response(response_text, obs)
            
            # Return (response, [action]) - response first, action as list
            return response_text, [action]
            
        except Exception as e:
            logger.error(f"Error calling vLLM API: {e}")
            return str(e), ["FAIL"]

    def _parse_response(self, response_text: str, obs: Dict) -> str:
        """
        Parse the model's response to extract the action.
        
        Expected format:
        Reasoning: <text>
        <tool_call>
        {"name": "computer_use", "arguments": {...}}
        </tool_call>
        """
        try:
            # Extract reasoning
            reasoning = ""
            if "Reasoning:" in response_text:
                reasoning_part = response_text.split("<tool_call>")[0]
                reasoning = reasoning_part.split("Reasoning:", 1)[1].strip()
            
            self.thoughts.append(reasoning)
            
            # Extract tool call JSON
            if "<tool_call>" in response_text and "</tool_call>" in response_text:
                tool_call_text = response_text.split("<tool_call>")[1].split("</tool_call>")[0].strip()
                tool_call = json.loads(tool_call_text)
                
                arguments = tool_call.get("arguments", {})
                action_type = arguments.get("action", "")
                
                # Convert to action string
                action = self._convert_to_action_string(arguments, obs)
                self.actions.append(reasoning)
                
                return action
            else:
                logger.error(f"No tool_call found in response: {response_text}")
                self.actions.append(reasoning)
                return "FAIL"
                
        except Exception as e:
            logger.error(f"Error parsing response: {e}\nResponse: {response_text}")
            self.thoughts.append("")
            self.actions.append("")
            return "FAIL"

    def _convert_to_action_string(self, arguments: Dict, obs: Dict) -> str:
        """
        Convert action arguments to PyAutoGUI action string.
        
        Coordinates are in relative 0-999 space and need to be scaled to actual screen size.
        """
        action_type = arguments.get("action", "")
        
        if action_type == "terminate":
            status = arguments.get("status", "success")
            return f"DONE" if status == "success" else "FAIL"
        
        # Get screen dimensions
        screenshot_bytes = obs.get("screenshot", b"")
        if screenshot_bytes:
            image = Image.open(BytesIO(screenshot_bytes))
            screen_width, screen_height = image.size
        else:
            screen_width, screen_height = 1920, 1080
        
        def scale_coordinate(coord):
            """Scale from 0-999 relative space to actual screen pixels."""
            if isinstance(coord, (list, tuple)) and len(coord) == 2:
                x_rel, y_rel = coord
                x = int(x_rel / 999.0 * screen_width)
                y = int(y_rel / 999.0 * screen_height)
                # Clamp to screen bounds
                x = max(0, min(screen_width - 1, x))
                y = max(0, min(screen_height - 1, y))
                return x, y
            return coord
        
        # Build action string based on action type
        if action_type == "left_click":
            coord = arguments.get("coordinate", [])
            if coord:
                x, y = scale_coordinate(coord)
                return f"pyautogui.click({x}, {y})"
            return "FAIL"
        
        elif action_type == "right_click":
            coord = arguments.get("coordinate", [])
            if coord:
                x, y = scale_coordinate(coord)
                return f"pyautogui.rightClick({x}, {y})"
            return "FAIL"
        
        elif action_type == "double_click":
            coord = arguments.get("coordinate", [])
            if coord:
                x, y = scale_coordinate(coord)
                return f"pyautogui.doubleClick({x}, {y})"
            return "FAIL"
        
        elif action_type == "middle_click":
            coord = arguments.get("coordinate", [])
            if coord:
                x, y = scale_coordinate(coord)
                return f"pyautogui.middleClick({x}, {y})"
            return "FAIL"
        
        elif action_type == "mouse_move":
            coord = arguments.get("coordinate", [])
            if coord:
                x, y = scale_coordinate(coord)
                return f"pyautogui.moveTo({x}, {y})"
            return "FAIL"
        
        elif action_type == "left_click_drag":
            start_coord = arguments.get("start_coordinate", [])
            end_coord = arguments.get("coordinate", [])
            duration = arguments.get("duration", 0.5)
            if start_coord and end_coord:
                x1, y1 = scale_coordinate(start_coord)
                x2, y2 = scale_coordinate(end_coord)
                return f"pyautogui.click({x1}, {y1})\npyautogui.drag({x2-x1}, {y2-y1}, {duration})"
            return "FAIL"
        
        elif action_type == "type":
            text = arguments.get("text", "")
            # Escape special characters for Python string
            text_escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
            return f'pyautogui.write("""{text}""", interval=0.01)'
        
        elif action_type == "key":
            keys = arguments.get("keys", [])
            if keys:
                # Convert to pyautogui.hotkey format
                keys_str = ", ".join([f"'{k}'" for k in keys])
                return f"pyautogui.hotkey({keys_str})"
            return "FAIL"
        
        elif action_type == "scroll":
            pixels = arguments.get("pixels", 0)
            # PyAutoGUI scroll is positive for up, negative for down
            # Our convention: negative is down, positive is up (same as PyAutoGUI)
            return f"pyautogui.scroll({pixels})"
        
        elif action_type == "wait":
            time_val = arguments.get("time", 1.0)
            return f"time.sleep({time_val})"
        
        else:
            logger.error(f"Unknown action type: {action_type}")
            return "FAIL"

