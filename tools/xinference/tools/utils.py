from typing import Any
import requests
from typing import Any

import requests


def get_common_params(tool_parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": tool_parameters.get("model").strip(),
        "prompt": tool_parameters.get("prompt", "").strip(),
        "n": int(tool_parameters.get("n", 1)),
        "size": tool_parameters.get("size", "1024x1024"),
        "guidance": float(tool_parameters.get("guidance", 3.5)),
        "seed": tool_parameters.get("seed"),
        "negative_prompt": tool_parameters.get("negative_prompt", "").strip(),
        "sample_method": tool_parameters.get("sample_method", "euler"),
        "sampling_steps": int(tool_parameters.get("sampling_steps", 20)),
        "schedule_method": tool_parameters.get("schedule_method", "discrete"),
        # "cfg_scale": float(tool_parameters.get("cfg_scale", 4.5)),
    }


def handle_api_error(response: requests.Response) -> str:
    try:
        error_data = response.json()
        return error_data.get("error", {}).get("message", f"Error: {response.status_code}")
    except requests.JSONDecodeError:
        return f"Error: {response.status_code} - {response.text[:200]}"
