from base64 import b64decode
from typing import Any, Generator

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage


def __handle_api_error__(response: requests.Response) -> str:
    try:
        error_data = response.json()
        return error_data.get("error", {}).get("message", f"Error: {response.status_code}")
    except requests.JSONDecodeError:
        return f"Error: {response.status_code} - {response.text[:200]}"


class TextToImageTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        try:
            default_negative_prompt = tool_parameters.get("negative_prompt", "")
            params = {
                "model": tool_parameters.get("model").strip(),
                "prompt": tool_parameters.get("prompt", "").strip(),
                "n": int(tool_parameters.get("n", 1)),
                "size": tool_parameters.get("size", "1024x1024"),
                "guidance": float(tool_parameters.get("guidance", 3.5)),
                "seed": tool_parameters.get("seed"),
                "negative_prompt": tool_parameters.get("negative_prompt", default_negative_prompt).strip(),
                "sample_method": tool_parameters.get("sample_method", "euler"),
                "sampling_steps": int(tool_parameters.get("sampling_steps", 20)),
                "schedule_method": tool_parameters.get("schedule_method", "discrete"),
                "response_format": "b64_json",
                "cfg_scale": float(tool_parameters.get("cfg_scale", 4.5)),
            }

            # 指定base64，否则返回的是路径
            base_url = self.runtime.credentials["base_url"]
            api_key = self.runtime.credentials["api_key"]
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {self.runtime.credentials['api_key']}"
            response = requests.post(
                f"{base_url}/v1/images/generations",
                headers=headers,
                json=params,
            )
            if not response.ok:
                raise Exception(__handle_api_error__(response))
            for image_data in response.json().get("data", []):
                if image_data.get("b64_json"):
                    yield self.create_blob_message(
                        blob=b64decode(image_data["b64_json"]),
                        meta={"mime_type": "image/png"}
                    )
        except ValueError as e:
            yield self.create_text_message(str(e))
        except Exception as e:
            yield self.create_text_message(f"An error occurred: {str(e)}")
