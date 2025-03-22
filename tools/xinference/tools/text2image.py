from typing import Any, Generator

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from utils import get_common_params, handle_api_error


class TextToImageTool(Tool):

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        try:
            params = get_common_params(tool_parameters)
            base_url = self.runtime.credentials["base_url"]
            api_key = self.runtime.credentials["api_key"]
            headers = {}
            if not api_key:
                headers["Authorization"] = f"Bearer {self.runtime.credentials['api_key']}"
            response = requests.post(
                f"{base_url}/v1/images/generations",
                headers=headers,
                json=params,
            )
            if not response.ok:
                raise Exception(handle_api_error(response))
            return response.json()
        except ValueError as e:
            yield self.create_text_message(str(e))
        except Exception as e:
            yield self.create_text_message(f"An error occurred: {str(e)}")
