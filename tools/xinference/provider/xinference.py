from typing import Any

import requests
from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError


class XinferenceProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        base_url = credentials.get("base_url", "")
        api_key = credentials.get("api_key", "")

        if not base_url:
            raise ToolProviderCredentialValidationError("Xinference base_url is required")
        headers = {
            "accept": "application/json",
        }
        if not api_key:
            headers["authorization"] = f"Bearer {api_key}"

        response = requests.get(f"{base_url}/v1/models", headers=headers)
        if response.status_code != 200:
            raise ToolProviderCredentialValidationError(
                f"Failed to validate Xinference API key, status code: {response.status_code}-{response.text}"
            )
