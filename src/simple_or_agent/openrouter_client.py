"""Minimal OpenRouter chat client with retries and helpers."""

import os
import logging
from typing import Dict, Any, List, Optional

import requests
from tenacity import Retrying, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log, after_log

logger = logging.getLogger(__name__)
class OpenRouterError(Exception):
    """Base exception for OpenRouter client errors."""
    pass

class OpenRouterAPIError(OpenRouterError):
    """API-specific errors with status codes and details."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data

class OpenRouterClient:
    """Chat completions client with retry/backoff and basic helpers."""
    BASE_URL = "https://openrouter.ai/api/v1"
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_s: int = 30,
        max_retries: int = 3,
        retry_base_wait: int = 1,
        retry_max_wait: int = 60,
        app_name: Optional[str] = None,
        app_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise OpenRouterError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.")
        
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_base_wait = retry_base_wait
        self.retry_max_wait = retry_max_wait
        
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

        if app_url:
            self.session.headers["HTTP-Referer"] = app_url
        if app_name:
            self.session.headers["X-Title"] = app_name
        
        logger.info(f"Initialized OpenRouter client with {max_retries} max retries, {timeout_s}s timeout")

    def _post_once(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = self.session.post(url, json=data, timeout=self.timeout_s)

            # Handle rate limiting with Retry-After header
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    logger.warning(f"Rate limited. Retry after: {retry_after}s")
                raise OpenRouterAPIError(
                    "Rate limited (429)",
                    status_code=429,
                    response_data=response.json() if response.content else None,
                )

            # Handle server errors (5xx)
            if 500 <= response.status_code < 600:
                error_msg = f"Server error ({response.status_code})"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = f"{error_msg}: {error_data['error']}"
                except Exception:
                    pass
                raise OpenRouterAPIError(error_msg, status_code=response.status_code)

            # Handle client errors (4xx) - don't retry these
            if 400 <= response.status_code < 500:
                error_msg = f"Client error ({response.status_code})"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = f"{error_msg}: {error_data['error']}"
                except Exception:
                    error_msg = f"{error_msg}: {response.text}"

                raise OpenRouterError(error_msg)  # Don't retry client errors

            # Parse successful response
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            raise OpenRouterAPIError(f"Request timeout after {self.timeout_s}s")
        except requests.exceptions.ConnectionError as e:
            raise OpenRouterAPIError(f"Connection error: {e}")
        except requests.exceptions.RequestException as e:
            raise OpenRouterAPIError(f"Request error: {e}")

    def _make_request(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        retrying = Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                multiplier=self.retry_base_wait, min=self.retry_base_wait, max=self.retry_max_wait
            ),
            retry=retry_if_exception_type(
                (requests.exceptions.RequestException, OpenRouterAPIError)
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.INFO),
            reraise=True,
        )

        # Use iterator form to ensure compatibility across Tenacity versions
        for attempt in retrying:
            with attempt:
                return self._post_once(url, data)

    def complete_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        # Validate inputs
        if not messages:
            raise OpenRouterError("Messages list cannot be empty")
        
        if not model:
            raise OpenRouterError("Model identifier is required")
        
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise OpenRouterError(f"Message at index {i} must have 'role' and 'content' fields")
        
        if not (0.0 <= temperature <= 2.0):
            raise OpenRouterError("Temperature must be between 0.0 and 2.0")
        
        # Build request payload
        payload = {"model": model, "messages": messages, "temperature": temperature}
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        if response_format is not None:
            payload["response_format"] = response_format
        if reasoning is not None:
            payload["reasoning"] = reasoning
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = parallel_tool_calls
        
        # Add any additional parameters
        payload.update(kwargs)
        
        # Log request (without sensitive data)
        log_payload = {k: v for k, v in payload.items() if k != 'messages'}
        log_payload['message_count'] = len(messages)
        logger.info(f"Making OpenRouter request: {log_payload}")
        
        # Make request with retries
        url = f"{self.BASE_URL}/chat/completions"
        response_data = self._make_request(url, payload)
        
        # Log response summary
        if 'usage' in response_data:
            usage = response_data['usage']
            logger.info(f"Request completed. Tokens - prompt: {usage.get('prompt_tokens')}, "
                       f"completion: {usage.get('completion_tokens')}, total: {usage.get('total_tokens')}")
        
        return response_data

    def extract_content(self, response: Dict[str, Any]) -> str:
        """Return assistant text content, tolerating structured content formats."""
        try:
            choices = response.get('choices') or []
            if not choices:
                raise OpenRouterError("No choices in response")
            msg = choices[0].get('message') or {}
            content = msg.get('content')
            # Plain string
            if isinstance(content, str) and content.strip():
                return content
            # OpenAI-style list of content parts
            if isinstance(content, list):
                parts: List[str] = []
                for p in content:
                    if isinstance(p, str) and p.strip():
                        parts.append(p.strip()); continue
                    if isinstance(p, dict):
                        t = p.get('text')
                        if isinstance(t, dict):
                            t = t.get('value') or t.get('text')
                        if isinstance(t, str) and t.strip():
                            parts.append(t.strip()); continue
                        for k in ('content', 'value'):
                            v = p.get(k)
                            if isinstance(v, str) and v.strip():
                                parts.append(v.strip()); break
                if parts:
                    return "\n".join(parts)
            # Fallback: summarize tool calls if present
            tc = msg.get('tool_calls') or []
            if tc:
                import json as _json
                summ = [{
                    'id': (c.get('id') or '')[:12],
                    'name': ((c.get('function') or {}).get('name') or ''),
                } for c in tc]
                return _json.dumps({'note': 'no_text_content', 'tool_calls': summ}, ensure_ascii=False)
            raise OpenRouterError("Empty content in response")
        except (KeyError, IndexError, TypeError) as e:
            raise OpenRouterError(f"Invalid response format: {e}")

    def extract_reasoning(self, response: Dict[str, Any]) -> Optional[str]:
        """Return reasoning (string or JSON string) if present."""
        try:
            choices = response.get("choices", [])
            if not choices:
                return None
            message = choices[0].get("message", {})
            reasoning = message.get("reasoning")
            if reasoning is None:
                return None
            if isinstance(reasoning, str):
                return reasoning
            # Fallback: stringify structured reasoning
            import json as _json  # local import to avoid top-level cost
            return _json.dumps(reasoning, ensure_ascii=False)
        except Exception:
            return None

    def get_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return assistant tool_calls list if present, else []."""
        try:
            choices = response.get("choices", [])
            if not choices:
                return []
            message = choices[0].get("message", {})
            calls = message.get("tool_calls")
            return calls or []
        except Exception:
            return []

    @staticmethod
    def make_tool_result(tool_call_id: str, content: str) -> Dict[str, str]:
        """Build a tool result message for follow-up calls."""
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

    def close(self):
        """Close the HTTP session."""
        if self.session:
            self.session.close()
            logger.debug("Closed OpenRouter client session")

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()
