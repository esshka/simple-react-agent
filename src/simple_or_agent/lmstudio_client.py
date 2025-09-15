# src/simple_or_agent/lmstudio_client.py
# Minimal LM Studio chat client for OpenAI-compatible local servers.
# This exists to call a LAN LM Studio server with simple retries and helpers.
# RELEVANT FILES: src/simple_or_agent/openrouter_client.py,src/simple_or_agent/simple_agent.py,src/simple_or_agent/react_agent.py

"""Minimal LM Studio chat client with retries and helpers."""

import os, logging, requests
from typing import Dict, Any, List, Optional
from tenacity import Retrying, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log, after_log

logger = logging.getLogger(__name__)

class LMStudioError(Exception):
    """Base exception for LM Studio client errors."""

class LMStudioAPIError(LMStudioError):
    """API-specific errors with status codes and details."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict[str, Any]] = None):
        super().__init__(message); self.status_code = status_code; self.response_data = response_data

class LMStudioClient:
    """Chat completions client with retry/backoff and basic helpers."""
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, timeout_s: int = 30, max_retries: int = 3, retry_base_wait: int = 1, retry_max_wait: int = 30):
        # Resolve base URL; ensure it ends with /v1
        url = (base_url or os.getenv("LMSTUDIO_BASE_URL") or "http://192.168.1.157:1234").rstrip("/")
        self.base_url = url if url.endswith("/v1") else f"{url}/v1"
        # API key is optional for local LM Studio
        self.api_key = api_key or os.getenv("LMSTUDIO_API_KEY")
        self.timeout_s, self.max_retries = timeout_s, max_retries
        self.retry_base_wait, self.retry_max_wait = retry_base_wait, retry_max_wait
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if self.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"
        logger.info(f"Initialized LMStudio client at {self.base_url} with {max_retries} retries, {timeout_s}s timeout")

    def _post_once(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = self.session.post(url, json=data, timeout=self.timeout_s)
            if response.status_code == 429:
                ra = response.headers.get("Retry-After");
                if ra: logger.warning(f"Rate limited. Retry after: {ra}s")
                raise LMStudioAPIError("Rate limited (429)", status_code=429, response_data=response.json() if response.content else None)
            if 500 <= response.status_code < 600:
                msg = f"Server error ({response.status_code})"
                try:
                    ed = response.json();
                    if isinstance(ed, dict) and "error" in ed:
                        msg = f"{msg}: {ed['error']}"
                except Exception: pass
                raise LMStudioAPIError(msg, status_code=response.status_code)
            if 400 <= response.status_code < 500:
                msg = f"Client error ({response.status_code})"
                try:
                    ed = response.json();
                    if isinstance(ed, dict) and "error" in ed:
                        msg = f"{msg}: {ed['error']}"
                except Exception:
                    msg = f"{msg}: {response.text}"
                raise LMStudioError(msg)
            response.raise_for_status(); return response.json()
        except requests.exceptions.Timeout:
            raise LMStudioAPIError(f"Request timeout after {self.timeout_s}s")
        except requests.exceptions.ConnectionError as e:
            raise LMStudioAPIError(f"Connection error: {e}")
        except requests.exceptions.RequestException as e:
            raise LMStudioAPIError(f"Request error: {e}")

    def _make_request(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        retryer = Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=self.retry_base_wait, max=self.retry_max_wait),
            retry=retry_if_exception_type(LMStudioAPIError),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.INFO),
            reraise=True,
        )
        for _ in retryer:
            return self._post_once(url, payload)
        raise LMStudioAPIError("Exhausted retries without a response")

    def chat_completions(self, model: str, messages: List[Dict[str, Any]], temperature: float = 0.7, max_tokens: Optional[int] = None, response_format: Optional[Dict[str, Any]] = None, reasoning: Optional[Dict[str, Any]] = None, tools: Optional[List[Dict[str, Any]]] = None, tool_choice: Optional[str | Dict[str, Any]] = None, parallel_tool_calls: Optional[bool] = None, **kwargs: Any) -> Dict[str, Any]:
        if not isinstance(messages, list) or not messages:
            raise LMStudioError("`messages` must be a non-empty list")
        if not isinstance(model, str) or not model.strip():
            raise LMStudioError("`model` must be a non-empty string")
        if not (0.0 <= float(temperature) <= 2.0):
            raise LMStudioError("Temperature must be between 0.0 and 2.0")
        payload: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens is not None: payload["max_tokens"] = max_tokens
        if response_format is not None: payload["response_format"] = response_format
        if reasoning is not None: payload["reasoning"] = reasoning
        if tools is not None: payload["tools"] = tools
        if tool_choice is not None: payload["tool_choice"] = tool_choice
        if parallel_tool_calls is not None: payload["parallel_tool_calls"] = parallel_tool_calls
        payload.update(kwargs)
        log_payload = {k: v for k, v in payload.items() if k != 'messages'}; log_payload['message_count'] = len(messages)
        logger.info(f"Making LMStudio request: {log_payload}")
        url = f"{self.base_url}/chat/completions"; data = self._make_request(url, payload)
        try:
            if isinstance(data, dict) and 'usage' in data:
                u = data['usage']; logger.info(f"Request completed. Tokens - prompt: {u.get('prompt_tokens')}, completion: {u.get('completion_tokens')}, total: {u.get('total_tokens')}")
        except Exception: pass
        return data

    def extract_content(self, response: Dict[str, Any]) -> str:
        """Return assistant text content, tolerating structured formats."""
        try:
            choices = response.get('choices') or []
            if not choices: raise LMStudioError("No choices in response")
            msg = choices[0].get('message') or {}; content = msg.get('content')
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                def _collect(obj: Any) -> List[str]:
                    out: List[str] = []
                    if isinstance(obj, str):
                        s = obj.strip(); return [s] if s else []
                    if isinstance(obj, dict):
                        t = obj.get('text')
                        if isinstance(t, str) and t.strip(): out.append(t.strip())
                        elif isinstance(t, list):
                            for it in t: out.extend(_collect(it))
                        for k in ('content','value','message'):
                            v = obj.get(k)
                            if v is not None: out.extend(_collect(v))
                        return out
                    if isinstance(obj, list):
                        for it in obj: out.extend(_collect(it))
                        return out
                    return out
                parts = _collect(content)
                if parts: return "\n".join(parts)
            parsed = msg.get('parsed')
            if parsed is not None:
                import json as _json
                try: return _json.dumps(parsed, ensure_ascii=False)
                except Exception: return str(parsed)
            tc = msg.get('tool_calls') or []
            if tc:
                import json as _json
                summ = [{'id': (c.get('id') or '')[:12], 'name': ((c.get('function') or {}).get('name') or '')} for c in tc]
                return _json.dumps({'note': 'no_text_content', 'tool_calls': summ}, ensure_ascii=False)
            raise LMStudioError("Empty content in response")
        except (KeyError, IndexError, TypeError) as e:
            raise LMStudioError(f"Invalid response format: {e}")

    def extract_reasoning(self, response: Dict[str, Any]) -> Optional[str]:
        """Return reasoning if present (string or JSON string)."""
        try:
            choices = response.get("choices", [])
            if not choices: return None
            message = choices[0].get("message", {})
            reasoning = message.get("reasoning")
            if reasoning is None: return None
            if isinstance(reasoning, str): return reasoning
            import json as _json; return _json.dumps(reasoning, ensure_ascii=False)
        except Exception:
            return None

    def get_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return assistant tool_calls list if present, else []."""
        try:
            choices = response.get("choices", [])
            if not choices: return []
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
        if getattr(self, 'session', None): self.session.close(); logger.debug("Closed LMStudio client session")

    def __enter__(self): return self
    def __exit__(self, _exc_type, _exc_val, _exc_tb): self.close()
