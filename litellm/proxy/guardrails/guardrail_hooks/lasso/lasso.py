# +-------------------------------------------------------------+
#
#           Use Lasso Security Guardrails for your LLM calls
#                   https://www.lasso.security/
#
# +-------------------------------------------------------------+

import json
import os
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    TypedDict,
)

try:
    import ulid

    ULID_AVAILABLE = True
except ImportError:
    ulid = None  # type: ignore
    ULID_AVAILABLE = False

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore
    HTTPX_AVAILABLE = False

from fastapi import HTTPException

from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.integrations.custom_guardrail import dc as global_cache

from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails._content_utils import (
    build_inspection_messages,
    has_non_string_content,
)
from litellm.types.guardrails import GuardrailEventHooks
import litellm


# ── Intent double-duty ───────────────────────────────────────────────────────
# When an app funnels its LLM calls through LiteLLM and seeds per-turn intent headers
# (via the lasso-sdk GatewayIntent primitive), the plugin's existing /classify call
# double-duties as intent ingestion: it stamps the classify payload's messages with a
# per-turn traceId, a deterministic eventId, and a stride-spaced eventIndex, so Lasso's
# server derives the intent signal from role x content-kind and dedups re-sent history on
# eventId. Inert when no x-lasso-trace-id is present (a plain guardrail call), so the vertical
# ships dark until the app starts seeding. Mirrors the Kong ai-lasso-guardrail plugin (the
# reference intent implementation) and the server-side LiteLLM generic-guardrail shim.
_INTENT_TRACE_ID_HEADER = "x-lasso-trace-id"
_INTENT_SESSION_ID_HEADER = "x-session-id"
_INTENT_APP_INTENT_HEADER = "x-lasso-application-intent"
_INTENT_APP_NAME_HEADER = "x-lasso-application-name"
_INTENT_ENCODING_HEADER = "x-lasso-encoding"
# A header not on LiteLLM's forward-allowlist arrives as this placeholder, not a value.
_HEADER_PRESENT_PLACEHOLDER = "[present]"

# Reproducible events (text / tool_use / tool_result) are indexed on a stride so COMPLETION-only,
# non-reproducible events (reasoning) can slot strictly between them. A reproducible event at
# sequence position p gets eventIndex p*STRIDE; a reasoning event that precedes it fills just above
# the previous reproducible event — strictly after it, strictly before the next, and off the stride
# grid so its derived eventId can't collide with a reproducible one.
_EVENT_INDEX_STRIDE = 10
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _derive_event_id(trace_id: str, event_index: int) -> str:
    """Deterministic ULID-shaped event id, stable for a (trace_id, event_index) pair and unique
    within the trace, so the server's eventId-keyed dedup collapses the history re-sent every turn.
    Keeps the 26-char Crockford shape (passes the server's ULID validation): the trace_id's high 20
    chars + 6 chars of base32 index (32^6 ~= 1e9 events per trace)."""
    prefix = (trace_id or "")[:20]
    if len(prefix) < 20:
        prefix = prefix + "0" * (20 - len(prefix))
    n = int(event_index) if event_index else 0
    suffix = [""] * 6
    for i in range(5, -1, -1):
        suffix[i] = _CROCKFORD[n % 32]
        n //= 32
    return prefix + "".join(suffix)


def _stamp_intent_messages(
    messages: List[Dict[str, Any]], trace_id: str, start_index: int
) -> int:
    """Stamp classify messages in place with intent ids (traceId + stride eventIndex + deterministic
    eventId). ``start_index`` offsets the (0-based) sequence position so the response phase's
    COMPLETION events continue after the request phase's PROMPT events. Reasoning blocks are
    COMPLETION-only and never re-sent in history, so they take a non-reproducible gap index rather
    than a reproducible slot; a reasoning block with no valid slot (the trace's first event) is
    dropped — it's intent-only pass-through, so losing it costs the guardrail nothing. Returns the
    count of reproducible events (the offset the next phase continues from). Mirrors the Kong
    plugin's ``to_intent_messages``."""
    base = int(start_index) if start_index else 0
    repro = 0
    # Reasoning events since the last reproducible one, so several in a row get distinct gap indices.
    gap = 0
    kept: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        is_reasoning = isinstance(content, dict) and content.get("type") == "reasoning"
        if is_reasoning:
            index = (base + repro - 1) * _EVENT_INDEX_STRIDE + 1 + gap
            # Only negative if reasoning is the trace's very first event — no valid slot before 0.
            if index < 0:
                continue
            gap += 1
        else:
            index = (base + repro) * _EVENT_INDEX_STRIDE
            repro += 1
            gap = 0
        msg["traceId"] = trace_id
        msg["eventIndex"] = index
        msg["eventId"] = _derive_event_id(trace_id, index)
        kept.append(msg)
    messages[:] = kept
    return repro


class LassoResponse(TypedDict):
    """Type definition for Lasso API response."""

    violations_detected: bool
    deputies: Dict[str, bool]
    findings: Dict[str, List[Dict[str, Any]]]
    messages: Optional[List[Dict[str, str]]]


if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class LassoGuardrailMissingSecrets(Exception):
    """Exception raised when Lasso API key is missing."""

    pass


class LassoGuardrailAPIError(Exception):
    """Exception raised when there's an error calling the Lasso API."""

    pass


class LassoGuardrail(CustomGuardrail):
    """
    Lasso Security Guardrail integration for LiteLLM.

    Provides content moderation, PII detection, and policy enforcement
    through the Lasso Security API.
    """

    @classmethod
    def get_supported_event_hooks(cls) -> List[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
        ]

    def __init__(
        self,
        lasso_api_key: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        mask: Optional[bool] = False,
        **kwargs,
    ):
        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)
        self.lasso_api_key = lasso_api_key or api_key or os.environ.get("LASSO_API_KEY")
        self.user_id = user_id or os.environ.get("LASSO_USER_ID")
        self.conversation_id = conversation_id or os.environ.get("LASSO_CONVERSATION_ID")
        self.mask = mask or False

        if self.lasso_api_key is None:
            raise LassoGuardrailMissingSecrets(
                "Couldn't get Lasso api key, either set the `LASSO_API_KEY` in the environment or "
                "pass it as a parameter to the guardrail in the config file"
            )

        self.api_base = api_base or os.getenv("LASSO_API_BASE") or "https://server.lasso.security/gateway/v3"

        verbose_proxy_logger.debug(
            f"Lasso guardrail initialized: {kwargs.get('guardrail_name', 'unknown')}, "
            f"event_hook: {kwargs.get('event_hook', 'unknown')}, mask: {self.mask}"
        )

        super().__init__(**kwargs)

    @staticmethod
    def _get_field(obj: Any, field: str, default: Any = None) -> Any:
        """Get a field from either a dict or a Pydantic object."""
        if isinstance(obj, dict):
            return obj.get(field, default)
        return getattr(obj, field, default)

    @staticmethod
    def _extract_tool_call_fields(
        call: Any,
    ) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """Extract (call_id, name, parsed_input) from a tool call.

        Handles both dict-style and Pydantic object-style tool_calls.
        Parses the JSON arguments string into a dict when possible.
        """
        get = LassoGuardrail._get_field
        call_id = get(call, "id")
        func = get(call, "function")
        if not func:
            return call_id, None, None
        name = get(func, "name")
        args_str = get(func, "arguments")
        input_data: Optional[Dict[str, Any]] = None
        if args_str:
            try:
                parsed = json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if isinstance(parsed, dict):
                input_data = parsed
            else:
                # Preserve the raw argument string so Lasso still inspects
                # callers that smuggle PII/blocked content as malformed JSON
                # or non-object payloads.
                input_data = {"arguments": args_str}
        return call_id, name, input_data

    def _generate_ulid(self) -> str:
        """
        Generate a ULID (Universally Unique Lexicographically Sortable Identifier).
        Falls back to UUID if ULID library is not available.
        """
        if ULID_AVAILABLE and ulid is not None:
            return str(ulid.ULID())  # type: ignore
        else:
            verbose_proxy_logger.debug("ULID library not available, using UUID")
            return str(uuid.uuid4())

    @staticmethod
    def _extract_request_headers(data: dict) -> Dict[str, Any]:
        """Pull the original proxy request's inbound headers from the request data."""
        if not isinstance(data, dict):
            return {}
        proxy_server_request = data.get("proxy_server_request") or {}
        headers = (
            proxy_server_request.get("headers")
            if isinstance(proxy_server_request, dict)
            else None
        )
        if headers:
            return headers
        metadata = data.get("metadata") or {}
        return (metadata.get("headers") if isinstance(metadata, dict) else None) or {}

    @staticmethod
    def _get_header(headers: Any, name: str) -> Optional[str]:
        """Case-insensitive header lookup (HTTP headers are case-insensitive)."""
        if not isinstance(headers, dict):
            return None
        target = name.lower()
        for key, value in headers.items():
            if isinstance(key, str) and key.lower() == target:
                return value
        return None

    @staticmethod
    def _decode_header_value(value: str, pct_encoded: bool) -> str:
        """Percent-decode a seeded value when the SDK flagged non-ASCII with x-lasso-encoding: pct."""
        if not pct_encoded:
            return value
        try:
            from urllib.parse import unquote

            return unquote(value)
        except Exception:
            return value

    def _intent_context(self, data: dict) -> Optional[Dict[str, Any]]:
        """Parse app-seeded intent headers from the proxy request. Returns None when no per-turn
        x-lasso-trace-id is present (or it's the non-allowlisted "[present]" placeholder), leaving
        the call a plain content-safety classify."""
        headers = self._extract_request_headers(data)
        trace_id = self._get_header(headers, _INTENT_TRACE_ID_HEADER)
        if not trace_id or trace_id == _HEADER_PRESENT_PLACEHOLDER:
            return None

        pct_encoded = self._get_header(headers, _INTENT_ENCODING_HEADER) == "pct"
        session_id = self._get_header(headers, _INTENT_SESSION_ID_HEADER)
        app_intent = self._get_header(headers, _INTENT_APP_INTENT_HEADER)
        app_name = self._get_header(headers, _INTENT_APP_NAME_HEADER)

        session_information: Optional[Dict[str, str]] = None
        if app_intent:
            session_information = {
                "applicationIntent": self._decode_header_value(app_intent, pct_encoded)
            }
            if app_name:
                session_information["agenticAppName"] = self._decode_header_value(
                    app_name, pct_encoded
                )

        return {
            "trace_id": trace_id,
            "session_id": session_id
            if session_id and session_id != _HEADER_PRESENT_PLACEHOLDER
            else None,
            "session_information": session_information,
        }

    @staticmethod
    def _reasoning_text(message: Any) -> Optional[str]:
        """Extract the model's reasoning text from a completion message, if the provider echoed it.
        OpenAI-compatible proxies surface extended thinking as ``reasoning_content`` (a string)
        and/or ``thinking_blocks`` ({type: thinking, thinking: ...}); prefer the string, else join
        the blocks. Returns None when the message carries no reasoning (thinking off — the common
        case), so full reasoning coverage is a property of the caller's config, not this plugin."""
        get = LassoGuardrail._get_field
        reasoning_content = get(message, "reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content:
            return reasoning_content
        thinking_blocks = get(message, "thinking_blocks")
        if isinstance(thinking_blocks, list):
            parts = [
                block.get("thinking")
                for block in thinking_blocks
                if isinstance(block, dict) and isinstance(block.get("thinking"), str) and block.get("thinking")
            ]
            if parts:
                return "".join(parts)
        return None

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,  # Deprecated, use global_cache instead (kept to align with CustomGuardrail interface)
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
            "mcp_call",
            "anthropic_messages",
        ],
    ) -> Union[Exception, str, dict, None]:
        """
        Runs before the LLM API call to validate and potentially modify input.
        Uses 'PROMPT' messageType as this is input to the model.
        """
        # Check if this guardrail should run for this request
        event_type: GuardrailEventHooks = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        # Get or generate conversation_id and store it in data for post-call consistency
        # The conversation_id is being stored in the cache so it can be used by the post_call hook
        self._get_or_generate_conversation_id(data, global_cache)

        return await self._run_lasso_guardrail(
            data, global_cache, message_type="PROMPT", intent_ctx=self._intent_context(data)
        )

    @log_guardrail_information
    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal[
            "completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "responses",
            "mcp_call",
            "anthropic_messages",
        ],
        cache: DualCache,
    ):
        """
        This is used for during_call moderation.
        Uses 'PROMPT' messageType as this runs concurrently with input processing.
        """
        # Check if this guardrail should run for this request
        event_type: GuardrailEventHooks = GuardrailEventHooks.during_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        return await self._run_lasso_guardrail(
            data, cache, message_type="PROMPT", intent_ctx=self._intent_context(data)
        )

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        """
        Runs after the LLM API call to validate the response.
        Uses 'COMPLETION' messageType as this is output from the model.
        """
        # Check if this guardrail should run for this request
        event_type: GuardrailEventHooks = GuardrailEventHooks.post_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return response

        # Intent context comes from the ORIGINAL request headers (data), not the synthetic
        # response_data below — so reasoning harvesting and id stamping are gated on it.
        intent_ctx = self._intent_context(data)

        # Extract messages from the response for validation
        if isinstance(response, litellm.ModelResponse):
            response_messages: List[Dict[str, Any]] = []
            for choice in response.choices:
                if not hasattr(choice, "message"):
                    continue
                msg = choice.message
                # Chronological order within the model turn: reasoning first (REASONING_TEXT), then
                # the answer text (MODEL_RESPONSE), then any tool calls. Reasoning is intent-only
                # (a non-classified pass-through block), so it's harvested only when intent is active.
                if intent_ctx:
                    reasoning = self._reasoning_text(msg)
                    if reasoning:
                        response_messages.append(
                            {"role": "model", "content": {"type": "reasoning", "content": reasoning}}
                        )
                if msg.content:
                    response_messages.append({"role": "assistant", "content": msg.content})
                for call in getattr(msg, "tool_calls", None) or []:
                    call_id, name, input_data = self._extract_tool_call_fields(call)
                    if not call_id or not name:
                        continue
                    response_messages.append(
                        {
                            "role": "model",
                            "content": {
                                "type": "tool_use",
                                "id": call_id,
                                "name": name,
                                "input": input_data,
                            },
                        }
                    )

            if response_messages:
                # Include litellm_call_id from original data for conversation_id consistency
                response_data = {
                    "messages": response_messages,
                    "litellm_call_id": data.get("litellm_call_id"),
                }

                # Handle masking for post-call
                if self.mask:
                    headers = self._prepare_headers(response_data, global_cache)
                    payload = self._prepare_payload(
                        response_messages, response_data, global_cache, "COMPLETION", intent_ctx
                    )
                    api_url = f"{self.api_base}/classifix"

                    try:
                        lasso_response = await self._call_lasso_api(headers=headers, payload=payload, api_url=api_url)
                        self._process_lasso_response(lasso_response)

                        # Apply masking to the actual response if masked content is available
                        masked_messages = lasso_response.get("messages")
                        if lasso_response.get("violations_detected") and masked_messages:
                            self._apply_masking_to_model_response(response, masked_messages)
                            verbose_proxy_logger.debug("Applied Lasso masking to model response")
                    except Exception as e:
                        if isinstance(e, HTTPException):
                            raise e
                        verbose_proxy_logger.error(f"Error in post-call Lasso masking: {str(e)}")
                        raise LassoGuardrailAPIError(f"Failed to apply post-call masking: {str(e)}")
                else:
                    # Use the same data for conversation_id consistency (no cache access needed)
                    await self._run_lasso_guardrail(
                        response_data, cache=global_cache, message_type="COMPLETION", intent_ctx=intent_ctx
                    )
                    verbose_proxy_logger.debug("Post-call Lasso validation completed")
            else:
                verbose_proxy_logger.warning("No response messages found to validate")
        else:
            verbose_proxy_logger.warning(f"Unexpected response type for post-call hook: {type(response)}")

        return response

    def _get_or_generate_conversation_id(self, data: dict, cache: DualCache) -> str:
        """
        Get or generate a conversation_id for this request.

        This method ensures session consistency by using litellm_call_id as a cache key.
        The same conversation_id is used for both pre-call and post-call hooks within
        the same request, enabling proper conversation grouping in Lasso UI.

        Example:
            >>> guardrail = LassoGuardrail(lasso_api_key="key")
            >>> data = {"litellm_call_id": "call_123"}
            >>> conversation_id = guardrail._get_or_generate_conversation_id(data, cache)
            >>> # Returns consistent ID for same litellm_call_id

        Args:
            data: The request data containing litellm_call_id
            cache: The cache instance for storing conversation_id

        Returns:
            str: The conversation_id to use for this request
        """
        # Use global conversation_id if set
        if self.conversation_id:
            return self.conversation_id

        # Get the litellm_call_id which is consistent across all hooks for this request
        litellm_call_id = data.get("litellm_call_id")

        if not litellm_call_id:
            # Fallback to generating a new ULID if no litellm_call_id available
            return self._generate_ulid()

        # Use litellm_call_id as cache key for conversation_id
        cache_key = f"lasso_conversation_id:{litellm_call_id}"

        # Try to get existing conversation_id from cache
        try:
            cached_conversation_id = cache.get_cache(cache_key)
            if cached_conversation_id:
                return cached_conversation_id
        except Exception as e:
            verbose_proxy_logger.warning(f"Cache retrieval failed: {e}")

        # Generate new conversation_id and store in cache
        generated_id = self._generate_ulid()

        try:
            cache.set_cache(cache_key, generated_id, ttl=3600)  # Cache for 1 hour
        except Exception as e:
            verbose_proxy_logger.warning(f"Cache storage failed: {e}")

        return generated_id

    async def _run_lasso_guardrail(
        self,
        data: dict,
        cache: DualCache,
        message_type: Literal["PROMPT", "COMPLETION"] = "PROMPT",
        intent_ctx: Optional[Dict[str, Any]] = None,
    ):
        """
        Run the Lasso guardrail with the specified message type.

        This is the core method that handles both classification and masking workflows.
        It chooses the appropriate API endpoint based on the masking configuration
        and processes the response according to Lasso's action-based system.

        Workflow:
        1. Validate messages are present
        2. Prepare headers and payload
        3. Choose API endpoint (classify vs classifix)
        4. Call Lasso API
        5. Process response and apply masking if needed
        6. Handle blocking vs non-blocking violations

        Args:
            data: The request data containing messages
            cache: The cache instance for storing conversation_id (optional for post-call)
            message_type: Either "PROMPT" for input or "COMPLETION" for output

        Raises:
            LassoGuardrailAPIError: If the Lasso API call fails
            HTTPException: If blocking violations are detected
        """
        raw_messages: List[Dict[str, Any]] = data.get("messages") or []
        messages: List[Dict[str, Any]] = self._expand_messages_for_classification(raw_messages) if raw_messages else []
        messages_count = len(messages)
        if data.get("input") is not None:
            # Responses-API payloads carry text in data["input"]. Inspect it
            # alongside any "messages" array — otherwise a caller can attach
            # benign messages and stash blocked content in input to bypass.
            messages.extend(build_inspection_messages({"input": data["input"]}))
        if not messages:
            return data

        # Lasso's classifix endpoint returns masked text that we copy back
        # into ``data["messages"]``. For multimodal/Responses-API input we
        # would silently strip image/audio parts, so fall back to the
        # classify endpoint (which still raises on BLOCK actions) and
        # leave the original payload intact.
        if self.mask and not has_non_string_content(data):
            return await self._handle_masking(data, cache, message_type, messages, messages_count, intent_ctx)
        return await self._handle_classification(data, cache, message_type, messages, intent_ctx)

    async def _handle_classification(
        self,
        data: dict,
        cache: DualCache,
        message_type: Literal["PROMPT", "COMPLETION"],
        messages: List[Dict[str, Any]],
        intent_ctx: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Handle classification without masking."""
        try:
            headers = self._prepare_headers(data, cache)
            payload = self._prepare_payload(messages, data, cache, message_type, intent_ctx)
            response = await self._call_lasso_api(headers=headers, payload=payload)
            self._process_lasso_response(response)
            return data
        except Exception as e:
            await self._handle_api_error(e, message_type)
            return data  # This line won't be reached due to exception, but satisfies type checker

    async def _handle_masking(
        self,
        data: dict,
        cache: DualCache,
        message_type: Literal["PROMPT", "COMPLETION"],
        messages: List[Dict[str, Any]],
        messages_count: int,
        intent_ctx: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Handle masking with classifix endpoint.

        ``messages_count`` is the number of inspected items derived from
        ``data["messages"]``; any items beyond that index came from
        ``data["input"]`` and must be written back there, not into messages.
        """
        try:
            headers = self._prepare_headers(data, cache)
            payload = self._prepare_payload(messages, data, cache, message_type, intent_ctx)
            api_url = f"{self.api_base}/classifix"
            response = await self._call_lasso_api(headers=headers, payload=payload, api_url=api_url)
            self._process_lasso_response(response)

            # Apply masking to messages if violations detected and masked messages are available.
            # Map masked content back onto the original OpenAI-format messages so the
            # downstream provider receives a compatible payload.
            masked = response.get("messages")
            if response.get("violations_detected") and masked:
                masked_for_messages = masked[:messages_count]
                masked_for_input = masked[messages_count:]
                if data.get("messages"):
                    data["messages"] = self._map_masked_messages_back(data["messages"], masked_for_messages)
                # Also update data["input"] for Responses-API payloads so the
                # unredacted text doesn't leak through that field.
                if isinstance(data.get("input"), str):
                    text_parts = [msg["content"] for msg in masked_for_input if isinstance(msg.get("content"), str)]
                    if text_parts:
                        data["input"] = "\n".join(text_parts)
                self._log_masking_applied(message_type, dict(response))

            return data
        except Exception as e:
            await self._handle_api_error(e, message_type)
            return data  # This line won't be reached due to exception, but satisfies type checker

    def _map_masked_messages_back(
        self,
        original_messages: List[Dict[str, Any]],
        masked_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Map Lasso-format masked messages back onto the original OpenAI-format messages.

        Lasso receives expanded messages (tool_use / tool_result blocks) and returns them
        in the same Lasso-internal format with sensitive values replaced.  Writing those
        blocks straight into data["messages"] would corrupt the OpenAI-compatible schema
        the downstream provider expects.  This helper re-applies only the masked content
        while preserving the original structure.
        """
        # Index masked content by type so we can look up by id without caring about order.
        masked_tool_use: Dict[str, Dict[str, Any]] = {}
        masked_tool_result: Dict[str, str] = {}
        masked_text: List[str] = []

        for msg in masked_messages:
            content = msg.get("content")
            if isinstance(content, dict):
                if content.get("type") == "tool_use":
                    call_id = content.get("id")
                    if call_id:
                        masked_tool_use[call_id] = content
                elif content.get("type") == "tool_result":
                    tool_use_id = content.get("tool_use_id")
                    if tool_use_id:
                        masked_tool_result[tool_use_id] = content.get("content", "")
            elif isinstance(content, str):
                masked_text.append(content)

        # Positional cursor only works if Lasso echoes every text message back.
        # Skip text remap on count mismatch to avoid writing masked content
        # onto the wrong original message.
        original_text_count = sum(
            1
            for m in original_messages
            if m.get("role") != "tool"
            and ((isinstance(m.get("content"), str) and m.get("content")) or isinstance(m.get("content"), list))
        )
        apply_text_cursor = original_text_count == len(masked_text)
        if not apply_text_cursor and masked_text:
            verbose_proxy_logger.warning(
                "Lasso masked-text count mismatch; skipping text remap",
                extra={
                    "original_text_count": original_text_count,
                    "masked_text_count": len(masked_text),
                },
            )

        result: List[Dict[str, Any]] = []
        text_cursor = 0

        for orig_msg in original_messages:
            msg = dict(orig_msg)
            role = msg.get("role")
            content = msg.get("content")

            if role == "tool":
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id and tool_call_id in masked_tool_result:
                    msg["content"] = masked_tool_result[tool_call_id]

            elif isinstance(content, str) and content:
                if apply_text_cursor and text_cursor < len(masked_text):
                    msg["content"] = masked_text[text_cursor]
                    text_cursor += 1
                if role == "assistant" and orig_msg.get("tool_calls"):
                    msg["tool_calls"] = self._update_tool_calls_from_masked(orig_msg["tool_calls"], masked_tool_use)

            elif isinstance(content, list):
                # Multimodal list content was flattened to a text string before
                # being sent to Lasso.  Replace the list with the masked text
                # so the cursor stays aligned with subsequent messages.
                if apply_text_cursor and text_cursor < len(masked_text):
                    msg["content"] = masked_text[text_cursor]
                    text_cursor += 1
                if role == "assistant" and orig_msg.get("tool_calls"):
                    msg["tool_calls"] = self._update_tool_calls_from_masked(orig_msg["tool_calls"], masked_tool_use)

            elif role == "assistant" and not content and orig_msg.get("tool_calls"):
                msg["tool_calls"] = self._update_tool_calls_from_masked(orig_msg["tool_calls"], masked_tool_use)

            result.append(msg)

        return result

    def _update_tool_calls_from_masked(
        self,
        tool_calls: List[Any],
        masked_tool_use: Dict[str, Dict[str, Any]],
    ) -> List[Any]:
        """Replace tool_call arguments with masked values returned by Lasso."""
        updated = []
        for call in tool_calls:
            call_id = self._get_field(call, "id")
            if call_id and call_id in masked_tool_use:
                masked_input = masked_tool_use[call_id].get("input")
                if masked_input is not None:
                    if isinstance(call, dict):
                        call = dict(call)
                        func_dict = dict(call.get("function", {}))
                        func_dict["arguments"] = json.dumps(masked_input)
                        call["function"] = func_dict
                    else:
                        func_obj = getattr(call, "function", None)
                        if func_obj:
                            func_obj.arguments = json.dumps(masked_input)
            updated.append(call)
        return updated

    async def _handle_api_error(
        self,
        error: Exception,
        message_type: Literal["PROMPT", "COMPLETION"],
    ) -> None:
        """Handle API errors with specific error types."""
        if isinstance(error, HTTPException):
            raise error

        # Log error with context
        verbose_proxy_logger.error(
            f"Error calling Lasso API: {str(error)}",
            extra={
                "guardrail_name": getattr(self, "guardrail_name", "unknown"),
                "message_type": message_type,
                "error_type": type(error).__name__,
            },
        )

        # Handle specific error types if httpx is available
        if HTTPX_AVAILABLE:
            if isinstance(error, httpx.TimeoutException):
                raise LassoGuardrailAPIError("Lasso API timeout")
            elif isinstance(error, httpx.HTTPStatusError):
                if error.response.status_code == 401:
                    raise LassoGuardrailMissingSecrets("Invalid API key")
                elif error.response.status_code == 429:
                    raise LassoGuardrailAPIError("Lasso API rate limit exceeded")
                else:
                    raise LassoGuardrailAPIError(f"API error: {error.response.status_code}")

        # Generic error handling
        raise LassoGuardrailAPIError(f"Failed to verify request safety with Lasso API: {str(error)}")

    def _log_masking_applied(
        self,
        message_type: Literal["PROMPT", "COMPLETION"],
        response: Dict[str, Any],
    ) -> None:
        """Log masking application with structured context."""
        conversation_id = getattr(self, "conversation_id", "unknown")
        verbose_proxy_logger.debug(
            "Lasso masking applied",
            extra={
                "guardrail_name": getattr(self, "guardrail_name", "unknown"),
                "message_type": message_type,
                "violations_count": len(response.get("findings", {})),
                "masked_fields": len(response.get("messages", [])),
                "conversation_id": conversation_id,
            },
        )

    def _expand_messages_for_classification(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert raw OpenAI-format messages to Lasso API format with content blocks.

        - assistant messages with `tool_calls` → assistant message per tool_use block
        - role=tool messages → developer role + tool_result block
        - plain text messages pass through unchanged
        """
        expanded: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content")

            if role == "tool":
                tool_call_id = msg.get("tool_call_id")
                if not tool_call_id:
                    verbose_proxy_logger.warning("Skipping tool message without tool_call_id")
                    continue
                # Flatten multimodal list content to text so Lasso's
                # tool_result.content field receives a string.
                if isinstance(content, list):
                    text_parts = [
                        part["text"]
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
                    ]
                    tool_result_content = "\n".join(text_parts)
                else:
                    tool_result_content = content or ""
                expanded.append(
                    {
                        "role": "developer",
                        "content": {
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": tool_result_content,
                        },
                    }
                )
                continue

            if isinstance(content, list):
                # Flatten multimodal content arrays to plain text for Lasso.
                text_parts = [
                    part["text"]
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
                ]
                if text_parts:
                    expanded.append({"role": role, "content": "\n".join(text_parts)})
            elif content:
                # Empty string and ``None`` are skipped on purpose: empty
                # carries no inspectable text and ``None`` is the standard
                # OpenAI shape for a pure tool-call turn. Dict content
                # (pre-built tool_use/tool_result blocks from the post-call
                # path) passes through unchanged.
                expanded.append({"role": role, "content": content})

            if role == "assistant":
                for call in msg.get("tool_calls") or []:
                    call_id, name, input_data = self._extract_tool_call_fields(call)
                    if not call_id or not name:
                        verbose_proxy_logger.warning(
                            "Skipping malformed tool_call",
                            extra={"call_id": call_id, "name": name},
                        )
                        continue
                    expanded.append(
                        {
                            "role": "model",
                            "content": {
                                "type": "tool_use",
                                "id": call_id,
                                "name": name,
                                "input": input_data,
                            },
                        }
                    )

        return expanded

    def _prepare_headers(self, data: dict, cache: DualCache) -> Dict[str, str]:
        """Prepare headers for the Lasso API request."""
        if not self.lasso_api_key:
            raise LassoGuardrailMissingSecrets(
                "Couldn't get Lasso api key, either set the `LASSO_API_KEY` in the environment or "
                "pass it as a parameter to the guardrail in the config file"
            )

        headers: Dict[str, str] = {
            "lasso-api-key": self.lasso_api_key,
            "Content-Type": "application/json",
        }

        # Add optional headers if provided
        if self.user_id:
            headers["lasso-user-id"] = self.user_id

        # Always include conversation_id (generated or provided)
        conversation_id = self._get_or_generate_conversation_id(data, cache)

        headers["lasso-conversation-id"] = conversation_id

        return headers

    def _prepare_payload(
        self,
        messages: List[Dict[str, Any]],
        data: dict,
        cache: DualCache,
        message_type: Literal["PROMPT", "COMPLETION"] = "PROMPT",
        intent_ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Prepare the payload for the Lasso API request.

        Args:
            messages: List of message objects (may contain tool_use/tool_result content blocks)
            message_type: Type of message - "PROMPT" for input, "COMPLETION" for output
            data: Request data (used for conversation_id generation and tools extraction)
            cache: Cache instance for storing conversation_id (optional for post-call)
            intent_ctx: Parsed intent headers (see _intent_context); when present the payload
                also carries per-turn intent ids so the same /classify call feeds intent.
        """
        payload: Dict[str, Any] = {"messages": messages, "messageType": message_type}

        # Add optional parameters if available
        if self.user_id:
            payload["userId"] = self.user_id

        # Always include sessionId (conversation_id - generated or provided)
        conversation_id = self._get_or_generate_conversation_id(data, cache)
        payload["sessionId"] = conversation_id

        # Attribute the inference to the LiteLLM guardrail ("Used By" badge on Application API Keys).
        payload["source"] = {"type": "litellm"}

        # Map OpenAI ChatCompletionToolParam array → ToolDefinition array
        tools_data: List[Dict[str, Any]] = data.get("tools") or []
        if tools_data:
            get = self._get_field
            tool_definitions = []
            for tool in tools_data:
                func = get(tool, "function")
                if not func:
                    continue
                name = get(func, "name")
                if not name:
                    continue
                td: Dict[str, Any] = {"name": name}
                description = get(func, "description")
                if description:
                    td["description"] = description
                parameters = get(func, "parameters")
                if parameters:
                    td["parameters"] = parameters
                tool_definitions.append(td)
            if tool_definitions:
                payload["tools"] = tool_definitions

        if intent_ctx:
            self._apply_intent(payload, intent_ctx, message_type, data, cache)

        return payload

    def _apply_intent(
        self,
        payload: Dict[str, Any],
        intent_ctx: Dict[str, Any],
        message_type: Literal["PROMPT", "COMPLETION"],
        data: dict,
        cache: DualCache,
    ) -> None:
        """Stamp the classify payload with intent ids so the same call feeds the intent pipeline.

        The PROMPT phase indexes the re-sent history from 0 and caches its reproducible-event count;
        the COMPLETION phase continues from that offset (keyed by litellm_call_id, stable across the
        pre/post hooks of one call) so the model's response sorts after the prompt and re-sent history
        keeps identical, deduped ids.
        """
        # The app-seeded conversation id wins for sessionId so guardrail + intent agree on the dialogue.
        if intent_ctx.get("session_id"):
            payload["sessionId"] = intent_ctx["session_id"]
        if intent_ctx.get("session_information"):
            payload["sessionInformation"] = intent_ctx["session_information"]

        call_id = data.get("litellm_call_id")
        offset_key = f"lasso_event_offset:{call_id}" if call_id else None

        start_index = 0
        if message_type == "COMPLETION" and offset_key:
            try:
                cached = cache.get_cache(offset_key)
                if cached is not None:
                    start_index = int(cached)
            except Exception as e:
                verbose_proxy_logger.warning(f"Intent offset retrieval failed: {e}")

        repro = _stamp_intent_messages(payload["messages"], intent_ctx["trace_id"], start_index)

        if message_type == "PROMPT" and offset_key:
            try:
                cache.set_cache(offset_key, repro, ttl=3600)
            except Exception as e:
                verbose_proxy_logger.warning(f"Intent offset storage failed: {e}")

    async def _call_lasso_api(
        self,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        api_url: Optional[str] = None,
    ) -> LassoResponse:
        """Call the Lasso API and return the response."""
        url = api_url or f"{self.api_base}/classify"
        verbose_proxy_logger.debug(f"Calling Lasso API with messageType: {payload.get('messageType')}")
        response = await self.async_handler.post(
            url=url,
            headers=headers,
            json=payload,
            timeout=10.0,
        )
        response.raise_for_status()
        return response.json()

    def _process_lasso_response(self, response: LassoResponse) -> None:
        """
        Process the Lasso API response and handle violations according to action types.

        This method implements the action-based blocking logic:
        - BLOCK: Raises HTTPException to stop request/response
        - AUTO_MASKING: Logs warning and continues (masking applied elsewhere)
        - WARN: Logs warning and continues

        Example Response:
            {
                "violations_detected": true,
                "findings": {
                    "jailbreak": [{
                        "action": "BLOCK",
                        "severity": "HIGH"
                    }]
                }
            }

        Args:
            response: The response dictionary from Lasso API

        Raises:
            HTTPException: If any finding has "action": "BLOCK"
        """
        if response and response.get("violations_detected") is True:
            violated_deputies = self._parse_violated_deputies(response)
            verbose_proxy_logger.warning(f"Lasso guardrail detected violations: {violated_deputies}")

            # Check if any findings have "BLOCK" action
            blocking_violations = self._check_for_blocking_actions(response)

            if blocking_violations:
                # Block the request/response for findings with "BLOCK" action
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Violated Lasso guardrail policy",
                        "detection_message": f"Blocking violations detected: {', '.join(blocking_violations)}",
                        "lasso_response": response,
                    },
                )
            else:
                # Continue with warning for non-blocking violations (e.g., AUTO_MASKING)
                verbose_proxy_logger.info(
                    f"Non-blocking Lasso violations detected, continuing with warning: {violated_deputies}"
                )

    def _check_for_blocking_actions(self, response: LassoResponse) -> List[str]:
        """
        Check findings for actions that should block the request/response.

        Examines the findings section of the Lasso response to identify which
        deputies have violations with "BLOCK" action. This enables granular
        control where some violations (like PII) can be masked while others
        (like jailbreaks) are blocked entirely.

        Args:
            response: The response dictionary from Lasso API

        Returns:
            List[str]: Names of deputies with blocking violations

        Example:
            >>> response = {
            ...     "findings": {
            ...         "jailbreak": [{"action": "BLOCK"}],
            ...         "pattern-detection": [{"action": "AUTO_MASKING"}]
            ...     }
            ... }
            >>> guardrail._check_for_blocking_actions(response)
            ['jailbreak']
        """
        blocking_violations = []
        findings = response.get("findings", {})

        for deputy_name, deputy_findings in findings.items():
            if isinstance(deputy_findings, list):
                for finding in deputy_findings:
                    if isinstance(finding, dict) and finding.get("action") == "BLOCK":
                        if deputy_name not in blocking_violations:
                            blocking_violations.append(deputy_name)
                        break  # No need to check other findings for this deputy

        return blocking_violations

    def _parse_violated_deputies(self, response: LassoResponse) -> List[str]:
        """Parse the response to extract violated deputies."""
        violated_deputies = []
        if "deputies" in response:
            for deputy, is_violated in response["deputies"].items():
                if is_violated:
                    violated_deputies.append(deputy)
        return violated_deputies

    def _apply_masking_to_model_response(
        self,
        model_response: litellm.ModelResponse,
        masked_messages: List[Dict[str, Any]],
    ) -> None:
        """Apply masking to the actual model response when mask=True and masked content is available."""
        # Index masked tool_use blocks by id for O(1) lookup.
        masked_tool_use: Dict[str, Dict[str, Any]] = {}
        masked_text: List[str] = []
        for masked_msg in masked_messages:
            content = masked_msg.get("content")
            if isinstance(content, dict) and content.get("type") == "tool_use":
                call_id = content.get("id")
                if call_id:
                    masked_tool_use[call_id] = content
            elif isinstance(content, str):
                masked_text.append(content)

        # Count text-bearing choices to verify 1:1 mapping with masked texts.
        original_text_count = sum(1 for c in model_response.choices if hasattr(c, "message") and c.message.content)
        apply_text = original_text_count == len(masked_text)
        if not apply_text and masked_text:
            verbose_proxy_logger.warning(
                "Lasso masked-text count mismatch in model response; skipping text remap",
                extra={
                    "original_text_count": original_text_count,
                    "masked_text_count": len(masked_text),
                },
            )

        text_cursor = 0
        for choice in model_response.choices:
            if not hasattr(choice, "message"):
                continue
            msg = choice.message

            if msg.content and apply_text and text_cursor < len(masked_text):
                msg.content = masked_text[text_cursor]
                text_cursor += 1
                verbose_proxy_logger.debug(f"Applied masked text content to choice {text_cursor}")

            for call in getattr(msg, "tool_calls", None) or []:
                call_id = self._get_field(call, "id")
                if call_id and call_id in masked_tool_use:
                    masked_input = masked_tool_use[call_id].get("input")
                    if masked_input is not None:
                        if isinstance(call, dict):
                            func = call.get("function", {})
                            if isinstance(func, dict):
                                func["arguments"] = json.dumps(masked_input)
                        else:
                            func = getattr(call, "function", None)
                            if func:
                                func.arguments = json.dumps(masked_input)
                        verbose_proxy_logger.debug(f"Applied masked tool_call arguments for call_id={call_id}")

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.lasso import (
            LassoGuardrailConfigModel,
        )

        return LassoGuardrailConfigModel
