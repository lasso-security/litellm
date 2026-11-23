from typing import Literal

from pydantic import Field

from litellm.types.guardrails import GuardrailNumRetries
from litellm.types.llms.base import LiteLLMBaseModel

from .base import GuardrailConfigModel


class LassoGuardrailConfigModelOptionalParams(LiteLLMBaseModel):
    user_id: str | None = Field(
        default=None,
        description="The user ID for the Lasso guardrail. If not provided, the `LASSO_USER_ID` environment variable is checked.",
    )
    conversation_id: str | None = Field(
        default=None,
        description="The conversation ID for the Lasso guardrail. If not provided, the `LASSO_CONVERSATION_ID` environment variable is checked.",
    )
    unreachable_fallback: Literal["fail_closed", "fail_open"] | None = Field(
        default="fail_closed",
        description=(
            "What to do when Lasso can't be reached, times out, or answers 408, 429, 502, 503 or 504. "
            "'fail_closed' (default) fails the request. 'fail_open' lets it through unchecked and records the failure "
            "in the request's guardrail logs."
        ),
    )
    fail_on_error: bool | None = Field(
        default=True,
        description=(
            "False lets the request through on any Lasso error, not only unreachability, so only a Lasso verdict "
            "can block or mask it. True (default) fails the request."
        ),
    )
    num_retries: GuardrailNumRetries = None


class LassoGuardrailConfigModel(GuardrailConfigModel[LassoGuardrailConfigModelOptionalParams]):
    api_key: str | None = Field(
        default=None,
        description="The API key for the Lasso guardrail. If not provided, the `LASSO_API_KEY` environment variable is checked.",
    )
    api_base: str | None = Field(
        default=None,
        description="The API base for the Lasso guardrail. Default is https://server.lasso.security. Also checks if the `LASSO_API_BASE` environment variable is set.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Lasso Guardrail"
