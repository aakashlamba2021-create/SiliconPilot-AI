# Vision Provider Setup

The text-only assistant does not require API credentials. Gemini and OpenAI are optional providers for user-requested image analysis.

## Environment variables

Set the relevant key in the environment that launches Streamlit:

| Provider | Required variable | Default model |
|---|---|---|
| Gemini | `GEMINI_API_KEY` | `gemini-3.8-flash` |
| OpenAI | `OPENAI_API_KEY` | `gpt-6-astra` |

Optional model selectors are `SILICONPILOT_GEMINI_MODEL` and `SILICONPILOT_OPENAI_MODEL`. Optional per-million-token rates for cost estimation are `SILICONPILOT_GEMINI_INPUT_USD_PER_MILLION_TOKENS`, `SILICONPILOT_GEMINI_OUTPUT_USD_PER_MILLION_TOKENS`, `SILICONPILOT_OPENAI_INPUT_USD_PER_MILLION_TOKENS`, and `SILICONPILOT_OPENAI_OUTPUT_USD_PER_MILLION_TOKENS`. Cost remains unavailable unless required response usage and configured rates exist.

For local deployments, Streamlit `secrets.toml` can provide the corresponding API key and model settings. Keep that file out of version control; never put keys in screenshots, exported reports, or application logs. The Overview/System information pages show configuration state without displaying secret values.

Restart the app after changing its environment. The assistant page must list a provider as configured before it can be used. A successful configuration check is not a successful live request.

## Consent and outbound data

The selected prompt, conversation context, and uploaded image bytes are sent to the chosen provider only when the user confirms the request. Compare mode sends the same selected input to both Gemini and OpenAI and requires explicit consent for both. Provider errors do not trigger a silent fallback. Use only material you are authorized to share and avoid confidential layout screenshots unless your organization allows that destination.

## Endpoints and current documentation

- Gemini uses the Interactions API: [Gemini API models](https://ai.google.dev/gemini-api/docs/models).
- OpenAI uses the Responses API: [OpenAI models](https://developers.openai.com/api/docs/models) and [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses).

Provider behavior can change. As checked on 15 November 2026, Google's Gemini docs list `gemini-3.8-flash` as a current stable Flash model, while OpenAI lists `gpt-6-astra` as a current vision-capable model with Responses API support. Recheck the provider model catalog and data controls before external use.

Gemini's official pricing page says free-tier content may be used to improve products and paid-tier content is not used for that purpose. OpenAI's data-controls page says `/v1/responses` inputs are not used for training and default abuse-monitoring retention is up to 30 days; setting `store=false` addresses application-state storage and does not remove the separate abuse-monitoring policy. The current UI states these distinctions and also notes account-specific controls are not verified by SiliconPilot.

The project uses a structured response schema and tests it with local provider fixtures. No live provider request was made during the current validation because this environment has no provider credentials configured. Verify model access, regional availability, quotas, pricing, and current API behavior on the provider account before a live demo.
