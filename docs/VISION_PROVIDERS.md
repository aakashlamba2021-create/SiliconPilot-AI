# Vision provider setup

SiliconPilot uses the official Gemini Interactions API or OpenAI Responses API through the existing `requests` dependency. No provider SDK is required. Keys are read from the Streamlit process environment or the ignored local `.streamlit/secrets.toml`; they are never included in the page, prompt, or logs. Provider selection is visible in the assistant, and an error never causes a provider switch.

## Supported models

- Gemini defaults to `gemini-3.8-flash`, a current stable model listed in the Gemini API catalog as checked on 15 November 2026. Set `SILICONPILOT_GEMINI_MODEL` to change it. See the [Gemini model catalog](https://ai.google.dev/gemini-api/docs/models).
- OpenAI defaults to `gpt-6-astra`, whose current model documentation lists image input and structured output support. Set `SILICONPILOT_OPENAI_MODEL` to use another supported vision model. See [OpenAI GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra).

Model IDs and API availability can change; a configured ID is not authenticated until a real request succeeds.

## Configure in PowerShell

Use the project `.venv`. Enter the key through a secure prompt; do not paste it into Codex chat or save it in a script. The environment variables last only in this PowerShell process and child processes.

```powershell
$secureApiKey = Read-Host "Gemini API key" -AsSecureString
$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureApiKey)
try {
    $env:GEMINI_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    Remove-Variable secureApiKey, keyPointer
}
$env:SILICONPILOT_VISION_PROVIDER = "gemini"
$env:SILICONPILOT_GEMINI_MODEL = "gemini-3.8-flash"
Set-Location "C:\Users\aakas\OneDrive\Desktop\samsung project\SiliconPilot-AI"
.\scripts\run_local.ps1
```

For OpenAI, use the same secure prompt with `OPENAI_API_KEY`, then select the OpenAI model and provider:

```powershell
$secureApiKey = Read-Host "OpenAI API key" -AsSecureString
$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureApiKey)
try {
    $env:OPENAI_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    Remove-Variable secureApiKey, keyPointer
}
$env:SILICONPILOT_VISION_PROVIDER = "openai"
$env:SILICONPILOT_OPENAI_MODEL = "gpt-6-astra"
Set-Location "C:\Users\aakas\OneDrive\Desktop\samsung project\SiliconPilot-AI"
.\scripts\run_local.ps1
```

To use a different model, change the corresponding `SILICONPILOT_*_MODEL` value. To switch an already running application between configured providers, use the visible provider selector; it preserves the chat and images. If multiple keys are configured, no provider is chosen silently until the selector is used. The optional existing compatible endpoint can be selected with `SILICONPILOT_VISION_PROVIDER=openai_compatible`, `SILICONPILOT_VISION_MODEL`, `OPENAI_API_KEY`, and `SILICONPILOT_VISION_API_URL`.

Remove process-local credentials when finished:

```powershell
Remove-Item Env:GEMINI_API_KEY -ErrorAction SilentlyContinue
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
```

The project `.gitignore` excludes `.streamlit/secrets.toml`; keep credentials out of tracked files. A local secrets section can use provider-specific fields:

```toml
[vision_assistant]
gemini_api_key = "<Gemini API key>"
openai_api_key = "<OpenAI API key>"
gemini_model = "gemini-3.8-flash"
openai_model = "gpt-6-astra"
```

## Limits, usage and pricing

Defaults are four selected images, 18 MB maximum encoded request payload, 10 MB per uploaded file, 45 seconds per request, and 3,000 output tokens. Configure with `SILICONPILOT_VISION_MAX_IMAGES`, `SILICONPILOT_VISION_MAX_PAYLOAD_MB`, `SILICONPILOT_VISION_TIMEOUT_SECONDS`, and `SILICONPILOT_VISION_MAX_OUTPUT_TOKENS`. Gemini's current inline image request ceiling is 20 MB, so the default request cap stays below it. The application does not upload images to a separate file store and requests `store=false` from both APIs.

When returned, input, output and total token usage, client-side latency, provider request ID and configured model are shown in the answer. Cost is not estimated by default. Optional per-million-token rates can be configured with `SILICONPILOT_GEMINI_INPUT_USD_PER_MILLION_TOKENS`, `SILICONPILOT_GEMINI_OUTPUT_USD_PER_MILLION_TOKENS`, or their `SILICONPILOT_OPENAI_*` equivalents. The app only labels a calculated value as an estimate, and omits it when cached or thinking tokens make the rate basis incomplete. These rates are user-supplied and should match the account, model, tier, date, and image billing rules; monitor the provider's own usage and billing page for authoritative charges.

## Privacy and verification

For every image request, review the provider disclosures and consent controls tied to that exact question, selected images, labels, model and entered engineering context. The request also includes up to eight recent conversation messages, which can contain prior assistant visual findings. Image bytes are sent only after pressing **Analyze selected images**; ordinary text chat never attaches selected images. After a request, images remain selected in session state but are not resent automatically; a new request requires the dedicated action and fresh consent.

- Gemini's Free tier may use submitted content to improve Google products; the Paid tier says content is not used for that purpose. SiliconPilot cannot detect your tier. `store=false` opts out of stored Interactions records but is not a promise of zero retention. See [Gemini pricing and data use](https://ai.google.dev/gemini-api/docs/pricing) and [Interactions retention](https://ai.google.dev/gemini-api/docs/interactions-overview#data-storage-retention).
- OpenAI says API inputs and outputs are not used for training by default. Standard abuse-monitoring logs may retain content up to 30 days, subject to account controls. SiliconPilot sets `store=false` and cannot verify organization-specific retention settings. See [OpenAI API data controls](https://developers.openai.com/api/docs/guides/your-data).
- The data policy of an OpenAI-compatible endpoint is unknown to SiliconPilot and must be reviewed separately.

Do not send proprietary images unless your organization permits the chosen provider and account tier.

There is no separate health request. To verify access, use a non-confidential test image, choose one provider in the UI, review its disclosure, consent, and submit one question. Only a successful authenticated response marks request connectivity as verified. Mocked tests do not verify API access.
