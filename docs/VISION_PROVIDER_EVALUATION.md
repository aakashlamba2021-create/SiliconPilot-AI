# Vision provider evaluation

No reviewed analog-layout image benchmark is bundled with this framework. It does not claim Gemini or OpenAI quality results. Keep evaluation cases separate from model training datasets. Use only reviewed, non-confidential images that you are permitted to submit to both providers.

## Case manifest

Create a local JSON manifest. Image paths are relative to the manifest, or may be absolute. Use labels that make comparisons clear. The tool accepts the same case prompt, image bytes, labels, and context for Gemini and OpenAI.

```json
{
  "cases": [
    {
      "case_id": "reviewed-pair-001",
      "question": "Compare the visible differential-pair placement and routing. State what cannot be determined from these screenshots.",
      "comparison_requested": true,
      "engineering_context": {
        "circuit_topology": "user supplied, if known",
        "critical_matching_requirements": "user supplied, if known"
      },
      "images": [
        {"path": "./layout_before.png", "label": "Original"},
        {"path": "./layout_after.png", "label": "Modified"}
      ],
      "reviewer_notes": "Expected observations and important unknowns, prepared before provider review."
    }
  ]
}
```

## Run the paired comparison

Configure both keys and model IDs in the PowerShell process, then run the explicit command below. `--confirm-external-upload` is required. Each image is sent once to each provider; there are no automatic retries or fallback providers. The output path must be new. The JSON report contains provider responses, filenames, metadata, empty human-review score fields, and no image bytes or API keys. Keep it in a location appropriate for the sensitivity of the generated analysis.

```powershell
..venv\Scripts\python.exe scripts\evaluate_vision_providers.py .\vision_cases.json --output .\vision_comparison.json --confirm-external-upload
```

The report leaves all quality scores empty until a human reviewer completes them. Score each quality category from 1 (poor) to 5 (strong), or leave it null if the case cannot assess that category. Record response latency, token usage and cost metadata as returned; do not infer missing metrics. Review these criteria:

1. Visible device recognition.
2. Differential-pair symmetry interpretation.
3. Current-mirror and common-centroid interpretation.
4. Routing observations.
5. Congestion identification.
6. Recognition of uncertainty.
7. Avoidance of invented details.
8. Technical usefulness of recommendations.
9. Quality of verification suggestions.
10. Response latency (observed milliseconds).
11. API usage and cost (provider-reported usage and optional configured-rate estimate).

Compare providers on the same cases, record image/crop limitations, and review uncertain or incorrect observations against the source screenshot. A small user-selected set is not representative proof of general model capability, DRC/LVS correctness, electrical performance, simulation, or measured silicon behavior. Never convert these scores into a model trust approval automatically.
