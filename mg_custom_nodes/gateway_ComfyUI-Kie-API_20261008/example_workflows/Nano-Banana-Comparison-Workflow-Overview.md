# ComfyUI KIE API Workflow — Nano Banana Pro + Nano Banana 2 Comparison

This workflow is built for side-by-side testing of **Nano Banana Pro** and **Nano Banana 2** inside ComfyUI using the Kie.ai API.

It is designed to make model differences easy to evaluate across prompt quality, image fidelity, and grounding behavior in a repeatable setup.

**Help Support Development**

Use this affiliate link when accessing Kie.ai:
[https://kie.ai](https://kie.ai?ref=e7565cf24a7fad4586341a87eaf21e42)

Project repository:
[https://github.com/gateway/ComfyUI-Kie-API](https://github.com/gateway/ComfyUI-Kie-API)

---

## Pricing Snapshot

- **Nano Banana 2**
  - `1K`: 8 credits (~$0.04)
  - `2K`: 12 credits (~$0.06)
  - `4K`: 18 credits (~$0.09)
  - Roughly 40% below official listed pricing.

- **Nano Banana Pro**
  - `1K/2K`: 18 credits (~$0.09)
  - `4K`: 24 credits (~$0.12)

Pricing may change over time. Check the latest details at:
[https://kie.ai/pricing](https://kie.ai/pricing)

---

## Included Nodes in This Workflow

### Generation Nodes

- **Nano Banana Pro Image**
  - Generates images with Nano Banana Pro settings.
  - Supports up to 8 optional reference images.

- **Nano Banana 2**
  - Generates images with Nano Banana 2 settings.
  - Supports up to 14 optional reference images (including single-image and two-image setups).
  - Supports optional Google web-search grounding.

---

### Utility / Helper Nodes

- **System Prompt Selector**
  - Lets you keep reusable prompt templates.
  - Combines your selected system prompt with a user prompt.
  - Useful for creating consistent, full prompts quickly.

---

## Workflow Scenarios Included

- **Single image test**
  - Quick baseline generation.

- **Two image test**
  - Compare how each model handles multi-reference guidance.

- **Nano Banana 2 with web search**
  - Demonstrates grounding-enabled prompt generation behavior.

- **Side-by-side model comparison**
  - Run similar prompts through Pro and Nano Banana 2 for direct visual comparison.

---

## Notes on Usage & Debugging

- All requests are sent to Kie.ai:
  [https://kie.ai](https://kie.ai?ref=e7565cf24a7fad4586341a87eaf21e42)

- If a node fails or times out in ComfyUI, check account logs:
  [https://kie.ai/logs](https://kie.ai/logs)

- Logs are usually the fastest way to verify whether a generation completed successfully.

---

## Issues & Feedback

To report issues or request improvements:
[https://github.com/gateway/ComfyUI-Kie-API/issues](https://github.com/gateway/ComfyUI-Kie-API/issues)

Please include:
- node name
- parameters used
- error message or log excerpt

---

## Development & Sponsorship

Development is supported by **Dreaming Computers**:
[https://dreamingcomputers.com](https://dreamingcomputers.com)
