# Reference Memo - ImageEdit Smoke / Fixture / Test Matrix Foundation

Date: 2026-04-23
Scope: `F169` image-edit regression, fixture, and live-smoke foundation

## Current shipped reality after `F168`

- `F168` already merged image-edit onto the canonical `img2img` surface.
- The backend request contract is already image-edit aware:
  - `rookieui/services/img2img.py`
  - official image-edit profiles force `mode="img2img"` on the public surface
  - runtime still routes through `execution_mode="edit"`
  - `reference_images` plus `main_reference_index` are accepted
  - manifest-derived `max_direct_references` is enforced
  - mask input is explicitly cleared for image-edit profiles
- The runtime translation layer already has strong per-family proof:
  - `tests/test_img2img_translation.py`
  - covers:
    - qwen image-edit
    - qwen multi-LoRA chain depth
    - firered multi-reference encode path
    - firered lightning template-LoRA path
    - flux kontext multi-reference path
    - flux2 advanced sampler path
    - klein KV cached reference path
    - longcat image-edit path

## Current live-smoke gap

- `scripts/run_live_smoke_tests.py`
  - catalog mode validates preset/model/text-encoder/VAE/LoRA readiness
  - execute mode can submit image-edit payloads
  - but `_build_edit_payload()` still only emits a single top-level `image_data` payload and does not pin ordered `reference_images`
  - there is no dedicated image-edit dry-run validation lane
  - there is no live-smoke assertion for:
    - ordered `reference_images`
    - non-zero `main_reference_index`
    - template-owned multi-LoRA chain depth
    - no-mask workflow topology on the live route payload
- `tests/test_live_smoke_tests.py`
  - currently has only narrow image-edit smoke proof:
    - selector resolution for `qwen_image_edit`
    - catalog acceptance for one qwen-edit preset
    - execute routing for `qwen_image_edit` through `/rookieui/generate/img2img`
  - it does not yet pin the broader first-wave image-edit matrix

## Current frontend regression gap

- `web/tests/rookieui_extension.test.js`
  - already proves:
    - image-edit profiles appear in the img2img preset list
    - mask controls are suppressed
    - reference section visibility changes with profile
  - but it stops at UI state and missing-source validation
  - it does not assert submitted `reference_images` / `main_reference_index`
- `tests/e2e/specs/bootstrap.spec.js`
  - already proves:
    - visible `Edit` mode is gone
    - qwen edit vs kontext edit changes reference-card visibility
  - but it only submits a classic inpaint payload afterward
  - it does not assert multi-reference image-edit submit payloads

## Existing canonical metadata to reuse

- `rookieui/contracts/family_template_manifest.py`
  - `image_edit_profile`
  - `request_contract_surface`
  - `reference_input_mode`
  - `max_direct_references`
  - `encoder_family`
  - `template_lora_chain_mode`
  - `official_template_lora_label`
- This means `F169` should stay manifest-driven rather than creating a second hand-maintained smoke matrix.

## Delivery implication for `F169`

- Do not reopen adapter/runtime family work already covered by `F165` / `F167`.
- Add a direct image-edit validation lane and fixture matrix that proves:
  - ordered reference payload construction
  - profile-specific direct-reference caps
  - template-owned LoRA chain expectations
  - no-mask img2img contract behavior
  - frontend request serialization for multi-reference profiles
- Keep `R172` focused on final live-host acceptance closure, not on inventing the first reusable image-edit smoke/test foundation there.
