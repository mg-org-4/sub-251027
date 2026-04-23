# Command Log - Official ImageEdit Manifest/Profile Matrix Expansion

Date: 2026-04-23
Environment: Windows PowerShell, repo-local `.venv`, branch `dev`

## Commands

1. `.venv\Scripts\python.exe -m unittest tests.test_model_family_registry tests.test_capabilities tests.test_img2img_translation`
2. `npm run test:unit -- web/tests/rookieui_api.test.js`
3. `powershell -File scripts/run_full_tests_windows.ps1`
4. `node -e "const fs=require('fs'); const path=require('path'); const crypto=require('crypto'); const repo=process.cwd(); function read(rel){return fs.readFileSync(path.join(repo, rel),'utf8');} function list(root){const out=[]; function walk(dir){for(const entry of fs.readdirSync(dir,{withFileTypes:true})){const abs=path.join(dir, entry.name); const rel=path.relative(repo, abs).replace(/\\\\/g,'/'); if(entry.isDirectory()){ if(rel==='web/tests') continue; walk(abs); continue;} if(!/\\.(js|css)$/i.test(entry.name)) continue; if(rel==='web/rookieui_asset_revision.js') continue; out.push(rel);} } walk(root); return out.sort(); } const hash=crypto.createHash('sha1'); for(const rel of list(path.join(repo,'web'))){ hash.update(rel); hash.update('\\0'); hash.update(read(rel)); hash.update('\\0'); } console.log(hash.digest('hex').slice(0,10));"`
5. `npm run test:unit -- web/tests/rookieui_frontend_architecture.test.js web/tests/rookieui_api.test.js`
6. `powershell -File scripts/run_full_tests_windows.ps1`

## Manual review checkpoints

- Verified the new manifest metadata is explicit and manifest-backed instead of inferred from profile ids.
- Verified presets and backend capabilities both retain the new fields, so later runtime/UI items can read the same contract.
- Verified `available_surface_flows` intentionally remain unchanged for `qwen_image_edit`, which preserves the planned sequencing boundary before `F168`.
- Verified the frontend asset revision token was refreshed only because shipped frontend modules changed, matching the existing cache-busting guardrail rather than bypassing it.
