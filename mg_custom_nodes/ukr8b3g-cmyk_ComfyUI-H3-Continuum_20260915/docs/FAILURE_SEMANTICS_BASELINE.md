# R0 Failure Semantics Baseline

Date: 2026-09-14  
Scope: current V3.8 Python-exception behavior  
Storage root: pytest `tmp_path` only

## Isolation contract

すべてのfailure injectionは一時Run Storage rootで実施した。

```text
real h3_continuum storage touched: NO
existing Take / revision touched: NO
user backend 8188 touched: NO
GPU used: NO
public workflow changed: NO
```

このbaselineは、catch可能なPython exceptionが`RunStorageController.__exit__()`へ到達する場合を記録する。OS強制終了、電源断、process killの挙動を成功扱いしない。

## Results

| Injection point | Committed records after cleanup | Manifest status | Canonical pointer | Current observable behavior |
|---|---:|---|---|---|
| before Sampling | 0 | `interrupted` | none | errorを記録し、review pause metadataを除去 |
| after Sampling / before chunk commit | 0 | `interrupted` | none | sampled resultは永続化されない |
| after first entry of terminal physical group | 1 | `interrupted` | none | terminal pairの片側raw/recordがinterrupted revision内に見える |
| before manifest update (one-shot exception) | 1 | `interrupted` | none | in-memory recordを`__exit__()`が再度manifestへ書き、interruptedとして回収 |
| after manifest update | 1 | `interrupted` | none | committed recordを保持してinterrupted化 |
| before output projection | 1 | `interrupted` | none | recordを保持してinterrupted化 |
| during output projection | 1 | `interrupted` | none | recordを保持してinterrupted化 |
| before storage finalize | 1 | `interrupted` | none | in-progress recordを保持してinterrupted化 |
| before canonical pointer update (first `_write_project`) | 1 | `interrupted` | none | finalize途中のmanifestをexception cleanupがinterruptedへ戻し、2回目のproject writeはcanonicalを設定しない |

Assertions are implemented in:

```text
tests/test_runtime_redesign_r0_characterization.py
```

## Current write ordering

### `commit_chunk()`

```text
validate one logical entry
→ write temporary safetensors
→ fsync temporary
→ os.replace to chunk_000N.safetensors
→ fsync chunks directory
→ construct one record
→ replace manifest chunks in memory
→ write manifest
```

この処理はphysical group単位ではなくlogical entry単位である。terminal merged pairでも、1件目commitと2件目commitの間に例外が起きる窓がある。

### `finalize()`

```text
validate committed prefix and Session length
→ set status / review metadata
→ write manifest
→ build branch provenance
→ write manifest again
→ write project with set_canonical=True
```

Python exceptionが上位のstorage scopeへ戻れば、`__exit__()`はstatusを`interrupted`へ変更し、canonical指定なしでprojectを再書込する。

## Process-crash limitations

このR0 testは、次を保証しない。

- `os.replace(raw)`直後、manifest更新前にprocessが消失した場合のorphan raw回収
- 既存`chunk_000N.safetensors`を上書き中にprocessが消失した場合の旧manifest/raw整合性
- terminal pairの両entryを不可分にpublishすること
- manifest provenance書込後、project canonical更新前にprocessが消失した場合の再起動時selection

これらはR5のprocess-crash consistency対象である。R0でwrite orderやschemaを変更していない。

## R5-approved delta boundary

将来R5で許可される差分はV9どおり次に限定する。

- physical group全entryを先にvalidate
- immutable transaction raw filenames
- raw durabilityとSHA verification後にmanifestを1回でswitch
- `commit_group()`はcanonical pointerを更新しない
- canonical pointerはsuccessful finalize後だけ更新

Sampling結果、Session/State/Assembly、Review user-visible semantics、Run Storage v3 readabilityは変更しない。

