import { $ as e, D as t, L as n, _t as r, a as i, et as a, ft as o, gt as s, ht as c, p as l, r as u } from "./events-BZGevJ-5.js";
//#region ui/app/settingsStore.ts
var d = "mjrSettings", f = "mjrMinimapSettings", p = new Set([
	"POST",
	"PUT",
	"DELETE",
	"PATCH"
]), m = 2e4, h = 3e5, ee = 3, g = 400, _ = "/mjr/am/settings/security/bootstrap-token", v = "/mjr/am/", y = /* @__PURE__ */ new Map();
function b(e) {
	return new Promise((t) => setTimeout(t, e));
}
function x(e) {
	return p.has(String(e || "").trim().toUpperCase());
}
function te(e) {
	try {
		let t = String(e || "").trim();
		if (!t) return "";
		if (t.startsWith("http://") || t.startsWith("https://")) {
			let e = typeof globalThis < "u" && globalThis?.location?.origin ? String(globalThis.location.origin) : "http://localhost";
			return new URL(t, e).pathname || "";
		}
		return t.split("?")[0] || "";
	} catch {
		return "";
	}
}
function ne(e) {
	return te(e).startsWith(v);
}
function S(e) {
	return te(e) === _;
}
function re(e) {
	try {
		if (!e) return !1;
		let t = String(e.name || "");
		if (t === "AbortError") return !1;
		let n = String(e.message || "").toLowerCase();
		return t === "TypeError" ? n.includes("failed to fetch") || n.includes("networkerror") || n.includes("load failed") || n.includes("fetch") || n.includes("network") : n.includes("fetch") || n.includes("network") || n.includes("failed");
	} catch {
		return !1;
	}
}
function ie(e = {}) {
	try {
		let t = Number(e?.timeoutMs);
		return Number.isFinite(t) ? Math.max(1e3, Math.min(h, Math.floor(t))) : m;
	} catch {
		return m;
	}
}
function ae(e = {}) {
	let t = e?.signal || null;
	if (typeof AbortController > "u") return {
		signal: t || void 0,
		timeoutMs: ie(e),
		cleanup: () => {}
	};
	let n = ie(e), r = new AbortController(), i = null, a = () => {
		try {
			i &&= (clearTimeout(i), null);
		} catch (e) {
			console.debug?.(e);
		}
		try {
			r.abort();
		} catch (e) {
			console.debug?.(e);
		}
	};
	try {
		i = setTimeout(() => {
			try {
				r.abort();
			} catch (e) {
				console.debug?.(e);
			}
		}, n);
	} catch (e) {
		console.debug?.(e);
	}
	try {
		t && (t.aborted ? a() : t.addEventListener("abort", a, { once: !0 }));
	} catch (e) {
		console.debug?.(e);
	}
	return {
		signal: r.signal,
		timeoutMs: n,
		cleanup: () => {
			try {
				i && clearTimeout(i);
			} catch (e) {
				console.debug?.(e);
			}
			try {
				t && t.removeEventListener("abort", a);
			} catch (e) {
				console.debug?.(e);
			}
		}
	};
}
function oe(e, t, n = {}) {
	let r = String(e || "GET").trim().toUpperCase(), i = String(t || "").trim();
	return !r || !i ? "" : `${r}:${i}:timeout=${ie(n)}`;
}
function se(e, t) {
	let n = String(e || "").trim();
	if (!n) return t();
	if (y.has(n)) return y.get(n);
	let r = Promise.resolve().then(() => t()).finally(() => {
		try {
			y.delete(n);
		} catch (e) {
			console.debug?.(e);
		}
	});
	return y.set(n, r), r;
}
function ce(e, t, n) {
	if (x(t)) try {
		e instanceof Headers ? e.has("X-Requested-With") || e.set("X-Requested-With", "XMLHttpRequest") : e["X-Requested-With"] ||= "XMLHttpRequest";
	} catch (e) {
		console.debug?.(e);
	}
	try {
		e instanceof Headers ? e.has("X-MJR-OBS") || e.set("X-MJR-OBS", n ? "on" : "off") : "X-MJR-OBS" in e || (e["X-MJR-OBS"] = n ? "on" : "off");
	} catch (e) {
		console.debug?.(e);
	}
}
function le(e, t) {
	if (t) try {
		e instanceof Headers ? (e.has("X-MJR-Token") || e.set("X-MJR-Token", t), e.has("Authorization") || e.set("Authorization", `Bearer ${t}`)) : ("X-MJR-Token" in e || (e["X-MJR-Token"] = t), "Authorization" in e || (e.Authorization = `Bearer ${t}`));
	} catch (e) {
		console.debug?.(e);
	}
}
async function ue(e, t, n, r, { ensureWriteAuthToken: i, normalizeWriteAuthFailure: a, fetchAPI: o, options: s, retryCount: c }) {
	let l = e.headers.get("content-type") || "";
	if (!l.includes("application/json")) return !r && x(n) && ne(t) && !S(t) && Number(e.status || 0) === 401 && await i({
		force: !0,
		allowCookieRefresh: !0
	}) ? await o(t, {
		...s,
		_authRetryDone: !0
	}, c) : {
		ok: !1,
		error: `Server returned non-JSON response (${e.status})`,
		code: "INVALID_RESPONSE",
		status: e.status,
		content_type: l,
		data: null
	};
	let u = await e.json().catch((e) => (console.debug?.("[MJR API] JSON parse error:", e), null));
	if (typeof u != "object" || !u) return {
		ok: !1,
		error: "Invalid response structure",
		code: "INVALID_RESPONSE",
		status: e.status,
		data: null
	};
	if (!("status" in u)) try {
		u.status = e.status;
	} catch (e) {
		console.debug?.(e);
	}
	return !r && x(n) && !S(t) && !u?.ok && (String(u?.code || "").toUpperCase() === "AUTH_REQUIRED" || Number(u?.status || 0) === 401) && await i({
		force: !0,
		allowCookieRefresh: !0
	}) ? await o(t, {
		...s,
		_authRetryDone: !0
	}, c) : (x(n) && ne(t) && !S(t) && (u = a(u)), u);
}
function de({ readObsEnabled: e = () => !1, readAuthToken: t = () => "", ensureWriteAuthToken: n = async () => "", normalizeWriteAuthFailure: r = (e) => e, trackApiCall: i = null } = {}) {
	async function a(o, s = {}, c = 0) {
		let l = typeof performance < "u" ? performance.now() : Date.now(), u = ae(s), d = null;
		try {
			let i = (s.method || "GET").toUpperCase(), l = !!s?._authRetryDone, f = typeof Headers < "u" ? new Headers(s.headers || {}) : { ...s.headers };
			ce(f, i, !!e());
			let p = t();
			if (!p && x(i) && ne(o) && !S(o)) {
				try {
					await n();
				} catch (e) {
					console.debug?.(e);
				}
				p = t();
			}
			le(f, p);
			let m = {
				...s,
				headers: f,
				signal: u.signal
			};
			try {
				delete m._authRetryDone, delete m.timeoutMs;
			} catch (e) {
				console.debug?.(e);
			}
			return d = await ue(await fetch(o, m), o, i, l, {
				ensureWriteAuthToken: n,
				normalizeWriteAuthFailure: r,
				fetchAPI: a,
				options: s,
				retryCount: c
			}), d;
		} catch (e) {
			try {
				if (String(e?.name || "") === "AbortError") return s?.signal && s.signal.aborted ? {
					ok: !1,
					error: "Aborted",
					code: "ABORTED",
					data: null
				} : {
					ok: !1,
					error: `Request timed out after ${u.timeoutMs}ms`,
					code: "TIMEOUT",
					data: null,
					timeout_ms: u.timeoutMs
				};
			} catch (e) {
				console.debug?.(e);
			}
			if (c < ee && re(e)) {
				try {
					await b(g * (c + 1));
				} catch (e) {
					console.debug?.(e);
				}
				return await a(o, s, c + 1);
			}
			return {
				ok: !1,
				error: e?.message || String(e || "Network error"),
				code: "NETWORK_ERROR",
				data: null,
				retries: c
			};
		} finally {
			try {
				let e = (typeof performance < "u" ? performance.now() : Date.now()) - l;
				typeof i == "function" ? i(e, !d?.ok) : typeof window < "u" && window.MajoorMetrics && window.MajoorMetrics.trackApiCall(e, !d?.ok);
			} catch (e) {
				console.debug?.(e);
			}
			try {
				u.cleanup?.();
			} catch (e) {
				console.debug?.(e);
			}
		}
	}
	async function o(e, t = {}) {
		return se(t?.dedupe === !1 ? "" : String(t?.dedupeKey || "").trim() || oe("GET", e, t), () => a(e, {
			...t,
			method: "GET"
		}));
	}
	async function s(e, t, n = {}) {
		return a(e, {
			...n,
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				...n.headers
			},
			body: JSON.stringify(t)
		});
	}
	return {
		fetchAPI: a,
		get: o,
		post: s
	};
}
//#endregion
//#region ui/utils/ttlCache.ts
function C({ ttlMs: e = 0, maxSize: t = 100, now: n = () => Date.now() } = {}) {
	let r = /* @__PURE__ */ new Map();
	function i() {
		try {
			let e = Number(n());
			return Number.isFinite(e) ? e : Date.now();
		} catch {
			return Date.now();
		}
	}
	function a() {
		try {
			let t = Number(typeof e == "function" ? e() : e);
			return Number.isFinite(t) ? Math.max(0, Math.floor(t)) : 0;
		} catch {
			return 0;
		}
	}
	function o() {
		try {
			let e = Number(t);
			return Number.isFinite(e) ? Math.max(1, Math.floor(e)) : 1;
		} catch {
			return 1;
		}
	}
	function s(e, t, n) {
		return e ? n > 0 ? t - Number(e.at || 0) > n : !1 : !0;
	}
	function c(e = i(), t = a()) {
		if (t > 0) for (let [n, i] of r.entries()) s(i, e, t) && r.delete(n);
	}
	function l() {
		let e = o();
		for (; r.size > e;) {
			let e = r.keys().next().value;
			if (e === void 0) break;
			r.delete(e);
		}
	}
	function u(e, t) {
		r.delete(e), r.set(e, t);
	}
	return {
		get(e) {
			let t = i(), n = a();
			c(t, n);
			let o = r.get(e);
			if (o) {
				if (s(o, t, n)) {
					r.delete(e);
					return;
				}
				return o.value;
			}
		},
		has(e) {
			return this.get(e) !== void 0;
		},
		set(e, t, n = {}) {
			let o = Number(n?.at), s = {
				value: t,
				at: Number.isFinite(o) ? o : i()
			};
			return r.has(e) ? u(e, s) : r.set(e, s), c(i(), a()), l(), t;
		},
		delete(e) {
			return r.delete(e);
		},
		clear() {
			r.clear();
		},
		prune() {
			return c(i(), a()), l(), r.size;
		},
		keys() {
			return this.prune(), Array.from(r.keys());
		},
		entries() {
			return this.prune(), Array.from(r.entries()).map(([e, t]) => [e, t.value]);
		},
		get size() {
			return this.prune(), r.size;
		}
	};
}
//#endregion
//#region ui/features/panel/messages/toastHistory.ts
var fe = "mjr_toast_history_v1", pe = "mjr_toast_history_last_read_v1", me = 60, he = "mjr:toast-history-changed", w = null;
function T(e) {
	return String(e || "").trim();
}
function ge(e) {
	let t = T(e).toLowerCase();
	return t === "warn" ? "warning" : t === "danger" ? "error" : t || "info";
}
function _e(e) {
	let t = Number(e);
	return Number.isFinite(t) ? t : null;
}
function ve(e) {
	if (!e || typeof e != "object") return null;
	let t = Number(e.percent), n = Number.isFinite(t) ? Math.max(0, Math.min(100, Math.round(t))) : null, r = Number(e.current), i = Number(e.total), a = Number.isFinite(r) ? Math.max(0, Math.floor(r)) : null, o = Number.isFinite(i) ? Math.max(0, Math.floor(i)) : null, s = Number(e.indexed), c = Number(e.skipped), l = Number(e.errors), u = Number.isFinite(s) ? Math.max(0, Math.floor(s)) : null, d = Number.isFinite(c) ? Math.max(0, Math.floor(c)) : null, f = Number.isFinite(l) ? Math.max(0, Math.floor(l)) : null, p = T(e.label);
	return n === null && a === null && o === null && u === null && d === null && f === null && !p ? null : {
		percent: n,
		current: a,
		total: o,
		indexed: u,
		skipped: d,
		errors: f,
		label: p
	};
}
function ye(e, t, n) {
	return e && t ? `${e}: ${t}` : t || e || n || "";
}
function be(e, t = "info", n = null) {
	if (!e || typeof e != "object") return null;
	let r = T(e.title || e.summary), i = T(e.detail), a = T(e.message || ye(r, i, T(e.fallbackMessage)));
	if (!a) return null;
	let o = _e(e.durationMs ?? e.duration ?? n), s = Number(e.createdAt), l = Number.isFinite(s) && s > 0 ? s : Date.now(), u = typeof e.persistent == "boolean" ? e.persistent : !(Number.isFinite(o) && (o ?? 0) > 0);
	return {
		id: T(e.id) || c(`th-${l}-`, 4),
		message: a,
		title: r,
		detail: i,
		type: ge(e.type || t),
		createdAt: l,
		durationMs: o,
		persistent: u,
		source: T(e.source),
		trackId: T(e.trackId),
		status: T(e.status),
		operation: T(e.operation),
		progress: ve(e.progress),
		forceStore: !!e.forceStore,
		actionLabel: T(e.actionLabel),
		actionUrl: T(e.actionUrl)
	};
}
function xe() {
	if (w === null) try {
		let e = localStorage.getItem(fe), t = e ? JSON.parse(e) : [];
		w = Array.isArray(t) ? t.map((e) => {
			if (e && typeof e == "object") return be(e);
			let t = T(e);
			return t ? be({ message: t }) : null;
		}).filter(Boolean) : [];
	} catch {
		w = [];
	}
}
function Se() {
	try {
		localStorage.setItem(fe, JSON.stringify(w));
	} catch {}
}
function Ce() {
	try {
		window.dispatchEvent(new CustomEvent(he));
	} catch {}
}
function we() {
	try {
		return Number(localStorage.getItem(pe)) || 0;
	} catch {
		return 0;
	}
}
function Te(e) {
	try {
		localStorage.setItem(pe, String(Number(e) || 0));
	} catch {}
}
function Ee(e, t, n) {
	xe();
	let r = be(e && typeof e == "object" ? e : {
		message: T(e),
		type: t,
		durationMs: n
	}, t, n);
	if (!r || !r.forceStore && !r.trackId && r.type === "info" && Number.isFinite(r.durationMs) && r.durationMs != null && r.durationMs > 0 && r.durationMs < 2500) return;
	let i = String(r.trackId || "").trim();
	if (i) {
		let e = w.findIndex((e) => String(e?.trackId || "").trim() === i);
		if (e >= 0) {
			let t = w[e] || {};
			w.splice(e, 1), r.id = String(t.id || r.id || "").trim() || r.id;
		}
	}
	w.unshift(r), w.length > me && (w = w.slice(0, me)), Se(), Ce();
}
function De() {
	return xe(), w.map((e) => ({ ...e }));
}
function Oe() {
	xe();
	let e = we();
	return w.filter((t) => t.createdAt > e).length;
}
function ke() {
	Te(Date.now()), Ce();
}
function Ae() {
	xe(), w = [], Se(), Te(Date.now()), Ce();
}
//#endregion
//#region ui/app/toast.ts
function je(e) {
	let t = String(e || "info").trim().toLowerCase();
	return t === "warn" ? "warning" : t === "danger" ? "error" : t === "success" || t === "warning" || t === "error" ? t : "info";
}
function Me(e) {
	if (typeof e == "string") return e;
	if (e && typeof e == "object") {
		let t = String(e.summary || "").trim(), n = String(e.detail || e.message || "").trim();
		if (t && n) return `${t}: ${n}`;
		if (n) return n;
		if (t) return t;
	}
	try {
		return String(e?.message || e || "").trim() || "Unknown message";
	} catch {
		return "Unknown message";
	}
}
function Ne(e, t, n, r) {
	let i = r?.history && typeof r.history == "object" ? r.history : null, a = {
		persistent: !(Number.isFinite(Number(n)) && Number(n) > 0),
		source: String(i?.source || r?.source || "").trim(),
		trackId: String(i?.trackId || "").trim(),
		status: String(i?.status || "").trim(),
		operation: String(i?.operation || "").trim(),
		progress: i?.progress && typeof i.progress == "object" ? { ...i.progress } : null,
		forceStore: !!i?.forceStore,
		actionLabel: String(i?.actionLabel || "").trim(),
		actionUrl: String(i?.actionUrl || "").trim()
	};
	return e && typeof e == "object" ? (a.title = String(i?.title || e.summary || "").trim(), a.detail = String(i?.detail || e.detail || e.message || "").trim(), a.message = Me(e), a) : (a.title = String(i?.title || "").trim(), a.detail = String(i?.detail || "").trim(), a.message = Me(e), a);
}
function Pe(e, t = "info", n, r) {
	try {
		let i = Ne(e, t, n, r);
		i.forceStore = !0, Ee(i, t, n ?? void 0);
	} catch {}
}
function Fe(e) {
	switch (e) {
		case "success": return 2e3;
		case "info": return 3e3;
		case "warning": return 4e3;
		case "error": return 5e3;
		default: return 5e3;
	}
}
function Ie(e) {
	if (typeof e != "string") return e;
	let t = e.trim(), n = {
		"Failed to update rating": l("toast.ratingUpdateFailed", "Failed to update rating"),
		"Error updating rating": l("toast.ratingUpdateError", "Error updating rating"),
		"Rating cleared": l("toast.ratingCleared", "Rating cleared"),
		"Failed to update tags": l("toast.tagsUpdateFailed", "Failed to update tags"),
		"Tags updated": l("toast.tagsUpdated", "Tags updated"),
		"Failed to toggle watcher": l("toast.watcherToggleFailed", "Failed to toggle watcher"),
		"No valid assets selected.": l("toast.noValidAssetsSelected", "No valid assets selected."),
		"Name collision in current view": l("toast.nameCollisionInView", "Name collision in current view"),
		"Failed to create collection.": l("toast.failedCreateCollectionDot", "Failed to create collection."),
		"Failed to add assets to collection.": l("toast.failedAddAssetsToCollection", "Failed to add assets to collection."),
		"Failed to remove from collection.": l("toast.removeFromCollectionFailed", "Failed to remove from collection."),
		"Collection created": l("toast.collectionCreated", "Collection created"),
		"Added to collection": l("toast.addedToCollection", "Added to collection"),
		"Removed from collection": l("toast.removedFromCollection", "Removed from collection"),
		"File renamed successfully!": l("toast.fileRenamedSuccess", "File renamed successfully!"),
		"Failed to rename file": l("toast.fileRenameFailed", "Failed to rename file"),
		"Failed to rename file.": l("toast.fileRenameFailed", "Failed to rename file."),
		"File deleted successfully!": l("toast.fileDeletedSuccess", "File deleted successfully!"),
		"Failed to delete file.": l("toast.fileDeleteFailed", "Failed to delete file."),
		"Failed to delete file. ": l("toast.fileDeleteFailed", "Failed to delete file."),
		"File deleted": l("toast.deleted", "File deleted"),
		"File renamed": l("toast.renamed", "File renamed"),
		"Folder created": l("toast.folderCreated", "Folder created"),
		"Folder renamed": l("toast.folderRenamed", "Folder renamed"),
		"Folder moved": l("toast.folderMoved", "Folder moved"),
		"Folder deleted": l("toast.folderDeleted", "Folder deleted"),
		"Failed to create folder": l("toast.createFolderFailed", "Failed to create folder"),
		"Failed to rename folder": l("toast.renameFolderFailed", "Failed to rename folder"),
		"Failed to move folder": l("toast.moveFolderFailed", "Failed to move folder"),
		"Failed to delete folder": l("toast.deleteFolderFailed", "Failed to delete folder"),
		"Failed to pin folder": l("toast.pinFolderFailed", "Failed to pin folder"),
		"Failed to unpin folder": l("toast.unpinFolderFailed", "Failed to unpin folder"),
		"Folder pinned as browser root": l("toast.folderPinnedAsBrowserRoot", "Folder pinned as browser root"),
		"Folder added": l("toast.folderAdded", "Folder added"),
		"Folder removed": l("toast.folderRemoved", "Folder removed"),
		"Folder linked successfully": l("toast.folderLinked", "Folder linked successfully"),
		"An error occurred while adding the custom folder": l("toast.errorAddingFolder", "An error occurred while adding the custom folder"),
		"An error occurred while removing the custom folder": l("toast.errorRemovingFolder", "An error occurred while removing the custom folder"),
		"Failed to add custom folder": l("toast.failedAddFolder", "Failed to add custom folder"),
		"Failed to remove custom folder": l("toast.failedRemoveFolder", "Failed to remove custom folder"),
		"Native folder browser unavailable. Please enter path manually.": l("toast.nativeBrowserUnavailable", "Native folder browser unavailable. Please enter path manually."),
		"Opened in folder": l("toast.openedInFolder", "Opened in folder"),
		"Failed to open folder": l("toast.openFolderFailed", "Failed to open folder"),
		"Failed to open folder.": l("toast.openFolderFailed", "Failed to open folder."),
		"File path copied to clipboard": l("toast.pathCopied", "File path copied to clipboard"),
		"Failed to copy path": l("toast.pathCopyFailed", "Failed to copy path"),
		"Failed to copy to clipboard": l("toast.copyClipboardFailed", "Failed to copy to clipboard"),
		"No file path available for this asset.": l("toast.noFilePath", "No file path available for this asset."),
		"Failed to refresh metadata.": l("toast.metadataRefreshFailed", "Failed to refresh metadata."),
		"Metadata refreshed": l("toast.metadataRefreshed", "Metadata refreshed"),
		"Duplicate analysis started": l("toast.dupAnalysisStarted", "Duplicate analysis started"),
		"Tags merged": l("toast.tagsMerged", "Tags merged"),
		"Duplicates deleted": l("toast.duplicatesDeleted", "Duplicates deleted"),
		"Rescanning file...": l("toast.rescanningFile", "Rescanning file..."),
		"Metadata enrichment complete": l("toast.enrichmentComplete", "Metadata enrichment complete"),
		"Playback speed is available for video media only": l("toast.playbackVideoOnly", "Playback speed is available for video media only"),
		"DB backup saved": l("toast.dbSaveSuccess", "DB backup saved"),
		"Failed to save DB backup": l("toast.dbSaveFailed", "Failed to save DB backup"),
		"DB restore started": l("toast.dbRestoreStarted", "DB restore started"),
		"Failed to restore DB backup": l("toast.dbRestoreFailed", "Failed to restore DB backup"),
		"Select a DB backup first": l("toast.dbRestoreSelect", "Select a DB backup first"),
		"Stopping running workers": l("toast.dbRestoreStopping", "Stopping running workers"),
		"Unlocking and resetting database": l("toast.dbRestoreResetting", "Unlocking and resetting database"),
		"Recreating database": l("toast.dbRestoreReplacing", "Recreating database"),
		"Replacing database files": l("toast.dbRestoreReplacing", "Replacing database files"),
		"Restarting scan": l("toast.dbRestoreRescan", "Restarting scan"),
		"Deleting database and rebuilding...": l("toast.dbDeleteTriggered", "Deleting database and rebuilding..."),
		"Database deleted and rebuilt. Files are being reindexed.": l("toast.dbDeleteSuccess", "Database deleted and rebuilt. Files are being reindexed."),
		"Failed to delete database": l("toast.dbDeleteFailed", "Failed to delete database"),
		"Database backup restored": l("toast.dbRestoreSuccess", "Database backup restored"),
		"Index reset started. Files will be reindexed in the background.": l("toast.resetStarted", "Index reset started. Files will be reindexed in the background."),
		"Failed to reset index": l("toast.resetFailed", "Failed to reset index"),
		"Reset triggered: Reindexing all files...": l("toast.resetTriggered", "Reset triggered: Reindexing all files..."),
		"Reset failed - database is corrupted. Use the \"Delete DB\" button to force-delete and rebuild.": l("toast.resetFailedCorrupt", "Reset failed - database is corrupted. Use the \"Delete DB\" button to force-delete and rebuild."),
		"Scan started": l("toast.scanStarted", "Scan started"),
		"Scan complete": l("toast.scanComplete", "Scan complete"),
		"Scan failed": l("toast.scanFailed", "Scan failed"),
		"Permission denied": l("toast.permissionDenied", "Permission denied"),
		"Language changed. Reload the page for full effect.": l("toast.languageChanged", "Language changed. Reload the page for full effect."),
		"Tag added": l("toast.tagAdded", "Tag added"),
		"Tag removed": l("toast.tagRemoved", "Tag removed"),
		"Rating saved": l("toast.ratingSaved", "Rating saved"),
		"Failed to create collection": l("toast.failedCreateCollection", "Failed to create collection"),
		"Failed to delete collection": l("toast.failedDeleteCollection", "Failed to delete collection")
	};
	if (n[t]) return n[t];
	let r;
	if (r = t.match(/Rating set to (\d+) star(?:s)?/i), r) return l("toast.ratingSetN", "Rating set to {n} stars", { n: Number(r[1]) });
	if (r = t.match(/Downloading (.+?)\.\.\./i), r) return l("toast.downloadingFile", "Downloading {filename}...", { filename: r[1] });
	if (r = t.match(/Playback ([0-9]+(?:\.[0-9]+)?)x/i), r) return l("toast.playbackRate", "Playback {rate}x", { rate: Number(r[1]).toFixed(2) });
	if (r = t.match(/Metadata refreshed(?:\s*(.*))?/i), r) return l("toast.metadataRefreshed", "Metadata refreshed{suffix}", { suffix: r[1] ? " (" + r[1] + ")" : "" });
	if (r = t.match(/Error renaming(?: file)?:\s*(.+)/i), r) return l("toast.errorRenaming", "Error renaming file: {error}", { error: r[1] });
	if (r = t.match(/Error deleting(?: files?| file)?:\s*(.+)/i), r) return l("toast.errorDeleting", "Error deleting file: {error}", { error: r[1] });
	if (r = t.match(/Tag merge failed:\s*(.+)/i), r) return l("toast.tagMergeFailed", "Tag merge failed: {error}", { error: r[1] });
	if (r = t.match(/Delete failed:\s*(.+)/i), r) return l("toast.deleteFailed", "Delete failed: {error}", { error: r[1] });
	if (r = t.match(/Analysis not started:\s*(.+)/i), r) return l("toast.analysisNotStarted", "Analysis not started: {error}", { error: r[1] });
	if (r = t.match(/(\d+)\s+files deleted successfully!/i), r) return l("toast.filesDeletedSuccessN", "{n} files deleted successfully!", { n: Number(r[1]) });
	if (r = t.match(/(\d+)\s+files deleted,\s+(\d+)\s+failed\./i), r) return l("toast.filesDeletedPartial", "{success} files deleted, {failed} failed.", {
		success: Number(r[1]),
		failed: Number(r[2])
	});
	if (r = t.match(/(\d+)\s+files?\s+deleted/i), r) return l("toast.filesDeletedShort", "{n} files deleted", { n: Number(r[1]) });
	if (r = t.match(/(\d+)\s+deleted,\s+(\d+)\s+failed/i), r) return l("toast.filesDeletedShortPartial", "{success} deleted, {failed} failed", {
		success: Number(r[1]),
		failed: Number(r[2])
	});
	if (r = t.match(/^(.+?)\s+copied to clipboard!$/i), r) return l("toast.copiedToClipboardNamed", "{name} copied to clipboard!", { name: r[1] });
	if (r = t.match(/Folder created:\s*(.+)/i), r) return l("toast.folderCreated", "Folder created: {name}", { name: r[1] });
	if (r = t.match(/Failed to create folder:\s*(.+)/i), r) return l("toast.createFolderFailedDetail", "Failed to create folder: {error}", { error: r[1] });
	if (r = t.match(/Failed to rename folder:\s*(.+)/i), r) return l("toast.renameFolderFailedDetail", "Failed to rename folder: {error}", { error: r[1] });
	if (r = t.match(/Failed to move folder:\s*(.+)/i), r) return l("toast.moveFolderFailedDetail", "Failed to move folder: {error}", { error: r[1] });
	if (r = t.match(/Failed to delete folder:\s*(.+)/i), r) return l("toast.deleteFolderFailedDetail", "Failed to delete folder: {error}", { error: r[1] });
	if (r = t.match(/Error removing from collection:\s*(.+)/i), r) return l("toast.removeFromCollectionError", "Error removing from collection: {error}", { error: r[1] });
	if (r = t.match(/^Failed to (.+)$/i), r) {
		let e = r[1].toLowerCase(), t = {
			"add folder": l("toast.failedAddFolder", "Failed to add custom folder"),
			"remove folder": l("toast.failedRemoveFolder", "Failed to remove custom folder"),
			"create collection": l("toast.failedCreateCollection", "Failed to create collection"),
			"delete collection": l("toast.failedDeleteCollection", "Failed to delete collection")
		};
		for (let [n, r] of Object.entries(t)) if (e.includes(n)) return r;
	}
	return t;
}
function Le(e, n = "info", r, i) {
	if (n = je(n), e = Ie(e), r ??= Fe(n), !i?.noHistory) try {
		Ee(Ne(e, n, r, i), n, r ?? void 0);
	} catch {}
	let a = !(Number.isFinite(Number(r)) && Number(r) > 0);
	try {
		let i = t();
		if (i && typeof i.add == "function") {
			let t = n;
			t === "warning" && (t = "warn");
			let o = l("manager.title"), s = e;
			if (typeof e == "object" && e?.summary) o = e.summary, s = e.detail || "";
			else if (typeof e != "string") try {
				s = e.message || String(e);
			} catch (e) {
				console.debug?.(e);
			}
			let c = {
				severity: t,
				summary: o,
				detail: s
			};
			a || (c.life = r), i.add(c);
			return;
		}
	} catch (e) {
		console.debug("Majoor: extensionManager.toast failed", e);
	}
	let o = globalThis?.app || (typeof window < "u" ? window?.app : null);
	if (o?.ui?.toast) try {
		return o.ui.toast(e, n);
	} catch (e) {
		console.debug("Native app.ui.toast failed", e);
	}
	console.warn("[Majoor Toast] Native toast API unavailable", {
		type: n,
		message: Me(e),
		duration: a ? 0 : r
	});
}
//#endregion
//#region ui/api/clientAuth.ts
var Re = 2e3, ze = 15e3, Be = 8e3, Ve = "__mjr_write_token", E = "token", He = null, D = null, Ue = null, O = C({
	ttlMs: Re,
	maxSize: 1
});
function We() {
	try {
		return String(sessionStorage?.getItem?.(Ve) || "").trim();
	} catch {
		return "";
	}
}
function Ge(e) {
	let t = String(e || "").trim();
	try {
		return t ? sessionStorage?.setItem?.(Ve, t) : sessionStorage?.removeItem?.(Ve), !0;
	} catch {
		return !1;
	}
}
function Ke() {
	try {
		let e = localStorage?.getItem?.(d), t = e ? JSON.parse(e) : {}, n = t && typeof t == "object" ? t : {}, r = n?.data && typeof n.data == "object" ? n.data : n;
		r?.security && typeof r.security == "object" && String(r.security.apiToken || "").trim() && (r.security.apiToken = "", localStorage?.setItem?.(d, JSON.stringify(n)));
	} catch (e) {
		console.debug?.(e);
	}
}
function qe() {
	try {
		O.delete(E);
	} catch (e) {
		console.debug?.(e);
	}
	Ge(""), Ke();
}
function Je() {
	let e = O.get(E);
	if (e !== void 0) return e;
	let t = Date.now(), n = We();
	if (n) return O.set(E, n, { at: t }), n;
	try {
		let e = localStorage?.getItem?.(d), n = e ? JSON.parse(e) : null, r = n?.data && typeof n.data == "object" ? n.data : n, i = String(r?.security?.apiToken || "").trim();
		if (i) {
			Ge(i);
			try {
				let e = n && typeof n == "object" ? n : {}, t = e?.data && typeof e.data == "object" ? e.data : e;
				t?.security && typeof t.security == "object" && (t.security.apiToken = "", localStorage?.setItem?.(d, JSON.stringify(e)), window?.dispatchEvent?.(new CustomEvent("mjr-settings-changed", { detail: { key: "security.apiToken" } })));
			} catch (e) {
				console.debug?.(e);
			}
		}
		return O.set(E, i, { at: t }), i;
	} catch {
		return O.set(E, "", { at: t }), "";
	}
}
function Ye(e) {
	let t = String(e || "").trim();
	if (!t) return !1;
	try {
		O.set(E, t), D = null, Ge(t), Ke();
		try {
			window?.dispatchEvent?.(new CustomEvent("mjr-settings-changed", { detail: { key: "security.apiToken" } }));
		} catch (e) {
			console.debug?.(e);
		}
		return !0;
	} catch {
		return !1;
	}
}
var Xe = /^[A-Za-z0-9._\-~+/]+=*$/;
function Ze(e) {
	let t = String(e || "").trim();
	return t ? Xe.test(t) ? Ye(t) : (console.debug?.("[MJR auth] Rejected malformed security token (invalid characters)"), !1) : !1;
}
function k(e = {}) {
	D = {
		code: String(e?.code || "").trim().toUpperCase(),
		error: String(e?.error || "").trim(),
		status: Number(e?.status || 0) || 0,
		at: Date.now()
	};
}
function Qe() {
	let e = D;
	if (!e) return null;
	let t = Date.now() - (Number(e.at || 0) || 0);
	return t < 0 || t > ze ? (D = null, null) : e;
}
function $e(e) {
	let t = Qe(), n = String(e?.code || "").trim().toUpperCase(), r = String(e?.error || "").trim(), i = String(t?.code || "").trim().toUpperCase(), a = String(t?.error || "").trim().toLowerCase(), o = r.toLowerCase();
	return i === "FORBIDDEN" && (a.includes("already configured") || a.includes("rotate-token")) ? l("toast.writeAuthConfiguredTokenRequired", "Write access requires the Majoor API token already configured on the server. Open Settings -> Security -> API Token and enter the matching token.") : i === "AUTH_REQUIRED" && (a.includes("sign in to comfyui") || a.includes("authenticated comfyui user")) ? l("toast.writeAuthSignInRequired", "Write access is blocked. Sign in to ComfyUI first, then retry so Majoor can bootstrap the remote session token automatically.") : i === "BOOTSTRAP_DISABLED" || i === "AUTH_REQUIRED" && a.includes("bootstrap") || n === "AUTH_REQUIRED" && o.includes("api token") ? l("toast.writeAuthBootstrapHelp", "Write access is blocked. Sign in to ComfyUI and retry so Majoor can bootstrap the remote session automatically, or set a Majoor API token in Settings -> Security.") : "";
}
function et(e) {
	let t = String(e || "").trim();
	if (!t) return;
	let n = Date.now(), r = Ue;
	if (!(r && r.message === t && n - (Number(r.at || 0) || 0) < Be)) {
		Ue = {
			message: t,
			at: n
		};
		try {
			Le({
				summary: l("toast.writeAuthTitle", "Majoor remote write access"),
				detail: t
			}, "warning", 6500, { noHistory: !0 });
		} catch (e) {
			console.debug?.(e);
		}
	}
}
function tt(e) {
	let t = String(e?.code || "").trim().toUpperCase(), n = String(e?.error || "").trim().toLowerCase(), r = t === "FORBIDDEN" && n.includes("write operation blocked");
	if (t !== "AUTH_REQUIRED" && !r) return e;
	let i = $e(e);
	return i ? (et(i), {
		...e,
		error: i
	}) : e;
}
async function nt() {
	try {
		let e = await fetch("/mjr/am/settings/security/bootstrap-token", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				"X-Requested-With": "XMLHttpRequest"
			},
			body: "{}"
		});
		if (!(e.headers.get("content-type") || "").includes("application/json")) return k({
			code: "INVALID_RESPONSE",
			error: `Bootstrap token request returned non-JSON response (${e.status})`,
			status: e.status
		}), {
			ok: !1,
			token: !1
		};
		let t = await e.json().catch((e) => (console.debug?.("[MJR auth] JSON parse error:", e), null));
		if (!t || typeof t != "object") return k({
			code: "INVALID_RESPONSE",
			error: "Bootstrap token response was invalid.",
			status: e.status
		}), {
			ok: !1,
			token: !1
		};
		if (!t.ok) return k({
			code: t?.code,
			error: t?.error,
			status: e.status
		}), {
			ok: !1,
			token: !1
		};
		let n = String(t?.data?.token || "").trim();
		return n ? {
			ok: Ye(n),
			token: !0
		} : (D = null, {
			ok: !0,
			token: !1
		});
	} catch (e) {
		return k({
			code: "NETWORK_ERROR",
			error: e?.message || "Bootstrap token request failed.",
			status: 0
		}), {
			ok: !1,
			token: !1
		};
	}
}
async function rt({ force: e = !1, allowCookieRefresh: t = !1 } = {}) {
	let n = Je();
	if (n && !e) return n;
	let r = {
		ok: !1,
		token: !1
	};
	He ||= (async () => {
		try {
			return await nt();
		} finally {
			He = null;
		}
	})();
	try {
		r = await He || r;
	} catch (e) {
		console.debug?.(e);
	}
	if (e && r?.ok && !r?.token && n) qe();
	else if (e && !r?.ok) {
		let e = Qe(), t = String(e?.code || "").trim().toUpperCase();
		(!t || !["NETWORK_ERROR", "INVALID_RESPONSE"].includes(t)) && qe();
	}
	let i = Je();
	return !i && t && r?.ok ? !0 : i;
}
function it() {
	O.clear();
}
//#endregion
//#region ui/api/clientOps.ts
async function at(e) {
	return !e || typeof e != "string" ? {
		ok: !1,
		error: "Missing mode",
		code: "INVALID_INPUT"
	} : P("/mjr/am/settings/probe-backend", { mode: e });
}
async function ot() {
	return N(e.SETTINGS_METADATA_FALLBACK);
}
async function st({ image: t, media: n } = {}) {
	return P(e.SETTINGS_METADATA_FALLBACK, {
		image: t,
		media: n
	});
}
async function ct() {
	return N(e.SETTINGS_VECTOR_SEARCH);
}
async function lt(t = !0) {
	if (t && typeof t == "object") {
		let n = {};
		return "enabled" in t && (n.enabled = !!t.enabled), "caption_on_index" in t && (n.caption_on_index = !!t.caption_on_index), "captionOnIndex" in t && (n.caption_on_index = !!t.captionOnIndex), P(e.SETTINGS_VECTOR_SEARCH, n);
	}
	return P(e.SETTINGS_VECTOR_SEARCH, { enabled: !!t });
}
async function ut() {
	return N(e.SETTINGS_EXECUTION_GROUPING);
}
async function dt(t = !0) {
	return P(e.SETTINGS_EXECUTION_GROUPING, { enabled: !!t });
}
async function ft() {
	return N(e.SETTINGS_HUGGINGFACE);
}
async function pt(t = "") {
	return P(e.SETTINGS_HUGGINGFACE, { token: String(t ?? "").trim() });
}
async function mt() {
	return N(e.SETTINGS_AI_LOGGING);
}
async function ht(t = !1) {
	return P(e.SETTINGS_AI_LOGGING, { enabled: !!t });
}
async function gt() {
	return N(e.SETTINGS_ROUTE_LOGGING);
}
async function _t(t = !1) {
	return P(e.SETTINGS_ROUTE_LOGGING, { enabled: !!t });
}
async function vt() {
	return N(e.SETTINGS_STARTUP_LOGGING);
}
async function yt(t = !1) {
	return P(e.SETTINGS_STARTUP_LOGGING, { enabled: !!t });
}
async function bt() {
	return N(e.SETTINGS_LTXAV_RGB_FALLBACK);
}
async function xt(t = !1) {
	return P(e.SETTINGS_LTXAV_RGB_FALLBACK, { enabled: !!t });
}
async function St() {
	return N(e.SETTINGS_OUTPUT_DIRECTORY);
}
async function Ct(t, n = {}) {
	let r = String(t ?? "").trim();
	return P(e.SETTINGS_OUTPUT_DIRECTORY, { output_directory: r }, n);
}
async function wt() {
	return N(e.SETTINGS_INDEX_DIRECTORY);
}
async function Tt(t, n = {}) {
	let r = String(t ?? "").trim();
	return P(e.SETTINGS_INDEX_DIRECTORY, { index_directory: r }, n);
}
async function Et(t = {}) {
	return N(e.SETTINGS_WORKFLOW_ROOTS, t);
}
async function Dt(t, n = {}) {
	let r = Array.isArray(t) ? t.map((e) => String(e ?? "").trim()).filter(Boolean) : String(t ?? "").trim();
	return P(e.SETTINGS_WORKFLOW_ROOTS, { workflow_roots: r }, n);
}
async function Ot() {
	return N("/mjr/am/settings/security");
}
async function kt(e) {
	return P("/mjr/am/settings/security", e && typeof e == "object" ? e : {});
}
async function At() {
	let e = await P("/mjr/am/settings/security/bootstrap-token", {});
	if (e?.ok) try {
		let t = String(e?.data?.token || "").trim();
		t && Ye(t);
	} catch (e) {
		console.debug?.(e);
	}
	return e;
}
async function jt(e) {
	if (e && typeof e == "object") {
		let t = String(e.filepath || e.path || e?.file_info?.filepath || "").trim();
		return e.id == null ? P("/mjr/am/open-in-folder", { filepath: t }) : P("/mjr/am/open-in-folder", { asset_id: s(e.id) });
	}
	return P("/mjr/am/open-in-folder", { asset_id: s(e) });
}
async function Mt({ op: t = "", path: n = "", name: r = "", destination: i = "", recursive: a = !0 } = {}, o = {}) {
	let s = {
		op: String(t || "").trim().toLowerCase(),
		path: String(n || "").trim()
	};
	return r != null && String(r).trim() && (s.name = String(r).trim()), i != null && String(i).trim() && (s.destination = String(i).trim()), s.op === "delete" && (s.recursive = !!a), P(e.BROWSER_FOLDER_OP, s, o);
}
async function Nt(t = {}) {
	let n = (e, t) => e == null ? t : !!e, r = String(t.scope || "output").trim().toLowerCase() || "output", i = t.customRootId ?? t.custom_root_id ?? t.rootId ?? t.root_id ?? t.customRoot ?? null, a = {
		scope: r,
		reindex: n(t.reindex, !0),
		hard_reset_db: n(t.hardResetDb ?? t.hard_reset_db ?? t.deleteDbFiles ?? t.delete_db_files ?? t.deleteDb ?? t.delete_db ?? void 0, r === "all"),
		clear_scan_journal: n(t.clearScanJournal ?? t.clear_scan_journal, !0),
		clear_metadata_cache: n(t.clearMetadataCache ?? t.clear_metadata_cache, !0),
		clear_asset_metadata: n(t.clearAssetMetadata ?? t.clear_asset_metadata, !0),
		clear_assets: n(t.clearAssets ?? t.clear_assets, !0),
		preserve_vectors: n(t.preserveVectors ?? t.preserve_vectors ?? t.keepVectors ?? t.keep_vectors, !1),
		rebuild_fts: n(t.rebuildFts ?? t.rebuild_fts, !0),
		incremental: n(t.incremental, !1),
		fast: n(t.fast, !0),
		background_metadata: n(t.backgroundMetadata ?? t.background_metadata, !0),
		maintenance_force: n(t.maintenanceForce ?? t.maintenance_force, !1)
	};
	return i && (a.custom_root_id = String(i)), P(e.INDEX_RESET, a);
}
async function Pt({ scope: t = "output", customRootId: n = "" } = {}) {
	let r = String(t || "output").trim().toLowerCase() || "output", i = String(n || "").trim(), a = { scope: r };
	return i && (a.custom_root_id = i), P(e.WATCHER_SCOPE, a);
}
async function Ft(t = {}) {
	return N(e.WATCHER_STATUS, t);
}
async function It(t = !0) {
	return P(e.WATCHER_TOGGLE, { enabled: !!t });
}
async function Lt() {
	return N(e.WATCHER_SETTINGS);
}
async function Rt(t = {}) {
	return P(e.WATCHER_SETTINGS, t);
}
async function zt(t = {}) {
	return N(e.TOOLS_STATUS, t);
}
async function Bt(t = {}) {
	return N(e.STATUS, t);
}
async function Vt() {
	return P("/mjr/am/db/force-delete", {});
}
async function Ht(t = {}) {
	return N(e.DB_BACKUPS, t);
}
async function Ut() {
	return P(e.DB_BACKUP_SAVE, {});
}
async function Wt({ name: t = "", useLatest: n = !1 } = {}) {
	let r = {};
	return t && (r.name = String(t)), n && (r.use_latest = !0), P(e.DB_BACKUP_RESTORE, r);
}
async function Gt(e = 250) {
	return P("/mjr/am/duplicates/analyze", { limit: Math.max(10, Math.min(5e3, Number(e) || 250)) });
}
async function Kt({ scope: e = "output", customRootId: t = "", maxGroups: n = 6, maxPairs: r = 10 } = {}, i = {}) {
	let a = `/mjr/am/duplicates/alerts?scope=${encodeURIComponent(String(e || "output"))}`;
	return t && (a += `&custom_root_id=${encodeURIComponent(String(t))}`), a += `&max_groups=${encodeURIComponent(String(Math.max(1, Number(n) || 6)))}`, a += `&max_pairs=${encodeURIComponent(String(Math.max(1, Number(r) || 10)))}`, N(a, i);
}
async function qt(e, t = []) {
	return P("/mjr/am/duplicates/merge-tags", {
		keep_asset_id: Number(e) || 0,
		merge_asset_ids: Array.isArray(t) ? t.map((e) => Number(e) || 0).filter((e) => e > 0) : []
	});
}
async function Jt(e) {
	let t, n;
	if (e && typeof e == "object") {
		t = s(e.id);
		let r = String(e.filepath || e.path || e?.file_info?.filepath || "").trim();
		n = t ? { asset_id: t } : { filepath: r };
	} else t = s(e), n = { asset_id: t };
	let r = await P("/mjr/am/asset/delete", n);
	return r?.ok && t && Xt([t]), r;
}
async function Yt(e) {
	let t = Array.isArray(e) ? e.map((e) => s(e)).filter(Boolean) : [], n = await P("/mjr/am/assets/delete", { ids: t });
	return n?.ok && Xt(t), n;
}
function Xt(e) {
	try {
		let t = (Array.isArray(e) ? e : [e]).map((e) => String(e || "").trim()).filter(Boolean);
		if (!t.length) return;
		window.dispatchEvent(new CustomEvent("mjr:assets-deleted", { detail: { ids: t } }));
	} catch (e) {
		console.debug?.(e);
	}
}
async function Zt(e, t) {
	let n;
	if (e && typeof e == "object") {
		n = s(e.id);
		let r = String(e.filepath || e.path || e?.file_info?.filepath || "").trim(), i = n ? await P("/mjr/am/asset/rename", {
			asset_id: n,
			new_name: t
		}) : await P("/mjr/am/asset/rename", {
			filepath: r,
			new_name: t
		});
		if (i?.ok && n) try {
			let e = await zn(n);
			e?.ok && e?.data && (i.data = {
				...i.data || {},
				asset: e.data
			});
		} catch (e) {
			console.debug?.(e);
		}
		return i;
	}
	n = s(e);
	let r = await P("/mjr/am/asset/rename", {
		asset_id: n,
		new_name: t
	});
	if (r?.ok && n) try {
		let e = await zn(n);
		e?.ok && e?.data && (r.data = {
			...r.data || {},
			asset: e.data
		});
	} catch (e) {
		console.debug?.(e);
	}
	return r;
}
async function Qt() {
	let e = typeof AbortController < "u" ? new AbortController() : null, t = null;
	try {
		return e && (t = setTimeout(() => e.abort(), 1e4)), await N("/mjr/am/collections", e ? { signal: e.signal } : {});
	} finally {
		t && clearTimeout(t);
	}
}
async function $t(e) {
	return P("/mjr/am/collections", { name: String(e || "").trim() });
}
async function en(e) {
	let t = String(e || "").trim();
	return P(`/mjr/am/collections/${encodeURIComponent(t)}/delete`, {});
}
async function tn(e, t) {
	let n = String(e || "").trim(), r = Array.isArray(t) ? t : [];
	return P(`/mjr/am/collections/${encodeURIComponent(n)}/add`, { assets: r });
}
async function nn(e, t) {
	let n = String(e || "").trim(), r = Array.isArray(t) ? t : [];
	return P(`/mjr/am/collections/${encodeURIComponent(n)}/remove`, { filepaths: r });
}
async function rn(e) {
	let t = String(e || "").trim();
	return N(`/mjr/am/collections/${encodeURIComponent(t)}/assets`);
}
async function an(t, n = 20) {
	let r = String(t || "").trim();
	if (!r) return {
		ok: !1,
		error: "Empty query"
	};
	let i = n && typeof n == "object" ? n : { topK: Number(n) }, o = Math.max(1, Math.min(200, Number(i?.topK ?? 20) || 20)), s = String(i?.scope || "").trim(), c = String(i?.customRootId || "").trim(), l = `${e.VECTOR_SEARCH}?q=${encodeURIComponent(r)}&top_k=${o}`;
	return s && (l += `&scope=${encodeURIComponent(s)}`), c && (l += `&custom_root_id=${encodeURIComponent(c)}`), l = a(l, {
		subfolder: i?.subfolder ?? null,
		kind: i?.kind ?? null,
		hasWorkflow: i?.hasWorkflow ?? null,
		minRating: i?.minRating ?? null,
		minSizeMB: i?.minSizeMB ?? null,
		maxSizeMB: i?.maxSizeMB ?? null,
		minWidth: i?.minWidth ?? null,
		minHeight: i?.minHeight ?? null,
		maxWidth: i?.maxWidth ?? null,
		maxHeight: i?.maxHeight ?? null,
		workflowType: i?.workflowType ?? null,
		workflowId: i?.workflowId ?? null,
		dateRange: i?.dateRange ?? null,
		dateExact: i?.dateExact ?? null
	}), N(l, { timeoutMs: 12e4 });
}
async function on(t, n = 20) {
	let r = String(t || "").trim();
	if (!r) return {
		ok: !1,
		error: "Missing asset ID"
	};
	let i = n && typeof n == "object" ? n : { topK: Number(n) }, a = Math.max(1, Math.min(200, Number(i?.topK ?? 20) || 20)), o = String(i?.scope || "").trim(), s = String(i?.customRootId || "").trim(), c = `${e.VECTOR_SIMILAR}/${encodeURIComponent(r)}?top_k=${a}`;
	return o && (c += `&scope=${encodeURIComponent(o)}`), s && (c += `&custom_root_id=${encodeURIComponent(s)}`), N(c, { dedupeKey: `vec:${r}:${a}:${o}:${s}` });
}
async function sn(t) {
	let n = String(t || "").trim();
	return n ? N(`${e.VECTOR_ALIGNMENT}/${encodeURIComponent(n)}`) : {
		ok: !1,
		error: "Missing asset ID"
	};
}
async function cn(t) {
	let n = String(t || "").trim();
	return n ? P(`${e.VECTOR_INDEX}/${encodeURIComponent(n)}`, {}) : {
		ok: !1,
		error: "Missing asset ID"
	};
}
async function ln() {
	return N(e.VECTOR_STATS);
}
async function un(t = 64, n = {}) {
	let r = Math.max(1, Math.min(200, t)), i = typeof n?.onProgress == "function" ? n.onProgress : null, a = String(n?.scope || "").trim().toLowerCase(), o = String(n?.customRootId ?? n?.custom_root_id ?? "").trim(), s = `${e.VECTOR_BACKFILL}?batch_size=${r}&async=1`;
	a && (s += `&scope=${encodeURIComponent(a)}`), o && (s += `&custom_root_id=${encodeURIComponent(o)}`);
	let c = await P(s, {}, { timeoutMs: 3e4 });
	if (!c?.ok) return c;
	let l = c?.data || {}, u = String(l?.status || "").toLowerCase(), d = String(l?.backfill_id || "").trim();
	try {
		i?.(l);
	} catch (e) {
		console.debug?.(e);
	}
	if (!d || ![
		"queued",
		"running",
		"pending"
	].includes(u)) return c;
	let f = Number(n?.pollIntervalMs), p = Number(n?.pollTimeoutMs), m = Number.isFinite(f) ? Math.max(500, Math.min(1e4, Math.floor(f))) : vn, h = Number.isFinite(p) ? Math.max(1e4, Math.min(bn, Math.floor(p))) : yn, ee = Date.now(), g = null;
	for (; Date.now() - ee < h;) {
		await b(m);
		let t = await N(`${e.VECTOR_BACKFILL_STATUS}?backfill_id=${encodeURIComponent(d)}`, { timeoutMs: 3e4 });
		if (!t?.ok) {
			g = t;
			continue;
		}
		let n = t?.data || {}, r = String(n?.status || "").toLowerCase();
		g = t;
		try {
			i?.(n);
		} catch (e) {
			console.debug?.(e);
		}
		if (r === "succeeded") return {
			ok: !0,
			data: n?.result || {},
			code: null,
			status: 200
		};
		if (r === "failed") return {
			ok: !1,
			error: String(n?.error || "Vector backfill failed"),
			code: String(n?.code || "DB_ERROR"),
			data: n,
			status: 500
		};
	}
	let _ = await N(`${e.VECTOR_BACKFILL_STATUS}?backfill_id=${encodeURIComponent(d)}`, { timeoutMs: 3e4 }), v = _?.data || g?.data || {}, y = String(v?.status || "").toLowerCase();
	if (_?.ok && [
		"queued",
		"running",
		"pending"
	].includes(y)) {
		try {
			i?.(v);
		} catch (e) {
			console.debug?.(e);
		}
		return {
			ok: !0,
			code: "PENDING",
			status: 202,
			data: {
				...v,
				pending: !0,
				timed_out: !0,
				poll_timeout_ms: h,
				backfill_id: String(v?.backfill_id || d),
				status: y || "running"
			},
			meta: { pending: !0 }
		};
	}
	return _?.ok && y === "failed" ? {
		ok: !1,
		error: String(v?.error || "Vector backfill failed"),
		code: String(v?.code || "DB_ERROR"),
		data: v,
		status: 500
	} : {
		ok: !1,
		error: `Vector backfill polling timed out after ${h}ms`,
		code: "TIMEOUT",
		data: v || null,
		status: 408
	};
}
async function dn(t) {
	let n = String(t || "").trim();
	return n ? P(`${e.VECTOR_CAPTION}/${encodeURIComponent(n)}`, {}) : {
		ok: !1,
		error: "Missing asset ID"
	};
}
async function fn(t, { topK: n = 50, scope: r = "output", customRootId: i = "", subfolder: o = null, kind: s = null, hasWorkflow: c = null, minRating: l = null, minSizeMB: u = null, maxSizeMB: d = null, minWidth: f = null, minHeight: p = null, maxWidth: m = null, maxHeight: h = null, workflowType: ee = null, workflowId: g = null, dateRange: _ = null, dateExact: v = null } = {}) {
	let y = String(t || "").trim();
	if (!y) return {
		ok: !1,
		error: "Empty query"
	};
	let b = `${e.HYBRID_SEARCH}?q=${encodeURIComponent(y)}&top_k=${Math.max(1, Math.min(200, n))}&scope=${encodeURIComponent(r)}`;
	return i && (b += `&custom_root_id=${encodeURIComponent(i)}`), b = a(b, {
		subfolder: o,
		kind: s,
		hasWorkflow: c,
		minRating: l,
		minSizeMB: u,
		maxSizeMB: d,
		minWidth: f,
		minHeight: p,
		maxWidth: m,
		maxHeight: h,
		workflowType: ee,
		workflowId: g,
		dateRange: _,
		dateExact: v
	}), N(b, { timeoutMs: 12e4 });
}
async function pn(t = 8) {
	return P(e.VECTOR_SUGGEST_COLLECTIONS, { k: Math.max(2, Math.min(20, t)) });
}
//#endregion
//#region ui/api/client.ts
var mn = 3e4, hn = "__MJR_API_CLIENT__", gn = 2e3, _n = 200, vn = 1e3, yn = 30 * 6e4, bn = 720 * 6e4, A = "settings", xn = "available-tags", j = C({
	ttlMs: gn,
	maxSize: 1
}), M = C({
	ttlMs: gn,
	maxSize: 1
}), Sn = C({
	ttlMs: () => En(),
	maxSize: 1
}), Cn = new Set([
	"1",
	"true",
	"yes",
	"on"
]), wn = new Set([
	"0",
	"false",
	"no",
	"off"
]);
function Tn(e, t = !1) {
	if (typeof e == "boolean") return e;
	if (typeof e == "number") return e !== 0;
	if (typeof e == "string") {
		let t = e.trim().toLowerCase();
		if (Cn.has(t)) return !0;
		if (wn.has(t)) return !1;
	}
	return !!t;
}
function En() {
	try {
		let e = localStorage?.getItem?.("mjrSettings") || "{}", t = JSON.parse(e), n = t?.cache?.tagsTTLms ?? t?.cache?.tagsTTL ?? t?.cache?.tags_ttl_ms ?? null, r = Number(n);
		return Number.isFinite(r) ? Math.max(1e3, Math.min(10 * 6e4, Math.floor(r))) : mn;
	} catch {
		return mn;
	}
}
function Dn() {
	j.clear();
}
function On() {
	M.clear();
}
function kn() {
	Sn.clear();
}
function An(e) {
	return String(e ?? "").trim().toLowerCase() || "";
}
function jn(e) {
	let t = [], n = /* @__PURE__ */ new Set();
	for (let r of Array.isArray(e) ? e : []) {
		let e = String(r ?? "").trim();
		if (!e) continue;
		let i = An(e);
		!i || n.has(i) || (n.add(i), t.push(e));
	}
	return t;
}
try {
	let e = typeof window < "u" ? window : null;
	e && !e[hn] && (e[hn] = { initialized: !0 }, e.addEventListener?.("storage", (e) => {
		try {
			e?.key === "mjrSettings" && (Dn(), On(), kn(), it());
		} catch (e) {
			console.debug?.(e);
		}
	}), e.addEventListener?.("mjr-settings-changed", () => {
		Dn(), On(), kn(), it();
	}));
} catch (e) {
	console.debug?.(e);
}
var Mn = () => {
	let e = j.get(A);
	if (e !== void 0) return e;
	let t = Date.now();
	try {
		let e = localStorage?.getItem?.(d);
		if (!e) return j.set(A, !1, { at: t }), !1;
		let n = !!JSON.parse(e)?.observability?.enabled;
		return j.set(A, n, { at: t }), n;
	} catch {
		return j.set(A, !1, { at: t }), !1;
	}
}, Nn = () => {
	let e = M.get(A);
	if (e !== void 0) return e;
	let t = Date.now();
	try {
		let e = localStorage?.getItem?.(d);
		if (!e) return M.set(A, !0, { at: t }), !0;
		let n = JSON.parse(e)?.ratingTagsSync?.enabled, r = n == null ? !0 : Tn(n, !0);
		return M.set(A, r, { at: t }), r;
	} catch {
		return M.set(A, !0, { at: t }), !0;
	}
}, Pn = de({
	readObsEnabled: Mn,
	readAuthToken: Je,
	ensureWriteAuthToken: rt,
	normalizeWriteAuthFailure: tt
}), Fn = Pn.fetchAPI;
async function N(e, t = {}) {
	return Pn.get(e, t);
}
async function P(e, t, n = {}) {
	return Pn.post(e, t, n);
}
async function In(e, t, n = {}) {
	let i = Nn(), a = e && typeof e == "object" ? e : null, o = s(a ? a.id : e), c = { rating: Math.max(0, Math.min(5, Number(t) || 0)) };
	return o ? c.asset_id = o : a && (c.filepath = a.filepath || a.path || a?.file_info?.filepath || "", c.type = a.type || "output", c.root_id = r(a)), Fn("/mjr/am/asset/rating", {
		...n,
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			...i ? { "X-MJR-RTSYNC": "on" } : {}
		},
		body: JSON.stringify(c)
	});
}
async function Ln(t, n, i = {}) {
	let a = Nn(), o = t && typeof t == "object" ? t : null, c = s(o ? o.id : t), l = String(o?.kind || o?.type || "").trim().toLowerCase() === "workflow", u = String(o?.filepath || o?.path || o?.file_info?.filepath || "").trim(), d = { tags: Array.isArray(n) ? n : [] };
	l && u ? d.filepath = u : c ? d.asset_id = c : o && (d.filepath = o.filepath || o.path || o?.file_info?.filepath || "", d.type = o.type || "output", d.root_id = r(o));
	let f = await Fn(l && u ? e.WORKFLOWS_TAGS : "/mjr/am/asset/tags", {
		...i,
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			...a ? { "X-MJR-RTSYNC": "on" } : {}
		},
		body: JSON.stringify(d)
	});
	return f?.ok && kn(), f;
}
async function Rn() {
	let e = Sn.get(xn);
	if (Array.isArray(e)) return {
		ok: !0,
		data: e,
		error: null,
		code: "OK",
		meta: { cached: !0 }
	};
	let t = await N("/mjr/am/tags");
	if (t?.ok && Array.isArray(t.data)) {
		let e = jn(t.data);
		return Sn.set(xn, e), {
			...t,
			data: e
		};
	}
	return t;
}
async function zn(e, t = {}) {
	let n = encodeURIComponent(s(e));
	return N(`/mjr/am/asset/${n}`, {
		...t,
		dedupeKey: t?.dedupeKey || `meta:${n}`
	});
}
async function Bn(e, t = {}) {
	let n = s(e);
	if (!n) return {
		ok: !1,
		data: null,
		error: "Missing assetId",
		code: "INVALID_INPUT"
	};
	let r = `/mjr/am/viewer/info?asset_id=${encodeURIComponent(n)}`;
	t.refresh && (r += "&refresh=1");
	let { refresh: i, ...a } = t;
	return N(r, a);
}
async function Vn(e, t = {}) {
	let n = Array.isArray(e) ? e : [], r = [];
	for (let e of n) {
		let t = Number(e);
		if (Number.isFinite(t) && (r.push(Math.trunc(t)), r.length >= _n)) break;
	}
	return r.length ? P("/mjr/am/assets/batch", { asset_ids: r }, t) : {
		ok: !0,
		data: [],
		error: null,
		code: "OK"
	};
}
async function Hn(e, t = {}) {
	let n = o(e);
	return n ? N(n, t) : {
		ok: !1,
		data: null,
		error: "Missing workflow filepath",
		code: "INVALID_INPUT"
	};
}
async function Un({ workflow: t = null, name: n = "", category: r = "", overwrite: i = !1, filepath: a = "" } = {}, o = {}) {
	return P(e.WORKFLOWS_SAVE, {
		workflow: t,
		name: n,
		category: r,
		overwrite: i,
		filepath: a
	}, o);
}
async function Wn({ filepath: t = "", name: n = "" } = {}, r = {}) {
	return P(e.WORKFLOWS_DUPLICATE, {
		filepath: t,
		name: n
	}, r);
}
async function Gn({ filepath: t = "", name: n = "", category: r = "" } = {}, i = {}) {
	return P(e.WORKFLOWS_MOVE, {
		filepath: t,
		name: n,
		category: r
	}, i);
}
async function Kn({ filepath: t = "" } = {}, n = {}) {
	return P(e.WORKFLOWS_DELETE, { filepath: t }, n);
}
async function qn({ filepath: t = "" } = {}, n = {}) {
	return P(e.WORKFLOWS_MARK_LOADED, { filepath: t }, n);
}
async function Jn({ filepath: t = "", favorite: n = !1 } = {}, r = {}) {
	return P(e.WORKFLOWS_FAVORITE, {
		filepath: t,
		favorite: !!n
	}, r);
}
async function Yn({ filepath: t = "", task: n = "", model_family: r = "", provider: i = "", runs_on: a = "", notes: o = "" } = {}, s = {}) {
	return P(e.WORKFLOWS_INFO, {
		filepath: t,
		task: n,
		model_family: r,
		provider: i,
		runs_on: a,
		notes: o
	}, s);
}
async function Xn({ filepath: t = "", limit: n = 12 } = {}, r = {}) {
	let i = Math.max(1, Math.min(50, Number(n) || 12));
	return N(`${e.WORKFLOWS_THUMBNAIL_CANDIDATES}?filepath=${encodeURIComponent(String(t || "").trim())}&limit=${encodeURIComponent(String(i))}`, r);
}
async function Zn(t = {}) {
	return N(e.WORKFLOWS_MODEL_FAMILIES, t);
}
async function Qn({ q: t = "*", limit: n = 100, offset: r = 0, sort: i = "mtime" } = {}, a = {}) {
	let o = Math.max(1, Math.min(500, Number(n) || 100)), s = Math.max(0, Number(r) || 0);
	return N(`${e.LIST}?scope=workflow&q=${encodeURIComponent(String(t || "*"))}&limit=${encodeURIComponent(String(o))}&offset=${encodeURIComponent(String(s))}&sort=${encodeURIComponent(String(i || "mtime"))}`, a);
}
async function $n({ filepath: t = "", source_filepath: n = "" } = {}, r = {}) {
	return P(e.WORKFLOWS_THUMBNAIL_SET, {
		filepath: t,
		source_filepath: n
	}, r);
}
async function er({ type: e = "output", filename: t = "", subfolder: n = "", root_id: r = "", rootId: i = "", filepath: a = "" } = {}, o = {}) {
	let s = String(e || "output").trim().toLowerCase() || "output", c = String(t || "").trim(), l = String(n || "").trim(), u = String(r || i || "").trim(), d = String(a || "").trim();
	if (!c) return {
		ok: !1,
		data: null,
		error: "Missing filename",
		code: "INVALID_INPUT"
	};
	let f = `/mjr/am/metadata?type=${encodeURIComponent(s)}&filename=${encodeURIComponent(c)}`;
	return d && (f += `&filepath=${encodeURIComponent(d)}`), l && (f += `&subfolder=${encodeURIComponent(l)}`), u && (f += `&root_id=${encodeURIComponent(u)}`), N(f, o);
}
async function tr({ filepath: t = "", root_id: n = "", subfolder: r = "" } = {}, i = {}) {
	try {
		if (globalThis.__mjrFolderInfoSupported === !1) return {
			ok: !1,
			data: null,
			error: "Folder info endpoint unavailable",
			code: "UNAVAILABLE"
		};
		if (globalThis.__mjrFolderInfoSupported == null) {
			let e = await N("/mjr/am/routes");
			if (e?.ok && Array.isArray(e.data)) {
				let t = e.data.some((e) => String(e?.path || "").trim() === "/mjr/am/folder-info");
				if (globalThis.__mjrFolderInfoSupported = !!t, !t) return {
					ok: !1,
					data: null,
					error: "Folder info endpoint unavailable",
					code: "UNAVAILABLE"
				};
			} else globalThis.__mjrFolderInfoSupported = null;
		}
	} catch (e) {
		console.debug?.(e);
	}
	let a = String(t || "").trim(), o = String(n || "").trim(), s = String(r || "").trim(), c = e.FOLDER_INFO, l = [];
	a ? (l.push(`filepath=${encodeURIComponent(a)}`), l.push("browser_mode=1")) : (o && l.push(`root_id=${encodeURIComponent(o)}`), s && l.push(`subfolder=${encodeURIComponent(s)}`)), l.length && (c += `?${l.join("&")}`);
	let u = await N(c, i);
	try {
		!u?.ok && Number(u?.status || 0) === 404 && (globalThis.__mjrFolderInfoSupported = !1);
	} catch (e) {
		console.debug?.(e);
	}
	return u;
}
//#endregion
//#region ui/utils/logging.ts
function nr(e, ...t) {
	try {
		i.DEBUG_VERBOSE_ERRORS && console.debug(e, ...t);
	} catch {}
}
function rr(e, t = "Majoor", { showToast: n = !1, toastType: r = "error" } = {}) {
	let a = e?.message || String(e || "Unknown error");
	try {
		i.DEBUG_VERBOSE_ERRORS ? console.error(`[Majoor][${t}]`, a, e) : console.debug(`[Majoor][${t}]`, a);
	} catch (e) {
		console.debug?.(e);
	}
	if (n && i.DEBUG_VERBOSE_ERRORS) try {
		Le(`${t}: ${a}`, r, 4e3);
	} catch (e) {
		console.debug?.(e);
	}
}
//#endregion
//#region ui/features/panel/panelRuntimeRefs.ts
var F = null;
function ir(e) {
	return e ? typeof e?.isConnected == "boolean" ? e.isConnected : !0 : !1;
}
function ar() {
	try {
		let e = document.getElementById("mjr-assets-grid") || document.querySelector(".mjr-grid");
		if (ir(e)) return e;
	} catch (e) {
		console.debug?.(e);
	}
	return null;
}
function or(e) {
	return F = e || null, F;
}
function sr(e = null) {
	(!e || F === e) && (F = null);
}
function cr() {
	return ir(F) ? F : (F = null, ar());
}
//#endregion
//#region ui/features/grid/GridSelectionManager.ts
function lr(e) {
	let t = /* @__PURE__ */ new Set();
	if (!e) return t;
	try {
		let n = e.dataset?.mjrSelectedAssetIds;
		if (n) {
			let e = JSON.parse(n);
			if (Array.isArray(e)) for (let n of e) n != null && t.add(String(n));
			return t;
		}
	} catch (e) {
		console.warn("[MJR] selection state parse error:", e);
	}
	try {
		let n = e.dataset?.mjrSelectedAssetId;
		n && t.add(String(n));
	} catch (e) {
		console.debug?.(e);
	}
	return t;
}
function ur(e, t, n = "") {
	let r = Array.from(t || []), i = n ? String(n) : r[0] ? String(r[0]) : "";
	try {
		r.length ? (e.dataset.mjrSelectedAssetIds = JSON.stringify(r), i ? e.dataset.mjrSelectedAssetId = String(i) : delete e.dataset.mjrSelectedAssetId) : (delete e.dataset.mjrSelectedAssetIds, delete e.dataset.mjrSelectedAssetId);
	} catch (e) {
		console.debug?.(e);
	}
	return r;
}
function dr(e, t, n) {
	if (!e) return;
	let r = t instanceof Set ? t : new Set(Array.from(t || []).map(String));
	try {
		let t = typeof n == "function" ? n(e) : [];
		for (let e of t) {
			let t = e?.dataset?.mjrAssetId;
			t && r.has(String(t)) ? (e.classList.add("is-selected"), e.setAttribute("aria-selected", "true")) : (e.classList.remove("is-selected"), e.setAttribute("aria-selected", "false"));
		}
	} catch (e) {
		console.debug?.(e);
	}
}
function fr(e, t, { activeId: n = "" } = {}, r) {
	if (!e) return [];
	let i = new Set(Array.from(t || []).map(String).filter(Boolean)), a = ur(e, i, n);
	dr(e, i, r);
	let o = {
		selectedIds: a,
		activeId: n || a[0] || ""
	};
	try {
		e.dispatchEvent?.(new CustomEvent("mjr:selection-changed", { detail: o }));
	} catch (e) {
		console.debug?.(e);
	}
	try {
		window.dispatchEvent(new CustomEvent("mjr:selection-changed", { detail: o }));
	} catch (e) {
		console.debug?.(e);
	}
	return a;
}
//#endregion
//#region ui/features/panel/controllers/hotkeysState.ts
var pr = {
	suspended: !1,
	scope: null,
	ratingHotkeysActive: !1
};
function mr() {
	return pr;
}
function hr(e) {
	pr.scope = e == null ? null : String(e);
}
function gr() {
	return !!pr.suspended;
}
function _r(e) {
	pr.ratingHotkeysActive = !!e;
}
//#endregion
//#region ui/features/viewer/viewerRuntimeHosts.ts
var I = null, L = null, vr = ".mjr-viewer-overlay", yr = ".mjr-mfv";
function br(e) {
	return !!e && typeof e.appendChild == "function";
}
function xr() {
	return typeof document > "u" ? null : document?.body || null;
}
function Sr() {
	return typeof document > "u" ? null : document?.body || document?.documentElement || null;
}
function Cr(e) {
	return br(e) ? e === xr() ? !0 : typeof e?.isConnected == "boolean" ? e.isConnected : !0 : !1;
}
function wr(e) {
	return br(e) ? e : null;
}
function R(e) {
	return Cr(e) ? e : xr();
}
function Tr(e) {
	let t = Sr();
	return Cr(t) ? t : R(e);
}
function Er(e, t, n = R) {
	let r = [], i = /* @__PURE__ */ new Set();
	for (let a of [
		n(t),
		xr(),
		t
	]) {
		if (!a || i.has(a)) continue;
		i.add(a);
		let t = [];
		try {
			t = Array.from(a.querySelectorAll?.(e) || []);
		} catch (e) {
			console.debug?.(e);
		}
		for (let e of t) !e || r.includes(e) || r.push(e);
	}
	return r;
}
function Dr(e, t, n = R) {
	let r = n(t);
	if (!r) return;
	let i = Er(e, t, n);
	for (let e of i) if (e && e.parentNode !== r) try {
		r.appendChild(e);
	} catch (e) {
		console.debug?.(e);
	}
}
function Or(e) {
	return I = wr(e), Dr(vr, I), () => kr(e);
}
function kr(e) {
	(!e || I === e) && (I = null);
}
function Ar(e) {
	return L = wr(e), Dr(yr, L, Tr), () => jr(e);
}
function jr(e) {
	(!e || L === e) && (L = null);
}
function Mr(e) {
	let t = R(I);
	try {
		t?.appendChild?.(e);
	} catch (e) {
		console.debug?.(e);
	}
	return t;
}
function Nr(e) {
	let t = Tr(L);
	try {
		t?.appendChild?.(e);
	} catch (e) {
		console.debug?.(e);
	}
	return t;
}
function Pr() {
	return Er(vr, I);
}
//#endregion
//#region ui/features/viewer/floatingViewerManager.ts
var Fr = null, Ir = null;
async function Lr() {
	return Fr || (Ir ||= import("./FloatingViewer-YN2IJngY.js").then((e) => (Fr = e.FloatingViewer, Fr)), Ir);
}
var z = null, Rr = null;
async function zr() {
	if (!z) return Rr ||= import("./NodeStreamController-EQygLyLg.js").then((e) => {
		z = e.setNodeStreamActive;
	}), Rr;
}
var B = Object.freeze({
	SIMPLE: "simple",
	AB: "ab",
	SIDE: "side",
	GRID: "grid",
	GRAPH: "graph"
});
function Br(e) {
	let t = String(e || "").trim().toLowerCase();
	return t === "sidebyside" ? B.SIDE : Object.values(B).includes(t) ? t : "";
}
var V = null;
function Vr() {
	return i.MFV_LIVE_DEFAULT !== !1;
}
function Hr() {
	return i.MFV_PREVIEW_DEFAULT !== !1;
}
var H = Vr(), U = Hr(), W = !1, Ur = null, Wr = !1, G = null, Gr = 0;
function Kr() {
	H = Vr(), U = Hr(), V?.setLiveActive(H), V?.setPreviewActive(U);
}
async function K() {
	if (!V) {
		let e = await Lr();
		V || (V = new e({ controller: {
			close: () => $.close(),
			toggle: () => $.toggle(),
			toggleLive: () => $.toggleLive(),
			togglePreview: () => $.togglePreview(),
			toggleNodeStream: () => $.toggleNodeStream(),
			popOut: () => $.popOut(),
			onModeChanged: (e) => {
				V?.isVisible && e !== B.SIMPLE && ei();
			},
			handleForwardedKeydown: (e) => yi(e)
		} }), Nr(V.render()));
	}
	try {
		let e = V?.element || null;
		e?.isConnected === !1 && Nr(e);
	} catch (e) {
		console.debug?.(e);
	}
	return V;
}
function qr() {
	try {
		G?.abort();
	} catch (e) {
		console.debug?.(e);
	}
	G = null;
}
function Jr() {
	try {
		let e = window.__MJR_LAST_SELECTION_GRID__;
		if (e?.isConnected) return e;
	} catch (e) {
		console.debug?.(e);
	}
	return cr();
}
function Yr() {
	if (V) {
		try {
			V.dispose?.();
		} catch (e) {
			console.debug?.(e);
		}
		V = null;
	}
}
function q(e) {
	typeof window > "u" || window.dispatchEvent(new CustomEvent(u.MFV_VISIBILITY_CHANGED, { detail: { visible: !!e } }));
}
function J(e) {
	e && (e.setLiveActive(H), e.setPreviewActive(U), e.setNodeStreamActive?.(W));
}
function Xr(e) {
	e?.setNodeStreamSelection && e.setNodeStreamSelection(Ur || null);
}
function Zr(e, t) {
	if (!e) return;
	let n = e._mode;
	if (!(n === B.AB || n === B.SIDE || n === B.GRID)) {
		e.loadMediaA(t, { autoMode: !0 });
		return;
	}
	let r = e.getPinnedSlots();
	if (n === B.GRID) {
		let n = [
			"A",
			"B",
			"C",
			"D"
		].find((e) => !r.has(e));
		if (!n) return;
		let i = {
			A: e._mediaA,
			B: e._mediaB,
			C: e._mediaC,
			D: e._mediaD
		};
		i[n] = t, e.loadMediaQuad(i.A, i.B, i.C, i.D);
		return;
	}
	let i = r.has("A"), a = r.has("B");
	if (!(i && a)) {
		if (a) {
			e.loadMediaPair(t, e._mediaB);
			return;
		}
		e.loadMediaPair(e._mediaA, t);
	}
}
function Qr(e) {
	try {
		let t = Jr();
		if (!t) return null;
		let n = Array.from(t.querySelectorAll("[data-mjr-asset-id]")), r = n.findIndex((t) => t.dataset.mjrAssetId === String(e));
		if (r < 0) return null;
		let i = (n[r + 1] ?? n[r - 1] ?? null)?.dataset?.mjrAssetId ?? null;
		return i && i !== String(e) ? i : null;
	} catch (e) {
		return console.debug?.("[MFV] _findAdjacentGridId error", e), null;
	}
}
async function $r(e) {
	if (!e.length || !V) return;
	qr();
	let t = ++Gr, n = typeof AbortController < "u" ? new AbortController() : null;
	G = n;
	try {
		let r = V.getPinnedSlots(), i = r.size > 0, a = V._mode, o = a === B.GRID, s = a === B.AB || a === B.SIDE, c = o ? 4 : 2, l = e.slice(0, c);
		if (i && (s || o)) {
			let e = c - r.size;
			l = l.slice(0, Math.max(1, e));
		} else if (l.length === 1 && s) {
			let e = Qr(l[0]);
			e && (l = [l[0], e]);
		}
		let u = await Vn(l, n ? { signal: n.signal } : {});
		if (n?.signal.aborted || Gr !== t || !u?.ok || !Array.isArray(u.data) || !u.data.length || !V) return;
		let d = u.data;
		if (o) {
			if (i) {
				let e = {
					A: V._mediaA,
					B: V._mediaB,
					C: V._mediaC,
					D: V._mediaD
				}, t = [
					"A",
					"B",
					"C",
					"D"
				].filter((e) => !r.has(e)), n = 0;
				for (let r of t) n < d.length && (e[r] = d[n++]);
				V.loadMediaQuad(e.A, e.B, e.C, e.D);
			} else d.length >= 3 ? V.loadMediaQuad(d[0], d[1], d[2], d[3] || null) : d.length >= 2 ? V.loadMediaPair(d[0], d[1]) : V.loadMediaA(d[0], { autoMode: !0 });
			return;
		}
		if (r.has("A") && r.has("B") && V._mediaA && V._mediaB) return;
		r.has("A") && V._mediaA ? V.loadMediaPair(V._mediaA, d[0]) : r.has("B") && V._mediaB ? V.loadMediaPair(d[0], V._mediaB) : l.length >= 2 && d.length >= 2 ? V.loadMediaPair(d[0], d[1]) : V.loadMediaA(d[0], { autoMode: !0 });
	} catch (e) {
		e?.name !== "AbortError" && rr(e, "floatingViewerManager._loadFromIds");
	} finally {
		G === n && (G = null);
	}
}
function ei() {
	try {
		let e = Jr();
		if (!e) return;
		let t = lr(e);
		if (!t.size) return;
		$r(Array.from(t));
	} catch (e) {
		console.debug?.("[MFV] Error reading current grid selection", e);
	}
}
function ti(e) {
	if (!V?.isVisible || V._mode === B.GRAPH) return;
	let t = Array.isArray(e?.detail?.selectedAssets) ? e.detail.selectedAssets : [], n = new Set(t.filter((e) => String(e?.kind || "").toLowerCase() === "folder").map((e) => String(e?.id || "")).filter(Boolean)), r = Array.isArray(e?.detail?.selectedIds) ? e.detail.selectedIds.map(String).filter((e) => !!e && !n.has(e)) : [];
	if (r.length) {
		$r(r);
		return;
	}
	try {
		let e = Jr();
		if (!e) return;
		let t = Array.from(lr(e)).map(String).filter(Boolean);
		if (!t.length) return;
		$r(t);
	} catch (e) {
		console.debug?.("[MFV] selection fallback failed", e);
	}
}
function Y() {
	Wr || typeof window > "u" || (window.addEventListener(u.SELECTION_CHANGED, ti), Wr = !0);
}
function ni() {
	typeof window < "u" && window.removeEventListener(u.SELECTION_CHANGED, ti), Wr = !1, qr();
}
var X = !1, ri = null, Z = null, Q = "";
function ii() {
	V?.isVisible && V.refreshSidebar?.();
}
function ai() {
	oi();
	let e = typeof window < "u" ? window : globalThis;
	if (typeof e.requestAnimationFrame == "function") {
		Q = "raf", Z = e.requestAnimationFrame(() => {
			Z = null, Q = "", ii();
		});
		return;
	}
	Q = "timeout", Z = e.setTimeout(() => {
		Z = null, Q = "", ii();
	}, 16);
}
function oi() {
	if (Z == null) return;
	let e = typeof window < "u" ? window : globalThis;
	try {
		Q === "raf" && typeof e.cancelAnimationFrame == "function" ? e.cancelAnimationFrame(Z) : typeof e.clearTimeout == "function" && e.clearTimeout(Z);
	} catch (e) {
		console.debug?.(e);
	}
	Z = null, Q = "";
}
function si() {
	if (!X) try {
		ri = n(ai, { includePointerFallback: !0 }), X = typeof ri == "function";
	} catch (e) {
		console.debug?.("[MFV] _bindNodeSelectionListener error", e);
	}
}
function ci() {
	if (X) {
		oi();
		try {
			ri?.();
		} catch (e) {
			console.debug?.("[MFV] _unbindNodeSelectionListener error", e);
		}
		ri = null, X = !1;
	}
}
var $ = {
	isGraphModeVisible() {
		return !!(V?.isVisible && V?._mode === B.GRAPH);
	},
	async openAssets({ assets: e = [], asset: t = null, index: n = 0, mode: r = "" } = {}) {
		let i = Array.isArray(e) ? e.filter(Boolean) : t ? [t] : [];
		if (!i.length) return !1;
		let a = await K(), o = !!a.isVisible, s = Math.max(0, Math.min(Number(n) || 0, i.length - 1)), c = Br(r);
		c && a.setMode(c), a.show(), J(a), Y(), si();
		let l = a._mode;
		return l === B.GRID && i.length >= 3 ? a.loadMediaQuad(i[0], i[1], i[2], i[3] || null) : (l === B.AB || l === B.SIDE) && i.length >= 2 ? a.loadMediaPair(i[0], i[1]) : a.loadMediaA(i[s], { autoMode: !1 }), o || q(!0), !0;
	},
	async open() {
		let e = await K();
		e.show(), J(e), Y(), si(), X || requestAnimationFrame(() => si()), ei(), W && Xr(e), q(!0);
	},
	close() {
		if (V) try {
			V.isPopped && V.popIn(), V.hide();
		} catch (e) {
			console.debug?.(e);
		}
		ni(), V?.setNodeStreamSelection?.(null), ci(), q(!1);
	},
	async toggle() {
		V?.isVisible ? $.close() : await $.open();
	},
	toggleLive() {
		$.setLiveActive(!H);
	},
	togglePreview() {
		$.setPreviewActive(!U);
	},
	async toggleCompareAB() {
		let e = await K();
		if (!e.isVisible) {
			e.setMode(B.AB), e.show(), J(e), Y(), ei(), q(!0);
			return;
		}
		let t = {
			[B.AB]: B.SIDE,
			[B.SIDE]: B.SIMPLE,
			[B.GRID]: B.SIMPLE,
			[B.SIMPLE]: B.AB
		}[e._mode] || B.AB;
		e.setMode(t), t !== B.SIMPLE && ei();
	},
	async upsertWithContent(e) {
		let t = await K(), n = !!t.isVisible;
		!n && i.MFV_LIVE_AUTO_OPEN === !1 || (t.show(), J(t), Y(), Zr(t, e), n || q(!0));
	},
	setLiveActive(e) {
		H = !!e, V?.setLiveActive(H);
	},
	getLiveActive() {
		return H;
	},
	async popOut() {
		let e = await K();
		e.isPopped ? e.popIn() : (e.isVisible || await $.open(), e.popOut());
	},
	setPreviewActive(e) {
		U = !!e, V?.setPreviewActive(U);
	},
	getPreviewActive() {
		return U;
	},
	async feedPreviewBlob(e, t = {}) {
		if (!U) return;
		let n = await K(), r = !!n.isVisible;
		!r && i.MFV_PREVIEW_AUTO_OPEN === !1 || (n.isVisible || n.show(), J(n), n.loadPreviewBlob(e, ...Object.keys(t).length ? [t] : []), r || q(!0));
	},
	toggleNodeStream() {
		$.setNodeStreamActive(!W);
	},
	setNodeStreamActive(e) {
		W = !!e, W || (Ur = null), zr().then(() => {
			z && z(W);
		}), V?.setNodeStreamActive?.(W), W ? V && Xr(V) : V?.setNodeStreamSelection?.(null);
	},
	getNodeStreamActive() {
		return W;
	},
	setNodeStreamSelection(e, t, n) {
		Ur = e == null || e === "" ? null : {
			nodeId: e,
			classType: t,
			title: n
		};
		let r = V;
		r && Xr(r);
	},
	async feedNodeStream(e) {
		if (!W) return;
		let t = await K(), n = !!t.isVisible;
		!n && i.MFV_NODE_STREAM_AUTO_OPEN === !1 || (t.isVisible || (t.show(), Y()), J(t), Xr(t), Zr(t, e), n || q(!0));
	}
}, li = !1, ui = () => $.open(), di = () => $.close(), fi = () => $.toggle(), pi = () => $.toggleLive(), mi = () => $.togglePreview(), hi = () => $.toggleNodeStream(), gi = () => $.popOut(), _i = () => {
	try {
		V?.isPopped && V.popIn();
	} catch {}
}, vi = (e) => {
	let t = String(e?.detail?.key || "");
	(!t || t === "viewer" || t === "viewer.mfvLiveDefault" || t === "viewer.mfvPreviewDefault") && Kr();
}, yi = (e) => {
	if (!V?.isVisible || gr() || mr().scope === "viewer") return;
	let t = e?.key?.toLowerCase?.() || "";
	if (e?.target?.isContentEditable || e?.target?.closest?.("input, textarea, select, [contenteditable='true']")) return;
	let n = () => {
		e.preventDefault?.(), e.stopPropagation?.(), e.stopImmediatePropagation?.();
	};
	if (!e?.ctrlKey && !e?.metaKey && !e?.altKey && !e?.shiftKey) {
		if (t === "v") {
			n(), $.toggle();
			return;
		}
		if (t === "k") {
			n(), $.togglePreview();
			return;
		}
		if (t === "l") {
			n(), $.toggleLive();
			return;
		}
		if (t === "n") {
			n(), $.toggleNodeStream();
			return;
		}
		t === "c" && (n(), $.toggleCompareAB());
		return;
	}
};
function bi() {
	li || typeof window > "u" || !window?.addEventListener || (window.addEventListener(u.MFV_OPEN, ui), window.addEventListener(u.MFV_CLOSE, di), window.addEventListener(u.MFV_TOGGLE, fi), window.addEventListener(u.MFV_LIVE_TOGGLE, pi), window.addEventListener(u.MFV_PREVIEW_TOGGLE, mi), window.addEventListener(u.MFV_NODESTREAM_TOGGLE, hi), window.addEventListener(u.MFV_POPOUT, gi), window.addEventListener(u.SETTINGS_CHANGED, vi), window.addEventListener("keydown", yi, !0), window.addEventListener("beforeunload", _i), li = !0);
}
function xi() {
	if (typeof window > "u" || !window?.removeEventListener) {
		li = !1;
		return;
	}
	window.removeEventListener(u.MFV_OPEN, ui), window.removeEventListener(u.MFV_CLOSE, di), window.removeEventListener(u.MFV_TOGGLE, fi), window.removeEventListener(u.MFV_LIVE_TOGGLE, pi), window.removeEventListener(u.MFV_PREVIEW_TOGGLE, mi), window.removeEventListener(u.MFV_NODESTREAM_TOGGLE, hi), window.removeEventListener(u.MFV_POPOUT, gi), window.removeEventListener(u.SETTINGS_CHANGED, vi), window.removeEventListener("keydown", yi, !0), window.removeEventListener("beforeunload", _i), li = !1;
}
function Si({ reinstallGlobalHandlers: e = !1 } = {}) {
	let t = !!V?.isVisible;
	try {
		V?.isPopped && V.popIn();
	} catch (e) {
		console.debug?.(e);
	}
	xi(), ni(), ci(), qr(), Gr += 1, H = Vr(), U = Hr(), W = !1;
	try {
		z && z(!1);
	} catch (e) {
		console.debug?.(e);
	}
	Yr(), t && q(!1), e && bi();
}
//#endregion
export { ut as $, Ae as $t, Zn as A, Ct as At, In as B, Rt as Bt, zn as C, Ut as Ct, tr as D, Tt as Dt, er as E, pt as Et, P as F, lt as Ft, $t as G, cn as Gt, tn as H, on as Ht, Un as I, Pt as It, en as J, pn as Jt, Jt as K, an as Kt, Jn as L, Dt as Lt, Qn as M, _t as Mt, qn as N, kt as Nt, Bn as O, xt as Ot, Gn as P, yt as Pt, Kt as Q, he as Qt, Yn as R, Gt as Rt, N as S, Wt as St, Rn as T, dt as Tt, At as U, dn as Ut, Ln as V, un as Vt, Mt as W, sn as Wt, mt as X, Le as Xt, Vt as Y, Ze as Yt, rn as Z, Pe as Zt, or as _, qt as _t, Pr as a, d as an, gt as at, Kn as b, Zt as bt, mr as c, vt as ct, _r as d, Lt as dt, Oe as en, ft as et, lr as f, Ft as ft, cr as g, Ht as gt, sr as h, Qt as ht, Mr as i, f as in, St as it, Xn as j, at as jt, Hn as k, st as kt, gr as l, zt as lt, fr as m, fn as mt, bi as n, ke as nn, bt as nt, Ar as o, Bt as ot, ur as p, Et as pt, Yt as q, ln as qt, Si as r, C as rn, ot as rt, Or as s, Ot as st, $ as t, De as tn, wt as tt, hr as u, ct as ut, nr as v, jt as vt, Vn as w, ht as wt, Wn as x, Nt as xt, rr as y, nn as yt, $n as z, It as zt };
