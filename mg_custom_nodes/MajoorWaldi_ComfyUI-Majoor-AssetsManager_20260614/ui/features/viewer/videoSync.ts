import { APP_CONFIG } from "../../app/config.js";

const _debug = () => {
    try {
        return Boolean(APP_CONFIG?.DEBUG_VIEWER);
    } catch {
        return false;
    }
};

/**
 * Sync a set of follower videos to a leader (one-way). Useful for compare modes where
 * the viewer bar controls a single "active" video.
 *
 * Must never throw. Returns an AbortController to remove listeners.
 */
export function installFollowerVideoSync(leader: any, followers: any[], { threshold = 0.15 }: { threshold?: number } = {}): AbortController | void {
    const ac = new AbortController();
    try {
        if (!leader) return ac;
        const list = Array.isArray(followers) ? followers.filter((v: any) => v && v !== leader) : [];
        if (!list.length) return ac;
        const videos = [leader, ...list].filter(Boolean);

        let syncing = false;
        const expectedPlaybackEvents = new WeakSet();

        const tryPlay = (v: any) => {
            try {
                if (v && v.paused === false) return;
                try {
                    expectedPlaybackEvents.add(v);
                } catch (e: any) {
                    console.debug?.(e);
                }
                const p = v.play?.();
                if (p && typeof p.catch === "function") p.catch(() => {});
            } catch (e: any) {
                console.debug?.(e);
            }
        };

        const syncTimeFrom = (source: any) => {
            if (syncing) return;
            try {
                const t = Number(source?.currentTime) || 0;
                for (const f of videos) {
                    if (!f || f === source) continue;
                    try {
                        if (Math.abs((Number(f.currentTime) || 0) - t) > threshold) {
                            syncing = true;
                            f.currentTime = t;
                            syncing = false;
                        }
                    } catch {
                        syncing = false;
                    }
                }
            } catch {
                syncing = false;
            }
        };

        const syncTimeToLeader = () => syncTimeFrom(leader);

        const syncPlayState = (playing: any, source = leader) => {
            if (syncing) return;
            for (const f of videos) {
                if (!f || f === source) continue;
                try {
                    if (playing) tryPlay(f);
                    else {
                        try {
                            expectedPlaybackEvents.add(f);
                        } catch (e: any) {
                            console.debug?.(e);
                        }
                        f.pause?.();
                    }
                } catch (e: any) {
                    console.debug?.(e);
                }
            }
        };

        const syncVolume = (source = leader) => {
            if (syncing) return;
            for (const f of videos) {
                if (!f || f === source) continue;
                try {
                    f.muted = Boolean(source.muted);
                    f.volume = Number(source.volume) || 0;
                } catch (e: any) {
                    console.debug?.(e);
                }
            }
        };

        const syncRate = (source = leader) => {
            if (syncing) return;
            for (const f of videos) {
                if (!f || f === source) continue;
                try {
                    f.playbackRate = Number(source.playbackRate) || 1;
                } catch (e: any) {
                    console.debug?.(e);
                }
            }
        };

        // Keep followers quiet and prevent native loop fighting the viewer bar loop logic.
        try {
            for (const f of list) {
                try {
                    f.muted = true;
                } catch (e: any) {
                    console.debug?.(e);
                }
                try {
                    f.loop = false;
                } catch (e: any) {
                    console.debug?.(e);
                }
            }
        } catch (e: any) {
            console.debug?.(e);
        }

        // Initial sync (best-effort).
        try {
            syncVolume();
            syncRate();
            syncTimeToLeader();
            if (!leader.paused) syncPlayState(true);
        } catch (e: any) {
            console.debug?.(e);
        }

        leader.addEventListener("play", () => syncPlayState(true), {
            signal: ac.signal,
            passive: true,
        });
        leader.addEventListener("pause", () => syncPlayState(false), {
            signal: ac.signal,
            passive: true,
        });
        leader.addEventListener("timeupdate", syncTimeToLeader, {
            signal: ac.signal,
            passive: true,
        });
        leader.addEventListener("seeking", syncTimeToLeader, { signal: ac.signal, passive: true });
        leader.addEventListener("seeked", syncTimeToLeader, { signal: ac.signal, passive: true });
        leader.addEventListener("ended", syncTimeToLeader, { signal: ac.signal, passive: true });
        leader.addEventListener("volumechange", syncVolume, { signal: ac.signal, passive: true });
        leader.addEventListener("ratechange", syncRate, { signal: ac.signal, passive: true });

        for (const media of list) {
            try {
                media.addEventListener("play", () => {
                    if (expectedPlaybackEvents.has(media)) {
                        expectedPlaybackEvents.delete(media);
                        return;
                    }
                    syncTimeFrom(media);
                    syncRate(media);
                    syncPlayState(true, media);
                }, {
                    signal: ac.signal,
                    passive: true,
                });
                media.addEventListener(
                    "pause",
                    () => {
                        if (expectedPlaybackEvents.has(media)) {
                            expectedPlaybackEvents.delete(media);
                            return;
                        }
                        if (media?.ended) return;
                        syncPlayState(false, media);
                    },
                    {
                        signal: ac.signal,
                        passive: true,
                    },
                );
                media.addEventListener("seeking", () => syncTimeFrom(media), {
                    signal: ac.signal,
                    passive: true,
                });
                media.addEventListener("seeked", () => syncTimeFrom(media), {
                    signal: ac.signal,
                    passive: true,
                });
                media.addEventListener("ratechange", () => syncRate(media), {
                    signal: ac.signal,
                    passive: true,
                });
            } catch (e: any) {
                console.debug?.(e);
            }
        }

        // Followers can reach "ended" and pause (we disable native looping on followers).
        // Ensure they keep looping in sync with the controlled leader.
        try {
            for (const f of list) {
                try {
                    f.addEventListener(
                        "ended",
                        () => {
                            if (syncing) return;
                            try {
                                syncing = true;
                                f.currentTime = Number(leader.currentTime) || 0;
                            } catch (e: any) {
                                console.debug?.(e);
                            } finally {
                                syncing = false;
                            }
                            try {
                                if (!leader.paused) tryPlay(f);
                            } catch (e: any) {
                                console.debug?.(e);
                            }
                        },
                        { signal: ac.signal, passive: true },
                    );
                } catch (e: any) {
                    console.debug?.(e);
                }
            }
        } catch (e: any) {
            console.debug?.(e);
        }

        // Followers might load later; resync once metadata is ready.
        try {
            for (const f of list) {
                try {
                    f.addEventListener("loadedmetadata", syncTimeToLeader, {
                        signal: ac.signal,
                        passive: true,
                        once: true,
                    });
                } catch (e: any) {
                    console.debug?.(e);
                }
            }
        } catch (e: any) {
            console.debug?.(e);
        }
    } catch (e: any) {
        if (_debug()) {
            try {
                console.warn("[Viewer] follower video sync setup failed", e);
            } catch (e: any) {
                console.debug?.(e);
            }
        }
    }
    return ac;
}
