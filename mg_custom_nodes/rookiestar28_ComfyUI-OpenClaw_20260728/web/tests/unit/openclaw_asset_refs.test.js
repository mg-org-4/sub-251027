import { describe, expect, it } from "vitest";

import {
    extractHistoryImageRefs,
    extractHistoryOutputRefs,
    isHdrImageFilename,
    isHdrImageOutputRef,
    normalizeComfyOutputRef,
} from "../../openclaw_asset_refs.js";

describe("openclaw asset refs", () => {
    it("keeps classic history refs on the /view filename+type contract", () => {
        expect(
            normalizeComfyOutputRef({
                filename: "result.png",
                subfolder: "session-a",
                type: "temp",
            })
        ).toEqual({
            filename: "result.png",
            subfolder: "session-a",
            type: "temp",
            media_type: "images",
            asset_hash: "",
            asset_api_id: "",
            asset_api_required: false,
            resolution: "view",
            unsupported_reason: "",
            is_asset_backed: false,
            content: "",
            text_truncated: false,
            viewParams: {
                filename: "result.png",
                subfolder: "session-a",
                type: "temp",
            },
        });
    });

    it("prefers asset hashes while keeping display filename metadata", () => {
        expect(
            normalizeComfyOutputRef({
                filename: "preview.png",
                type: "output",
                asset_hash: "blake3:abc123",
            })
        ).toEqual({
            filename: "preview.png",
            subfolder: "",
            type: "output",
            media_type: "images",
            asset_hash: "blake3:abc123",
            asset_api_id: "",
            asset_api_required: false,
            resolution: "view",
            unsupported_reason: "",
            is_asset_backed: true,
            content: "",
            text_truncated: false,
            viewParams: {
                filename: "blake3:abc123",
            },
        });
    });

    it("keeps filename refs previewable when host omits hash metadata", () => {
        expect(
            normalizeComfyOutputRef({
                filename: "filename-only.png",
                subfolder: "session-a",
                type: "output",
                asset: {
                    id: "asset-without-hash",
                },
            })
        ).toEqual({
            filename: "filename-only.png",
            subfolder: "session-a",
            type: "output",
            media_type: "images",
            asset_hash: "",
            asset_api_id: "asset-without-hash",
            asset_api_required: false,
            resolution: "view",
            unsupported_reason: "",
            is_asset_backed: true,
            content: "",
            text_truncated: false,
            viewParams: {
                filename: "filename-only.png",
                subfolder: "session-a",
                type: "output",
            },
        });
    });

    it("accepts upload-style nested asset metadata", () => {
        expect(
            normalizeComfyOutputRef({
                name: "uploaded.png",
                asset: {
                    asset_hash: "blake3:def456",
                },
            })
        ).toEqual({
            filename: "uploaded.png",
            subfolder: "",
            type: "output",
            media_type: "images",
            asset_hash: "blake3:def456",
            asset_api_id: "",
            asset_api_required: false,
            resolution: "view",
            unsupported_reason: "",
            is_asset_backed: true,
            content: "",
            text_truncated: false,
            viewParams: {
                filename: "blake3:def456",
            },
        });
    });

    it("accepts top-level hash as an asset_hash alias", () => {
        expect(
            normalizeComfyOutputRef({
                filename: "hash-alias.png",
                hash: "blake3:alias123",
            })
        ).toEqual({
            filename: "hash-alias.png",
            subfolder: "",
            type: "output",
            media_type: "images",
            asset_hash: "blake3:alias123",
            asset_api_id: "",
            asset_api_required: false,
            resolution: "view",
            unsupported_reason: "",
            is_asset_backed: true,
            content: "",
            text_truncated: false,
            viewParams: {
                filename: "blake3:alias123",
            },
        });
    });

    it("accepts nested asset.hash as an asset_hash alias", () => {
        expect(
            normalizeComfyOutputRef({
                name: "nested-hash-alias.png",
                asset: {
                    hash: "blake3:nested-alias",
                },
            })
        ).toEqual({
            filename: "nested-hash-alias.png",
            subfolder: "",
            type: "output",
            media_type: "images",
            asset_hash: "blake3:nested-alias",
            asset_api_id: "",
            asset_api_required: false,
            resolution: "view",
            unsupported_reason: "",
            is_asset_backed: true,
            content: "",
            text_truncated: false,
            viewParams: {
                filename: "blake3:nested-alias",
            },
        });
    });

    it("keeps asset-api-only refs explicit instead of silently turning them into /api/assets fetches", () => {
        expect(
            normalizeComfyOutputRef({
                asset: {
                    id: "asset-only-42",
                },
            })
        ).toEqual({
            filename: "asset-only-42",
            subfolder: "",
            type: "output",
            media_type: "images",
            asset_hash: "",
            asset_api_id: "asset-only-42",
            asset_api_required: true,
            resolution: "asset_api_required",
            unsupported_reason: "asset_api_required",
            is_asset_backed: true,
            content: "",
            text_truncated: false,
            viewParams: null,
        });
    });

    it("extracts mixed history outputs without dropping temp classifications", () => {
        expect(
            extractHistoryImageRefs({
                outputs: {
                    "1": {
                        images: [
                            {
                                filename: "classic.png",
                                subfolder: "",
                                type: "output",
                            },
                            {
                                filename: "temp-preview.png",
                                subfolder: "preview",
                                type: "temp",
                                asset_hash: "blake3:temp123",
                            },
                        ],
                    },
                },
            })
        ).toEqual([
            {
                filename: "classic.png",
                subfolder: "",
                type: "output",
                media_type: "images",
                asset_hash: "",
                asset_api_id: "",
                asset_api_required: false,
                resolution: "view",
                unsupported_reason: "",
                is_asset_backed: false,
                content: "",
                text_truncated: false,
                viewParams: {
                    filename: "classic.png",
                    type: "output",
                },
            },
            {
                filename: "temp-preview.png",
                subfolder: "preview",
                type: "temp",
                media_type: "images",
                asset_hash: "blake3:temp123",
                asset_api_id: "",
                asset_api_required: false,
                resolution: "view",
                unsupported_reason: "",
                is_asset_backed: true,
                content: "",
                text_truncated: false,
                viewParams: {
                    filename: "blake3:temp123",
                },
            },
        ]);
    });

    it("extracts previewable media outputs while keeping image-only wrapper compatibility", () => {
        const historyItem = {
            outputs: {
                "1": {
                    images: [{ filename: "classic.png", type: "output" }],
                    video: [{ filename: "clip.webm", type: "output", format: "video/webm" }],
                    audio: [{ filename: "sound.wav", type: "output" }],
                    "3d": ["mesh.glb"],
                    text: ["hello text"],
                },
            },
        };

        const outputs = extractHistoryOutputRefs(historyItem);
        expect(outputs.map((output) => output.media_type)).toEqual([
            "images",
            "video",
            "audio",
            "3d",
            "text",
        ]);
        expect(outputs[1]).toEqual(expect.objectContaining({
            filename: "clip.webm",
            media_type: "video",
            viewParams: { filename: "clip.webm", type: "output" },
        }));
        expect(outputs[3]).toEqual(expect.objectContaining({
            filename: "mesh.glb",
            media_type: "3d",
        }));
        expect(outputs[4]).toEqual(expect.objectContaining({
            media_type: "text",
            content: "hello text",
            resolution: "inline_text",
            viewParams: null,
        }));

        expect(extractHistoryImageRefs(historyItem).map((output) => output.media_type)).toEqual([
            "images",
        ]);
    });

    it("keeps 3d dimension metadata from widening asset-hash view routing", () => {
        const output = normalizeComfyOutputRef(
            {
                filename: "scene.glb",
                subfolder: "previews",
                type: "output",
                media_type: "3d",
                asset_hash: "blake3:mesh123",
                width: 1200,
                height: 800,
                metadata: { width: 1200, height: 800 },
            },
            "3d"
        );

        expect(output).toEqual(expect.objectContaining({
            filename: "scene.glb",
            media_type: "3d",
            asset_hash: "blake3:mesh123",
            asset_api_required: false,
            resolution: "view",
            is_asset_backed: true,
        }));
        expect(output.viewParams).toEqual({ filename: "blake3:mesh123" });
        expect(output.unsupported_reason).toBe("");
    });

    it("bounds inline text output previews", () => {
        const longText = "x".repeat(1100);
        const output = extractHistoryOutputRefs({ outputs: { "1": { text: [longText] } } })[0];

        expect(output.media_type).toBe("text");
        expect(output.content).toHaveLength(1024);
        expect(output.text_truncated).toBe(true);
    });

    it("normalizes the official files/result.txt shape as file-backed text", () => {
        const outputs = extractHistoryOutputRefs({
            outputs: {
                "9": {
                    files: [{ filename: "result.txt", subfolder: "", type: "output" }],
                    text: "some generated text",
                },
            },
        });

        expect(outputs).toHaveLength(1);
        expect(outputs[0]).toEqual(expect.objectContaining({
            filename: "result.txt",
            media_type: "text",
            resolution: "view",
            content: "",
            text_truncated: false,
            viewParams: { filename: "result.txt", type: "output" },
        }));
    });

    it("accepts only bounded allowlisted text file refs", () => {
        const suffixes = ["txt", "md", "markdown", "json", "csv", "yaml", "yml", "xml", "log"];
        const files = [
            ...suffixes.map((suffix) => ({ filename: `result.${suffix.toUpperCase()}`, type: "output" })),
            { filename: "image.png", type: "output" },
            { filename: "archive.bin", type: "output" },
            { filename: "README", type: "output" },
            "result.txt",
            null,
        ];

        const outputs = extractHistoryOutputRefs({ outputs: { "9": { files } } });

        expect(outputs).toHaveLength(suffixes.length);
        expect(outputs.every((output) => output.media_type === "text")).toBe(true);
        expect(outputs.some((output) => output.media_type === "images")).toBe(false);
    });

    it("fails closed for oversized files containers and unsafe file fields", () => {
        const oversized = Array.from({ length: 65 }, (_, index) => ({
            filename: `result-${index}.txt`,
            type: "output",
        }));
        expect(extractHistoryOutputRefs({ outputs: { "9": { files: oversized } } })).toEqual([]);

        const invalidRefs = [
            { filename: `${"x".repeat(1021)}.txt`, type: "output" },
            { filename: "result.txt", subfolder: "x".repeat(1025), type: "output" },
            { filename: "../result.txt", type: "output" },
            { filename: "folder/result.txt", type: "output" },
            { filename: "result.txt", subfolder: "../private", type: "output" },
            { filename: "result.txt", subfolder: "/absolute", type: "output" },
            { filename: "result.txt", type: "unknown" },
            { filename: 123, type: "output" },
        ];
        for (const ref of invalidRefs) {
            expect(extractHistoryOutputRefs({ outputs: { "9": { files: [ref] } } })).toEqual([]);
        }
    });

    it("builds file text view params only from normalized fields", () => {
        const output = extractHistoryOutputRefs({
            outputs: {
                "9": {
                    files: [{
                        filename: "report 1.txt",
                        subfolder: "reports/2026",
                        type: "temp",
                        url: "https://evil.example/secret.txt",
                    }],
                },
            },
        })[0];

        expect(output.viewParams).toEqual({
            filename: "report 1.txt",
            subfolder: "reports/2026",
            type: "temp",
        });
        expect(JSON.stringify(output)).not.toContain("evil.example");
    });

    it("counts Unicode code points consistently and rejects trim-based bound bypasses", () => {
        expect(extractHistoryOutputRefs({
            outputs: { "9": { files: [{ filename: `${"😀".repeat(1020)}.txt`, type: "output" }] } },
        })).toHaveLength(1);

        const rejected = [
            { filename: `${"😀".repeat(1021)}.txt`, type: "output" },
            { filename: `${" ".repeat(1025)}safe.txt`, type: "output" },
            { filename: "safe.txt", subfolder: `${" ".repeat(1025)}reports`, type: "output" },
        ];
        for (const ref of rejected) {
            expect(extractHistoryOutputRefs({ outputs: { "9": { files: [ref] } } })).toEqual([]);
        }
    });

    it("detects HDR image refs by filename suffix without treating hashes as HDR", () => {
        expect(isHdrImageFilename("render.EXR")).toBe(true);
        expect(isHdrImageFilename("studio.hdr")).toBe(true);
        expect(isHdrImageFilename("studio.hdr.png")).toBe(false);

        expect(isHdrImageOutputRef({
            filename: "render.exr",
            media_type: "images",
            view_url: "/view?filename=blake3:abc123",
        })).toBe(true);
        expect(isHdrImageOutputRef({
            filename: "blake3:abc123",
            media_type: "images",
            view_url: "/view?filename=blake3:abc123",
        })).toBe(false);
        expect(isHdrImageOutputRef({
            filename: "",
            media_type: "images",
            view_url: "/view?filename=preview.hdr&type=output",
        })).toBe(true);
        expect(isHdrImageOutputRef({
            filename: "mesh.exr",
            media_type: "3d",
            view_url: "/view?filename=mesh.exr&type=output",
        })).toBe(false);
    });
});
