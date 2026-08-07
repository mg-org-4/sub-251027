/** Portable digest helpers for governed UTF-8 text contracts. */

import crypto from "node:crypto";
import fs from "node:fs";

export function stableTextDigest(filePath) {
    // IMPORTANT: normalize text newlines; raw hashing breaks frozen contracts after Windows checkout.
    const normalized = fs.readFileSync(filePath, "utf8").replace(/\r\n?/g, "\n");
    return crypto.createHash("sha256").update(normalized, "utf8").digest("hex");
}
