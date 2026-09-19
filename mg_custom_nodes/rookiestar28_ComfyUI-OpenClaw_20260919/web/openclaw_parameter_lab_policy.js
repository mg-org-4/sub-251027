export const PARAMETER_LAB_POLICY = Object.freeze({
    version: "1.0",
    maxRequestBytes: 5 * 1024 * 1024,
    maxWorkflowUtf8Bytes: 4 * 1024 * 1024,
    maxSweepDimensions: 8,
    maxValuesPerDimension: 50,
    maxNodeIdUtf8Bytes: 128,
    maxWidgetNameUtf8Bytes: 256,
    maxScalarStringUtf8Bytes: 16 * 1024,
    maxPlanUtf8Bytes: 8 * 1024 * 1024,
    maxSweepCombinations: 50,
    maxCompareItems: 8,
});

const encoder = new TextEncoder();
const valid = () => ({ ok: true, reason: "" });
const invalid = (reason) => ({ ok: false, reason });

function utf8Size(value) {
    return encoder.encode(value).byteLength;
}

function hasControlCharacters(value) {
    return Array.from(value).some((character) => {
        const code = character.charCodeAt(0);
        return code < 32 || code === 127;
    });
}

function validateNodeId(nodeId) {
    if (nodeId === null || nodeId === undefined || nodeId === "") {
        return invalid("node_id_required");
    }
    if (
        typeof nodeId !== "string" &&
        !(typeof nodeId === "number" && Number.isInteger(nodeId))
    ) {
        return invalid("invalid_node_id");
    }
    const normalized = String(nodeId);
    if (!normalized.trim()) {
        return invalid("node_id_required");
    }
    // IMPORTANT: replay keys use the first "." as the node/widget separator.
    if (normalized.includes(".") || hasControlCharacters(normalized)) {
        return invalid("invalid_node_id");
    }
    if (utf8Size(normalized) > PARAMETER_LAB_POLICY.maxNodeIdUtf8Bytes) {
        return invalid("node_id_too_large");
    }
    return valid();
}

function validateWidgetName(widgetName) {
    if (typeof widgetName !== "string" || !widgetName.trim()) {
        return invalid("widget_name_required");
    }
    if (hasControlCharacters(widgetName)) {
        return invalid("invalid_widget_name");
    }
    if (utf8Size(widgetName) > PARAMETER_LAB_POLICY.maxWidgetNameUtf8Bytes) {
        return invalid("widget_name_too_large");
    }
    return valid();
}

export function isParameterLabScalar(value) {
    return (
        typeof value === "string" ||
        typeof value === "boolean" ||
        (typeof value === "number" && Number.isFinite(value))
    );
}

function validateScalar(value) {
    if (!isParameterLabScalar(value)) {
        return invalid("invalid_scalar_value");
    }
    if (
        typeof value === "string" &&
        utf8Size(value) > PARAMETER_LAB_POLICY.maxScalarStringUtf8Bytes
    ) {
        return invalid("scalar_string_too_large");
    }
    return valid();
}

export function validateParameterLabScalar(value) {
    return validateScalar(value);
}

export function filterParameterLabCandidates(candidates) {
    if (!Array.isArray(candidates)) {
        return [];
    }
    const filtered = [];
    const presentations = new Set();
    for (const candidate of candidates) {
        if (!validateScalar(candidate).ok) {
            continue;
        }
        const presentation = String(candidate);
        if (presentations.has(presentation)) {
            continue;
        }
        presentations.add(presentation);
        filtered.push(candidate);
    }
    return filtered;
}

export function validateParameterLabDimensions(dimensions) {
    if (!Array.isArray(dimensions) || dimensions.length === 0) {
        return invalid("dimensions_required");
    }
    if (dimensions.length > PARAMETER_LAB_POLICY.maxSweepDimensions) {
        return invalid("too_many_dimensions");
    }

    const dimensionKeys = new Set();
    let combinations = 1;
    for (const dimension of dimensions) {
        if (!dimension || typeof dimension !== "object" || Array.isArray(dimension)) {
            return invalid("invalid_dimension");
        }
        const nodeResult = validateNodeId(dimension.node_id);
        if (!nodeResult.ok) {
            return nodeResult;
        }
        const widgetResult = validateWidgetName(dimension.widget_name);
        if (!widgetResult.ok) {
            return widgetResult;
        }
        const dimensionKey = `${String(dimension.node_id)}\u0000${dimension.widget_name}`;
        if (dimensionKeys.has(dimensionKey)) {
            return invalid("duplicate_dimension");
        }
        dimensionKeys.add(dimensionKey);

        if (!Array.isArray(dimension.values) || dimension.values.length === 0) {
            return invalid("values_required");
        }
        if (dimension.values.length > PARAMETER_LAB_POLICY.maxValuesPerDimension) {
            return invalid("too_many_values");
        }
        const presentations = new Set();
        for (const value of dimension.values) {
            const scalarResult = validateScalar(value);
            if (!scalarResult.ok) {
                return scalarResult;
            }
            const presentation = String(value);
            if (presentations.has(presentation)) {
                return invalid("duplicate_ambiguous_value");
            }
            presentations.add(presentation);
        }

        const strategy = dimension.strategy || "grid";
        if (strategy !== "grid" && strategy !== "compare") {
            return invalid("invalid_strategy");
        }
        combinations *= dimension.values.length;
        if (combinations > PARAMETER_LAB_POLICY.maxSweepCombinations) {
            return invalid("sweep_too_large");
        }
    }
    return valid();
}

export function validateParameterLabWorkflow(workflowJson) {
    if (typeof workflowJson !== "string" || !workflowJson.trim()) {
        return invalid("workflow_required");
    }
    if (utf8Size(workflowJson) > PARAMETER_LAB_POLICY.maxWorkflowUtf8Bytes) {
        return invalid("workflow_too_large");
    }
    return valid();
}

export function validateParameterLabRequestBody(payload) {
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
        return invalid("invalid_payload");
    }
    let serialized;
    try {
        serialized = JSON.stringify(payload);
    } catch {
        return invalid("invalid_payload");
    }
    if (typeof serialized !== "string") {
        return invalid("invalid_payload");
    }
    if (utf8Size(serialized) > PARAMETER_LAB_POLICY.maxRequestBytes) {
        return invalid("payload_too_large");
    }
    return valid();
}
