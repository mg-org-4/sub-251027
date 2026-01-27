/**
 * WaveSpeed Predictor - Parameter parsing and processing module
 */

// Format display name
export function formatDisplayName(propName) {
    return propName
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

// Map JSON Schema type to ComfyUI type
export function mapJsonSchemaType(prop) {
    if (prop.enum && prop.enum.length > 0) {
        return "COMBO";
    }

    switch (prop.type) {
        case 'integer':
            return "INT";
        case 'number':
            return "FLOAT";
        case 'boolean':
            return "BOOLEAN";
        case 'string':
            return "STRING";
        case 'array':
            return "STRING";
        default:
            return "STRING";
    }
}

// Clean default value
export function cleanDefaultValue(defaultValue, propName) {
    if (defaultValue === undefined || defaultValue === null) {
        return defaultValue;
    }

    const paramName = propName.toLowerCase();

    if (paramName.includes('image') || paramName.includes('video') ||
        paramName.includes('audio') || paramName.includes('url')) {
        return '';
    }

    if (paramName.includes('prompt') || paramName.includes('text') ||
        paramName.includes('description')) {
        return '';
    }

    return defaultValue;
}

// Get original API type
export function getOriginalApiType(param) {
    if (param.originalType) {
        return param.originalType;
    }
    switch (param.type) {
        case 'STRING':
            return 'string';
        case 'INT':
            return 'integer';
        case 'FLOAT':
            return 'number';
        case 'BOOLEAN':
            return 'boolean';
        default:
            return 'string';
    }
}

// Check if parameter is an array type
export function isArrayParameter(paramName, apiType) {
    if (apiType !== 'string' && apiType !== 'array') {
        return false;
    }
    const lowerName = paramName.toLowerCase();
    return lowerName.endsWith('images') || lowerName.endsWith('image_urls') ||
           lowerName.endsWith('videos') || lowerName.endsWith('video_urls') ||
           lowerName.endsWith('audios') || lowerName.endsWith('audio_urls') ||
           lowerName.endsWith('loras') || lowerName.endsWith('lora_urls');
}

// Get media type
export function getMediaType(paramName, apiType) {
    if (apiType && apiType !== 'string' && apiType !== 'array') {
        return 'file';
    }

    const lowerName = paramName.toLowerCase();

    // Check plural forms (array types)
    if (lowerName.endsWith('images') || lowerName.endsWith('image_urls')) return 'image';
    if (lowerName.endsWith('videos') || lowerName.endsWith('video_urls')) return 'video';
    if (lowerName.endsWith('audios') || lowerName.endsWith('audio_urls')) return 'audio';
    if (lowerName.endsWith('loras') || lowerName.endsWith('lora_urls')) return 'lora';

    // Check singular forms
    if (lowerName.endsWith('image') || lowerName.endsWith('image_url')) return 'image';
    if (lowerName.endsWith('video') || lowerName.endsWith('video_url')) return 'video';
    if (lowerName.endsWith('audio') || lowerName.endsWith('audio_url')) return 'audio';
    if (lowerName.endsWith('lora') || lowerName.endsWith('lora_url')) return 'lora';

    return 'file';
}

// Parse model parameters
export function parseModelParameters(inputSchema) {
    if (!inputSchema?.properties) {
        return [];
    }

    const parameters = [];
    const properties = inputSchema.properties;
    const required = inputSchema.required || [];
    const order = inputSchema['x-order-properties'] || Object.keys(properties);

    for (const propName of order) {
        if (!properties[propName]) continue;

        const prop = properties[propName];
        if (prop.disabled || prop.hidden) continue;

        const param = {
            name: propName,
            displayName: formatDisplayName(propName),
            type: mapJsonSchemaType(prop),
            required: required.includes(propName),
            default: cleanDefaultValue(prop.default, propName),
            description: prop.description || "",
            isArray: prop.type === 'array',
            originalType: prop.type,
            uiComponent: prop['x-ui-component'],
            xHidden: prop['x-hidden'] === true,
        };

        if (prop.enum && prop.enum.length > 0) {
            param.options = prop.enum;
        }

        // Save min/max/step info
        if (prop.type === 'integer' || prop.type === 'number') {
            param.min = prop.minimum;
            param.max = prop.maximum;
            param.step = prop.step || (prop.type === 'integer' ? 1 : 0.01);
        }

        // Special handling for size parameters (type: string, but may have size constraints)
        // Only apply min/max for range-type size (not enum-type size)
        if (isSizeParameter(propName) && !param.options) {
            // Extract min/max from API, fallback to defaults (256-1536)
            param.min = prop.minimum !== undefined ? prop.minimum : 256;
            param.max = prop.maximum !== undefined ? prop.maximum : 1536;
        }

        // Extract maxItems info and detect object array (e.g., bbox_condition with height/length/width)
        if (prop.type === 'array' || isArrayParameter(propName, prop.type)) {
            const apiMaxItems = prop.maxItems || 5;
            param.maxItems = Math.min(apiMaxItems, 5);
            
            // Check if array items are objects (e.g., bbox_condition with height/length/width)
            // BUT: Force loras to always be string arrays (with input slots), never object arrays
            const isLorasParam = propName.toLowerCase().includes('lora');
            if (!isLorasParam && prop.items && prop.items.type === 'object' && prop.items.properties) {
                param.isObjectArray = true;
                param.objectProperties = Object.keys(prop.items.properties);
            }
        }

        parameters.push(param);
    }

    return parameters;
}

// Get ComfyUI input type
export function getComfyInputType(param) {
    const typeMap = {
        'STRING': 'STRING',
        'INT': 'INT',
        'FLOAT': 'FLOAT',
        'BOOLEAN': 'BOOLEAN',
        'LORA_WEIGHT': 'WAVESPEED_LORAS',
    };

    if (param.isArray) {
        return '*';
    }

    return typeMap[param.type] || '*';
}

// Check if parameter is a size parameter
export function isSizeParameter(paramName) {
    const lowerName = paramName.toLowerCase();
    return lowerName === 'size' ||
           lowerName === 'image_size' ||
           lowerName === 'output_size';
}

// Check if size parameter is enum type (dropdown with fixed options)
export function isSizeEnum(param) {
    return isSizeParameter(param.name) && param.options && param.options.length > 0;
}

// Check if size parameter is range type (flexible width/height inputs)
export function isSizeRange(param) {
    return isSizeParameter(param.name) && (!param.options || param.options.length === 0);
}

// Detect file input type
export function detectFileType(name) {
    const lowerName = name.toLowerCase();

    // Check plural forms (arrays)
    if (lowerName.endsWith('images') || lowerName.endsWith('image_urls')) {
        return { accept: 'image/*', type: 'file-array' };
    }
    if (lowerName.endsWith('videos') || lowerName.endsWith('video_urls')) {
        return { accept: 'video/*', type: 'file-array' };
    }
    if (lowerName.endsWith('audios') || lowerName.endsWith('audio_urls')) {
        return { accept: 'audio/*', type: 'file-array' };
    }

    // Check singular forms
    if (lowerName.endsWith('image') || lowerName.endsWith('image_url')) {
        return { accept: 'image/*', type: 'file' };
    }
    if (lowerName.endsWith('video') || lowerName.endsWith('video_url')) {
        return { accept: 'video/*', type: 'file' };
    }
    if (lowerName.endsWith('audio') || lowerName.endsWith('audio_url')) {
        return { accept: 'audio/*', type: 'file' };
    }

    return null;
}
