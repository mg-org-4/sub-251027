// Utility to parse and format metadata for the Gallery preview
import type { FileDetails } from './types';

export function parseMetadata(details: FileDetails) {
    if (!details || !details.metadata) return {};
    const { metadata } = details;
    const fileinfo = metadata.fileinfo || {};
    const prompt = metadata.prompt || {};
    const workflow = metadata.workflow || {};

    // Try to extract fields as in gallery.js
    return {
        Filename: fileinfo.filename || details.name,
        Resolution: fileinfo.resolution,
        'File Size': fileinfo.size,
        'Date Created': fileinfo.date || details.date,
        Model: prompt.model,
        'Positive Prompt': prompt.positive,
        'Negative Prompt': prompt.negative,
        Sampler: prompt.sampler,
        Scheduler: prompt.scheduler,
        Steps: prompt.steps,
        'CFG Scale': prompt.cfg_scale,
        Seed: prompt.seed,
        LoRAs: prompt.loras,
        // Add more fields as needed
        // Raw: metadata
    };
}

// Ported from gallery.js: robust, fallback-rich metadata parser
export function parseComfyMetadata(metadata: Record<string, any>): Record<string, string> {
    if (!metadata) return {};
    // Defensive parse for workflow/prompt JSON strings
    let prompt: Record<string, any> | undefined = metadata.prompt;
    let workflow: Record<string, any> | undefined = metadata.workflow;
    try { if (typeof prompt === 'string') prompt = JSON.parse(prompt); } catch { }
    try { if (typeof workflow === 'string') workflow = JSON.parse(workflow); } catch { }

    // File info
    const fileinfo: Record<string, any> = metadata.fileinfo || {};
    const result: Record<string, string> = {};
    result["Filename"] = fileinfo.filename || '';
    result["Resolution"] = fileinfo.resolution || '';
    result["File Size"] = fileinfo.size || '';
    result["Date Created"] = fileinfo.date || '';

    // --- Workflow/Prompt Extraction (matches gallery.js) ---
    let workflowToParse: any = null;
    if (workflow && typeof workflow === 'object' && workflow.nodes && Array.isArray(workflow.nodes)) {
        workflowToParse = workflow.nodes;
    } else if (workflow && typeof workflow === 'object') {
        workflowToParse = workflow;
    } else if (prompt && typeof prompt === 'object') {
        workflowToParse = prompt;
    }

    // Helper: parseWorkflow (port from gallery.js)
    function parseWorkflow(wf: any): Record<string, any> {
        // Simple object fallback (ComfyUI default prompt structure)
        if (!Array.isArray(wf) && typeof wf === 'object') {
            const extracted: Record<string, any> = {};
            for (const key in wf) {
                if (wf[key]?.inputs?.ckpt_name) extracted['Model'] = wf[key].inputs.ckpt_name;
                else if (wf[key]?.inputs?.text && (key === '2' || key === '7')) extracted['Positive Prompt'] = wf[key].inputs.text;
                else if (wf[key]?.inputs?.text && (key === '3' || key === '8')) extracted['Negative Prompt'] = wf[key].inputs.text;
                else if (wf[key]?.inputs?.sampler_name) extracted['Sampler'] = wf[key].inputs.sampler_name;
                else if (wf[key]?.inputs?.scheduler) extracted['Scheduler'] = wf[key].inputs.scheduler;
                else if (wf[key]?.inputs?.steps) extracted['Steps'] = wf[key].inputs.steps;
                else if (wf[key]?.inputs?.cfg) extracted['CFG Scale'] = wf[key].inputs.cfg;
                else if (wf[key]?.inputs?.seed) extracted['Seed'] = wf[key].inputs.seed;
            }
            return extracted;
        }
        // Array of nodes (advanced workflows)
        const parsingConfig: Record<string, any> = {
            Model: {
                type: ["CheckpointLoaderSimple", "CheckpointLoader|pysssss"],
                extract: (node: any) => node.widgets_values?.[0]?.content || node.widgets_values?.[0] || null,
            },
            "Positive Prompt": {
                type: ["CR Prompt Text", "CLIPTextEncode", "ImpactWildcardProcessor", "Textbox", "easy showAnything"],
                extract: (node: any) => {
                    if (node.title === "Positive Prompt") return node.widgets_values?.[0] || null;
                    if (node.type === "CLIPTextEncode" && node.inputs?.find((input: any) => input.name === "text")) return node.widgets_values?.[0] || null;
                    if (node.type === "ImpactWildcardProcessor") return node.widgets_values?.[1] || null;
                    if (node.type === "Textbox") return node.widgets_values?.[0] || null;
                    if (node.type === "easy showAnything") return node.widgets_values?.[0]?.[0] || null;
                    return null;
                },
                getColor: (node: any) => node.color,
                getBgColor: (node: any) => node.bgcolor,
                getText: (node: any) => node.widgets_values?.[0]?.[0],
            },
            "Negative Prompt": {
                type: ["CR Prompt Text", "CLIPTextEncode", "Textbox", "easy showAnything"],
                extract: (node: any) => {
                    if (node.title === "Negative Prompt") return node.widgets_values?.[0] || null;
                    if (node.type === "CLIPTextEncode" && node.inputs?.find((input: any) => input.name === "text")) return node.widgets_values?.[0] || null;
                    if (node.type === "Textbox") return node.widgets_values?.[0] || null;
                    if (node.type === "easy showAnything") return node.widgets_values?.[0]?.[0] || null;
                    return null;
                },
                getColor: (node: any) => node.color,
                getBgColor: (node: any) => node.bgcolor,
                getText: (node: any) => node.widgets_values?.[0]?.[0],
            },
            Sampler: {
                type: ["KSampler", "SamplerCustom", "FaceDetailerPipe", "Ultimate SD Upscale"],
                extract: (node: any) => {
                    if (node.type === "KSampler") return node.widgets_values?.[4] || null;
                    if (node.type === "SamplerCustom") return node.inputs?.find((input: any) => input.name === "sampler")?.widget?.name || null;
                    if (node.type === "FaceDetailerPipe") return node.widgets_values?.[7] || null;
                    if (node.type === "Ultimate SD Upscale") return node.widgets_values?.[5] || null;
                    return null;
                },
            },
            Scheduler: {
                type: ["KSampler", "KarrasScheduler", "FaceDetailerPipe", "Ultimate SD Upscale"],
                extract: (node: any) => {
                    if (node.type === "KSampler") return node.widgets_values?.[5] || null;
                    if (node.type === "KarrasScheduler") return "karras";
                    if (node.type === "FaceDetailerPipe") return node.widgets_values?.[8] || null;
                    if (node.type === "Ultimate SD Upscale") return node.widgets_values?.[6] || null;
                    return null;
                },
            },
            Steps: {
                type: ["KSampler", "KarrasScheduler", "FaceDetailerPipe", "Ultimate SD Upscale"],
                extract: (node: any) => {
                    if (node.type === "KSampler") return node.widgets_values?.[2] || null;
                    if (node.type === "KarrasScheduler") return node.widgets_values?.[0] || null;
                    if (node.type === "FaceDetailerPipe") return node.widgets_values?.[5] || null;
                    if (node.type === "Ultimate SD Upscale") return node.widgets_values?.[2] || null;
                    return null;
                },
            },
            "CFG Scale": {
                type: ["KSampler", "SamplerCustom", "FaceDetailerPipe", "Ultimate SD Upscale"],
                extract: (node: any) => {
                    if (node.type === "KSampler") return node.widgets_values?.[3] || null;
                    if (node.type === "SamplerCustom") return node.inputs?.find((input: any) => input.name === "cfg")?.value || null;
                    if (node.type === "FaceDetailerPipe") return node.widgets_values?.[6] || null;
                    if (node.type === "Ultimate SD Upscale") return node.widgets_values?.[4] || null;
                    return null;
                },
            },
            Seed: {
                type: ["KSampler", "Seed Generator", "SamplerCustom", "FaceDetailerPipe", "Ultimate SD Upscale", "ImpactWildcardProcessor"],
                extract: (node: any) => {
                    if (node.type === "KSampler") return node.widgets_values?.[0] || null;
                    if (node.type === "Seed Generator") return node.widgets_values?.[0] || null;
                    if (node.type === "SamplerCustom") return node.inputs?.find((input: any) => input.name === "noise_seed")?.widget?.value || null;
                    if (node.type === "FaceDetailerPipe") return node.widgets_values?.[3] || null;
                    if (node.type === "Ultimate SD Upscale") return node.widgets_values?.[1] || null;
                    if (node.type === "ImpactWildcardProcessor") return node.widgets_values?.[3] || null;
                    return null;
                },
            },
            LoRAs: {
                type: ["LoraLoader", "Power Lora Loader (rgthree)"],
                extract: (node: any) => {
                    const loras: any[] = [];
                    if (node.type === "LoraLoader") {
                        if (node.widgets_values && node.widgets_values.length >= 3) {
                            loras.push({
                                name: node.widgets_values[0],
                                model_strength: node.widgets_values[1],
                                clip_strength: node.widgets_values[2],
                            });
                        }
                    } else if (node.type === "Power Lora Loader (rgthree)") {
                        if (node.widgets_values) {
                            for (let i = 1; i <= 9; i++) {
                                if (node.widgets_values[i] && node.widgets_values[i].on) {
                                    loras.push({
                                        name: node.widgets_values[i].lora,
                                        strength: node.widgets_values[i].strength,
                                    });
                                }
                            }
                        }
                    }
                    return loras.length > 0 ? loras : null;
                },
            },
        };
        const extractedData: Record<string, any> = {};
        if (Array.isArray(wf)) {
            for (const key in parsingConfig) {
                extractedData[key] = [];
                const config = parsingConfig[key];
                const seenValues = new Set();
                for (const node of wf) {
                    if (config.type.includes(node.type)) {
                        if (key !== "Positive Prompt" && key !== "Negative Prompt") {
                            const extractedValue = config.extract(node);
                            if (extractedValue !== null && extractedValue !== undefined) {
                                if (key === "LoRAs" && Array.isArray(extractedValue)) {
                                    extractedValue.forEach((lora: any) => {
                                        if (!seenValues.has(lora.name)) {
                                            extractedData[key].push(lora);
                                            seenValues.add(lora.name);
                                        }
                                    });
                                } else if (!seenValues.has(extractedValue)) {
                                    extractedData[key].push(extractedValue);
                                    seenValues.add(extractedValue);
                                }
                            }
                        } else if (node.title === key) {
                            const extractedValue = config.extract(node);
                            if (extractedValue !== null && extractedValue !== undefined) {
                                if (!seenValues.has(extractedValue)) {
                                    extractedData[key].push(extractedValue);
                                    seenValues.add(extractedValue);
                                }
                            }
                        }
                    }
                }
                if (extractedData[key].length === 0) {
                    extractedData[key] = null;
                } else if (extractedData[key].length === 1 && key !== "LoRAs") {
                    extractedData[key] = extractedData[key][0];
                }
            }
            // Prompt inference fallback (for easy showAnything, CLIPTextEncode, etc)
            const promptInference: Record<string, any> = {
                "Positive Prompt": {
                    colorPrefixes: ["#232", "#2"],
                    bgColorPrefixes: ["#353", "#3"],
                    keywords: ["positive", "prompt", "masterpiece", "best quality", "detailed"],
                },
                "Negative Prompt": {
                    colorPrefixes: ["#322", "#533", "#3", "#5"],
                    bgColorPrefixes: ["#533", "#653", "#5", "#6"],
                    keywords: ["negative", "prompt", "unrealistic", "bad", "worst quality", "low quality", "unwanted"],
                },
            };
            for (const promptType in promptInference) {
                if (extractedData[promptType] === null) {
                    extractedData[promptType] = [];
                    const seenValues = new Set();
                    for (const node of wf) {
                        const config = parsingConfig[promptType];
                        if (config.type.includes(node.type) && !["CR Prompt Text"].includes(node.type)) {
                            let extractedValue = null;
                            const color = config.getColor(node);
                            const bgColor = config.getBgColor(node);
                            const text = config.getText(node)?.toLowerCase() || "";
                            const colorMatch = promptInference[promptType].colorPrefixes.some((prefix: string) => color?.startsWith(prefix));
                            const bgColorMatch = promptInference[promptType].bgColorPrefixes.some((prefix: string) => bgColor?.startsWith(prefix));
                            const keywordMatch = promptInference[promptType].keywords.some((keyword: string) => text.includes(keyword));
                            if (colorMatch || bgColorMatch || keywordMatch) {
                                extractedValue = config.extract(node);
                            }
                            if (extractedValue !== null && extractedValue !== undefined) {
                                if (!seenValues.has(extractedValue)) {
                                    extractedData[promptType].push(extractedValue);
                                    seenValues.add(extractedValue);
                                }
                            }
                        }
                    }
                    if (extractedData[promptType].length === 0) {
                        extractedData[promptType] = null;
                    } else if (extractedData[promptType].length === 1) {
                        extractedData[promptType] = extractedData[promptType][0];
                    }
                }
            }
        }
        return extractedData;
    }

    const parsed = workflowToParse ? parseWorkflow(workflowToParse) : {};
    result["Model"] = parsed['Model'] || prompt?.['1']?.inputs?.ckpt_name || prompt?.['1']?.inputs?.ckpt_name?.content || '';
    result["Positive Prompt"] = parsed['Positive Prompt'] || prompt?.['2']?.inputs?.prompt || prompt?.['7']?.inputs?.text || '';
    result["Negative Prompt"] = parsed['Negative Prompt'] || prompt?.['3']?.inputs?.prompt || prompt?.['8']?.inputs?.text || '';
    result["Sampler"] = parsed['Sampler'] || prompt?.['10']?.inputs?.sampler_name || '';
    result["Scheduler"] = parsed['Scheduler'] || prompt?.['10']?.inputs?.scheduler || '';
    result["Steps"] = parsed['Steps'] || prompt?.['10']?.inputs?.steps || '';
    result["CFG Scale"] = parsed['CFG Scale'] || prompt?.['10']?.inputs?.cfg || '';
    result["Seed"] = parsed['Seed'] || prompt?.['10']?.inputs?.seed || '';
    // LoRAs
    let loras: string[] = [];
    if (parsed['LoRAs']) {
        if (Array.isArray(parsed['LoRAs'])) {
            parsed['LoRAs'].forEach((lora: any) => {
                if (typeof lora === 'object' && lora.name) {
                    loras.push(`${lora.name} (Model: ${lora.model_strength ?? lora.strength}, Clip: ${lora.clip_strength ?? 'N/A'})`);
                } else if (typeof lora === 'string') {
                    loras.push(lora);
                }
            });
        } else if (typeof parsed['LoRAs'] === 'object' && parsed['LoRAs'].name) {
            loras.push(`${parsed['LoRAs'].name} (Model: ${parsed['LoRAs'].model_strength ?? parsed['LoRAs'].strength}, Clip: ${parsed['LoRAs'].clip_strength ?? 'N/A'})`);
        }
    } else if (prompt) {
        for (const key in prompt) {
            if (prompt[key].class_type === 'LoraLoader') {
                loras.push(prompt[key].inputs.lora_name);
            } else if (prompt[key].class_type === 'Power Lora Loader (rgthree)') {
                for (let loraKey in prompt[key].inputs) {
                    if (loraKey.startsWith('lora_') && prompt[key].inputs[loraKey].on) {
                        loras.push(prompt[key].inputs[loraKey].lora);
                    }
                }
            }
        }
    }
    result["LoRAs"] = loras.length > 0 ? loras.join(', ') : 'N/A';
    return result;
}
