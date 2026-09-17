/**
 * Illustrious Smart Scene Generator UI Panel
 * Floating panel for interactive anime scene generation
 */

(function() {
    'use strict';
    try { console.log('[Illustrious Smart Panel] script loaded'); } catch {}
    if (window.__illustrious_disable_smart_panel) {
        try { console.log('[Illustrious Smart Panel] disabled by flag'); } catch {}
        return;
    }
    
    let panel = null;
    let isGenerating = false;
    
    // API helpers
    async function apiPost(path, body = {}) {
        const response = await fetch(`/illustrious/smart_prompt${path}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        
        const json = await response.json();
        if (!response.ok || json.success === false) {
            throw new Error(json.error || `HTTP ${response.status}`);
        }
        return json;
    }
    
    async function apiGet(path) {
        const response = await fetch(`/illustrious/smart_prompt${path}`);
        const json = await response.json();
        if (!response.ok || json.success === false) {
            throw new Error(json.error || `HTTP ${response.status}`);
        }
        return json;
    }
    
    // Try to import app for settings (may not be available immediately)
    let app = null;
    try {
        if (window.app) {
            app = window.app;
        }
    } catch (e) {
        // App not available yet
    }
    
    // Settings helpers
    function getSetting(key) {
        try {
            if (!app && window.app) app = window.app;
        return app?.extensionManager?.setting?.get(`illustrious.smart_scene.${key}`);
        } catch (e) {
            return null;
        }
    }
    
    function setSetting(key, value) {
        try {
        app?.extensionManager?.setting?.set(`illustrious.smart_scene.${key}`, value);
        } catch (e) {
            // Settings not available
        }
    }
    
    // Create the floating panel
    function createPanel() {
        if (panel) return;
        
        // Check if auto-show is disabled
        const autoShow = getSetting('auto_show_panel');
        if (autoShow === false) return;
        
        panel = document.createElement('div');
        panel.className = 'smart-prompt-panel';
        panel.innerHTML = `
            <div class="panel-header">
                <span class="panel-title">🎭 Illustrious Scene Generator</span>
                <div class="panel-controls">
                    <button class="minimize-btn" title="Minimize">−</button>
                    <button class="close-btn" title="Close">×</button>
                </div>
            </div>
            <div class="panel-content">
                <div class="section">
                    <h3>Scene Configuration</h3>
                    <div class="form-row">
                        <label>Category:</label>
                        <select id="category">
                            <option value="Daily Life">Daily Life</option>
                            <option value="Outdoor" selected>Outdoor</option>
                            <option value="Indoor">Indoor</option>
                            <option value="Seasonal">Seasonal</option>
                            <option value="Atmospheric">Atmospheric</option>
                            <option value="Action">Action</option>
                            <option value="Emotional">Emotional</option>
                            <option value="School Life">School Life</option>
                            <option value="Fantasy Adventure">Fantasy Adventure</option>
                            <option value="Romance">Romance</option>
                            <option value="Convention">Convention</option>
                        </select>
                    </div>
                    <div class="form-row">
                        <label>Complexity:</label>
                        <select id="complexity">
                            <option value="simple">Simple</option>
                            <option value="medium" selected>Medium</option>
                            <option value="detailed">Detailed</option>
                        </select>
                    </div>
                </div>
                
                <div class="section">
                    <h3>Include Options</h3>
                    <div class="form-row">
                        <input type="checkbox" id="include-time-weather" checked />
                        <label for="include-time-weather">Time/Weather</label>
                    </div>
                    <div class="form-row">
                        <input type="checkbox" id="include-ambience" checked />
                        <label for="include-ambience">Ambience</label>
                    </div>
                    <div class="form-row">
                        <input type="checkbox" id="include-event" />
                        <label for="include-event">Event</label>
                    </div>
                    <div class="form-row">
                        <input type="checkbox" id="include-prop" checked />
                        <label for="include-prop">Props</label>
                    </div>
                    <div class="form-row">
                        <input type="checkbox" id="include-density" />
                        <label for="include-density">Population</label>
                    </div>
                    <div class="form-row">
                        <input type="checkbox" id="include-pose" checked />
                        <label for="include-pose">Pose/Action</label>
                    </div>
                    <div class="form-row">
                        <input type="checkbox" id="include-person" />
                        <label for="include-person">Person Description</label>
                    </div>
                    <div class="form-row">
                        <input type="checkbox" id="include-clothing" />
                        <label for="include-clothing">Clothing</label>
                    </div>
                </div>
                
                <div class="section">
                    <h3>Options</h3>
                    <div class="form-row">
                        <input type="checkbox" id="use-chain-insert" checked />
                        <label for="use-chain-insert">Use Chain Insert</label>
                    </div>
                    <div class="form-row">
                        <input type="checkbox" id="strict-tags" checked />
                        <label for="strict-tags">Strict Tags</label>
                    </div>
                    <div class="form-row">
                        <input type="checkbox" id="deduplicate" checked />
                        <label for="deduplicate">De-duplicate</label>
                    </div>
                    <div class="form-row">
                        <input type="checkbox" id="safe-adult" checked />
                        <label for="safe-adult">Safe Adult Subject</label>
                    </div>
                    <div class="form-row">
                        <input type="checkbox" id="danbooru-style" checked />
                        <label for="danbooru-style">Danbooru Tag Style</label>
                    </div>
                </div>

                <div class="section">
                    <h3>Chain Prefix/Suffix</h3>
                    <div class="form-row">
                        <label>Prefix:</label>
                        <input id="prefix" placeholder="optional prefix or include CHAIN token" />
                    </div>
                    <div class="form-row">
                        <label>Suffix:</label>
                        <input id="suffix" placeholder="optional suffix" />
                    </div>
                </div>

                <div class="section">
                    <h3>TIPO Optimization</h3>
                    <div class="form-row">
                        <input type="checkbox" id="tipo-enable" checked />
                        <label for="tipo-enable">Enable TIPO</label>
                    </div>
                    <div class="form-row">
                        <label>Flavor:</label>
                        <select id="tipo-flavor">
                            <option value="balanced" selected>Balanced</option>
                            <option value="creative">Creative</option>
                            <option value="conservative">Conservative</option>
                        </select>
                    </div>
                    <div class="form-row">
                        <label>Candidates:</label>
                        <input id="tipo-candidates" type="number" min="1" max="8" value="4" />
                    </div>
                </div>
                
                <div class="section">
                    <div class="generate-controls">
                        <button id="generate-btn" class="generate-button">Generate Scene</button>
                        <button id="quick-generate" class="quick-button">Quick Generate</button>
                    </div>
                </div>
                
                <div class="section results-section" style="display: none;">
                    <h3>Generated Scene</h3>
                    <div class="result-item">
                        <label>Scene Prompt:</label>
                        <textarea id="prompt-result" readonly rows="4"></textarea>
                        <button class="copy-btn" data-target="prompt-result">📋 Copy</button>
                        <button id="paste-into-focused" class="copy-btn" title="Paste into focused input">⬇ Paste</button>
                    </div>
                    <div class="result-item">
                        <label>Negative prompt:</label>
                        <textarea id="negative-prompt-result" readonly rows="4"></textarea>
                        <button class="copy-btn" data-target="negative-prompt-result">📋 Copy</button>
                        <button id="paste-negative-prompt" class="copy-btn" title="Paste into focused input">⬇ Paste</button>
                    </div>
                </div>
                
                <div class="section stats-section" style="display: none;">
                    <h3>Statistics</h3>
                    <div id="stats-content">Loading...</div>
                </div>
            </div>
        `;
        
        // Add styles
        if (!document.getElementById('smart-prompt-styles')) {
            const styles = document.createElement('style');
            styles.id = 'smart-prompt-styles';
            styles.textContent = `
                :root {
                    --bg: #0f1115;
                    --bg-elev: #141821;
                    --border: #1f2430;
                    --muted: #2a2f3a;
                    --text: #e5e7eb;
                    --text-muted: #a0a7b4;
                    --accent: #4a90e2;
                    --accent-600: #3a78bf;
                    --accent-700: #336aa9;
                    --success: #22c55e;
                }

                .smart-prompt-panel {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    width: 380px;
                    background: var(--bg);
                    border: 1px solid var(--border);
                    border-radius: 10px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
                    z-index: 10000;
                    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial, "Apple Color Emoji", "Segoe UI Emoji";
                    font-size: 13px;
                    color: var(--text);
                    max-height: 80vh;
                    overflow: hidden;
                }

                .panel-header {
                    background: var(--bg-elev);
                    padding: 10px 14px;
                    border-bottom: 1px solid var(--border);
                    border-radius: 10px 10px 0 0;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    cursor: move;
                    user-select: none;
                }

                .panel-title {
                    font-weight: 600;
                    color: var(--text);
                }

                .panel-controls button {
                    background: transparent;
                    border: 1px solid var(--border);
                    color: var(--text-muted);
                    font-size: 14px;
                    cursor: pointer;
                    padding: 2px 8px;
                    margin-left: 6px;
                    border-radius: 6px;
                    transition: all .15s ease;
                }

                .panel-controls button:hover {
                    background: var(--muted);
                    color: var(--text);
                }

                .panel-content {
                    padding: 12px;
                    overflow-y: auto;
                    max-height: calc(80vh - 46px);
                }

                .section {
                    margin-bottom: 14px;
                    padding: 10px;
                    background: var(--bg-elev);
                    border: 1px solid var(--border);
                    border-radius: 8px;
                }

                .section h3 {
                    margin: 0 0 10px 0;
                    color: var(--text);
                    font-size: 13px;
                    font-weight: 600;
                }

                .form-row {
                    display: flex;
                    align-items: center;
                    margin-bottom: 8px;
                    gap: 10px;
                    flex-wrap: wrap;
                }

                .form-row label {
                    min-width: 90px;
                    font-size: 12px;
                    color: var(--text-muted);
                }

                .form-row select,
                .form-row input[type="text"],
                .form-row input[type="number"] {
                    flex: 1;
                    background: var(--bg);
                    border: 1px solid var(--border);
                    color: var(--text);
                    padding: 6px 8px;
                    border-radius: 6px;
                    font-size: 12px;
                    transition: border-color .15s ease, box-shadow .15s ease;
                }

                .form-row select:focus,
                .form-row input[type="text"]:focus,
                .form-row input[type="number"]:focus {
                    outline: none;
                    border-color: var(--accent);
                    box-shadow: 0 0 0 3px rgba(74,144,226,0.25);
                }

                /* Two-column layout for toggle rows */
                .section.grid { display: flex; flex-wrap: wrap; gap: 8px 12px; }
                .section.grid > h3 { flex: 0 0 100%; margin-bottom: 4px; }
                .section.grid .form-row { flex: 0 0 calc(50% - 8px); }

                /* Toggle switch (shadcn-like) */
                .ui-toggle { display:flex; align-items:center; gap:10px; width:100%; }
                .ui-toggle input[type="checkbox"] { position:absolute; opacity:0; width:0; height:0; }
                .ui-toggle .track {
                    position: relative; width: 42px; height: 24px; flex: 0 0 42px;
                    background: var(--muted); border: 1px solid var(--border); border-radius: 9999px;
                    transition: background-color .15s ease, border-color .15s ease;
                }
                .ui-toggle .track::after {
                    content: ""; position: absolute; top: 2px; left: 2px; width: 20px; height: 20px;
                    background: var(--bg); border: 1px solid var(--border); border-radius: 9999px;
                    transition: transform .15s ease, background-color .15s ease, border-color .15s ease;
                }
                .ui-toggle input[type="checkbox"]:checked + .track {
                    background: var(--accent); border-color: var(--accent-600);
                }
                .ui-toggle input[type="checkbox"]:checked + .track::after {
                    transform: translateX(18px); background: #0b1220; border-color: var(--accent-700);
                }
                .ui-toggle .label { color: var(--text); font-size:12px; }

                .generate-controls { display:flex; gap: 8px; }
                .generate-button, .quick-button {
                    flex:1; padding: 10px; border: 1px solid var(--border); border-radius: 8px; cursor: pointer;
                    font-weight: 600; transition: background-color .15s ease, border-color .15s ease, transform .02s;
                    background: var(--bg-elev); color: var(--text);
                }
                .generate-button { background: linear-gradient(180deg, var(--accent), var(--accent-600)); border-color: var(--accent-700); color: #0b1220; }
                .generate-button:hover { filter: brightness(1.05); }
                .generate-button:active { transform: translateY(1px); }
                .generate-button:disabled { filter: grayscale(0.6); opacity: 0.7; cursor: not-allowed; }
                .quick-button:hover { background: #19202b; }

                .result-item { margin-bottom: 12px; }
                .result-item label { display:block; color: var(--text-muted); font-size:12px; margin-bottom:4px; }
                .result-item textarea {
                    width:100%; background: var(--bg); border:1px solid var(--border); color: var(--text);
                    padding: 8px; border-radius: 6px; resize: vertical; font-size:12px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                }
                .copy-btn { background: var(--bg-elev); color: var(--text); border:1px solid var(--border); padding:4px 8px; border-radius:6px; cursor:pointer; font-size:11px; margin-top:4px; }
                .copy-btn:hover { background: #19202b; }
                .loading { opacity: 0.6; }
                #stats-content { font-size:11px; color: var(--text-muted); }
            `;
            document.head.appendChild(styles);
        }
        
        document.body.appendChild(panel);
        
        // Restore position if enabled
        const rememberPos = getSetting('remember_position');
        if (rememberPos !== false) {
            const x = getSetting('panel_x');
            const y = getSetting('panel_y');
            if (x !== null && y !== null) {
                panel.style.left = x + 'px';
                panel.style.top = y + 'px';
                panel.style.right = 'auto';
            }
        }
        
        // Load default settings
        loadDefaultSettings();
        
    setupEventListeners();

    // Transform legacy checkboxes into modern toggles and apply grid layout
    transformCheckboxesToToggles();
    applyGridLayoutToSections();
        loadStats();
    }
    
    function setupEventListeners() {
        if (!panel) return;
        
        // Panel controls
        panel.querySelector('.close-btn').onclick = () => closePanel();
        panel.querySelector('.minimize-btn').onclick = () => toggleMinimize();
        
        // Generate buttons
        panel.querySelector('#generate-btn').onclick = () => generatePrompt(false);
        panel.querySelector('#quick-generate').onclick = () => generatePrompt(true);
        
        // Copy buttons
    panel.querySelectorAll('.copy-btn').forEach(btn => {
            btn.onclick = () => copyToClipboard(btn.dataset.target);
        });
    const pasteBtn = panel.querySelector('#paste-into-focused');
    if (pasteBtn) pasteBtn.onclick = () => pasteIntoFocused('prompt-result');
    const pasteNegativeBtn = panel.querySelector('#paste-negative-prompt');
    if (pasteNegativeBtn) pasteNegativeBtn.onclick = () => pasteIntoFocused('negative-prompt-result');
        
        // Make panel draggable
        makeDraggable(panel, panel.querySelector('.panel-header'));
    }
    
    function transformCheckboxesToToggles() {
        // Convert rows that have [checkbox + label] into a single .ui-toggle structure
        const rows = panel.querySelectorAll('.section .form-row');
        rows.forEach(row => {
            const input = row.querySelector('input[type="checkbox"]');
            const label = row.querySelector('label');
            if (!input || !label) return;

            // Build toggle label wrapper
            const toggleLabel = document.createElement('label');
            toggleLabel.className = 'ui-toggle';

            // Ensure the label is associated
            if (input.id) toggleLabel.setAttribute('for', input.id);

            // Move input into toggleLabel, then add track and text
            toggleLabel.appendChild(input);
            const track = document.createElement('span');
            track.className = 'track';
            toggleLabel.appendChild(track);

            const text = document.createElement('span');
            text.className = 'label';
            text.textContent = label.textContent || '';
            toggleLabel.appendChild(text);

            // Replace row contents
            row.innerHTML = '';
            row.appendChild(toggleLabel);
        });
    }

    function applyGridLayoutToSections() {
        // Make Include Options and Options sections two-column grids
        const selectorMatches = [
            'Include Options',
            'Options'
        ];
        Array.from(panel.querySelectorAll('.section')).forEach(sec => {
            const titleEl = sec.querySelector('h3');
            const title = titleEl ? titleEl.textContent.trim() : '';
            if (selectorMatches.includes(title)) {
                sec.classList.add('grid');
            }
        });
    }

    async function generatePrompt(quick = false) {
        if (isGenerating) return;
        
        isGenerating = true;
        const generateBtn = panel.querySelector('#generate-btn');
        const quickBtn = panel.querySelector('#quick-generate');
        
        generateBtn.disabled = true;
        generateBtn.textContent = 'Generating...';
        quickBtn.disabled = true;
        
        try {
            const params = quick ? getQuickParams() : getFullParams();
            const result = await apiPost('/generate', params);
            
            displayResults(result);
            
        } catch (error) {
            console.error('Generation error:', error);
            alert(`Generation failed: ${error.message}`);
        } finally {
            isGenerating = false;
            generateBtn.disabled = false;
            generateBtn.textContent = 'Generate Scene';
            quickBtn.disabled = false;
        }
    }
    
    function getFullParams() {
        return {
            "Category": panel.querySelector('#category').value,
            "Complexity": panel.querySelector('#complexity').value,
            "Include Time/Weather": panel.querySelector('#include-time-weather').checked,
            "Include Ambience": panel.querySelector('#include-ambience').checked,
            "Include Event": panel.querySelector('#include-event').checked,
            "Include Prop": panel.querySelector('#include-prop').checked,
            "Include Density": panel.querySelector('#include-density').checked,
            "Include Person Description": panel.querySelector('#include-person').checked,
            "Include Pose/Action": panel.querySelector('#include-pose').checked,
            "Include Clothing": panel.querySelector('#include-clothing').checked,
            "Safe Adult Subject": panel.querySelector('#safe-adult').checked,
            "Use Chain Insert": panel.querySelector('#use-chain-insert').checked,
            "Strict Tags (no phrases)": panel.querySelector('#strict-tags').checked,
            "De-duplicate With Prefix/Suffix": panel.querySelector('#deduplicate').checked,
            "Danbooru Tag Style": panel.querySelector('#danbooru-style').checked,
            "Enable TIPO Optimization": panel.querySelector('#tipo-enable').checked,
            "TIPO Flavor": panel.querySelector('#tipo-flavor').value,
            "TIPO Candidates": parseInt(panel.querySelector('#tipo-candidates').value || '4', 10),
            "prefix": panel.querySelector('#prefix').value,
            "suffix": panel.querySelector('#suffix').value
        };
    }
    
    function getQuickParams() {
        return {
            "Category": "Outdoor",
            "Complexity": "medium",
            "Include Time/Weather": true,
            "Include Ambience": true,
            "Include Event": false,
            "Include Prop": true,
            "Include Density": false,
            "Include Person Description": false,
            "Include Pose/Action": true,
            "Include Clothing": false,
            "Safe Adult Subject": true,
            "Use Chain Insert": true,
            "Strict Tags (no phrases)": true,
            "De-duplicate With Prefix/Suffix": true,
            "Danbooru Tag Style": true,
            "Enable TIPO Optimization": true,
            "TIPO Flavor": "balanced",
            "TIPO Candidates": 4,
            "prefix": "",
            "suffix": ""
        };
    }
    
    function displayResults(result) {
        const resultsSection = panel.querySelector('.results-section');
        resultsSection.style.display = 'block';

        const promptResult = panel.querySelector('#prompt-result');
        promptResult.value = result.prompt || '';

        const negativePromptResult = panel.querySelector('#negative-prompt-result');
        negativePromptResult.value = result.negative_prompt || '';

        // Update stats if panel still exists
        if (panel && document.body.contains(panel)) {
            loadStats();
        }
    }
    
    function pasteIntoFocused(sourceId) {
        const source = panel.querySelector(`#${sourceId}`);
        const active = document.activeElement;
        if (!source || !active) return;
        const text = source.value || '';
        if (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA') {
            const start = active.selectionStart ?? active.value.length;
            const end = active.selectionEnd ?? active.value.length;
            active.setRangeText(text, start, end, 'end');
        }
    }

    async function loadStats() {
        if (!panel || !document.body.contains(panel)) return;
        try {
            const result = await apiGet('/stats');
            const statsContent = panel.querySelector('#stats-content');
            
            if (result.stats) {
                const stats = result.stats;
                statsContent.innerHTML = `
                    <div>Total: ${stats.total_generations || 0}</div>
                    <div>Avg Time: ${(stats.avg_time || 0).toFixed(3)}s</div>
                    <div>Last Updated: ${stats.last_updated || 'Never'}</div>
                `;
                panel.querySelector('.stats-section').style.display = 'block';
            }
        } catch (error) {
            console.log('Stats loading failed:', error.message);
        }
    }
    
    async function copyToClipboard(targetId) {
        const target = panel.querySelector(`#${targetId}`);
        if (target) {
            try {
                // Try modern clipboard API first
                if (navigator.clipboard && window.isSecureContext) {
                    await navigator.clipboard.writeText(target.value);
                } else {
                    // Fallback for older browsers
                    target.select();
                    document.execCommand('copy');
                }
                
                // Visual feedback
                const btn = panel.querySelector(`[data-target="${targetId}"]`);
                const originalText = btn.textContent;
                btn.textContent = '✓ Copied!';
                setTimeout(() => {
                    btn.textContent = originalText;
                }, 2000);
            } catch (err) {
                console.error('Failed to copy text: ', err);
            }
        }
    }
    
    function toggleMinimize() {
        const content = panel.querySelector('.panel-content');
        if (content.style.display === 'none') {
            content.style.display = 'block';
            panel.querySelector('.minimize-btn').textContent = '−';
        } else {
            content.style.display = 'none';
            panel.querySelector('.minimize-btn').textContent = '+';
        }
    }
    
    function closePanel() {
        if (panel) {
            // Save position if enabled
            const rememberPos = getSetting('remember_position');
            if (rememberPos !== false) {
                setSetting('panel_x', panel.offsetLeft);
                setSetting('panel_y', panel.offsetTop);
            }
            
            panel.remove();
            panel = null;
        }
    }
    
    function loadDefaultSettings() {
        if (!panel) return;
        
        // Apply default complexity
        const defaultComplexity = getSetting('default_complexity');
        if (defaultComplexity) {
            const complexitySelect = panel.querySelector('#complexity');
            if (complexitySelect) {
                complexitySelect.value = defaultComplexity;
            }
        }
    }
    
    function makeDraggable(element, handle) {
        let isDragging = false;
        let offset = { x: 0, y: 0 };
        
        handle.addEventListener('mousedown', (e) => {
            isDragging = true;
            offset.x = e.clientX - element.offsetLeft;
            offset.y = e.clientY - element.offsetTop;
            handle.style.cursor = 'grabbing';
        });
        
        document.addEventListener('mousemove', (e) => {
            if (isDragging) {
                element.style.left = (e.clientX - offset.x) + 'px';
                element.style.top = (e.clientY - offset.y) + 'px';
                element.style.right = 'auto';
            }
        });
        
        document.addEventListener('mouseup', () => {
            isDragging = false;
            handle.style.cursor = 'move';
        });
    }
    
    // Global functions for external control (renamed to avoid conflicts)
    window.smartSceneOpenPanel = createPanel;
    window.smartSceneClosePanel = closePanel;

    // Initialize panel only when API is available to avoid early DOM/state issues
    function tryInitSmartPanel() {
        try {
            if (!window.api) { setTimeout(tryInitSmartPanel, 200); return; }
            if (!panel) createPanel();
        } catch (e) {
            setTimeout(tryInitSmartPanel, 300);
        }
    }

    // Auto-create panel when page loads (with API readiness check)
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', tryInitSmartPanel);
    } else {
        tryInitSmartPanel();
    }
    
})();