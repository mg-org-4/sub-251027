/**
 * WaveSpeed Predictor - Media upload and preview module
 */

import { api } from "../../../../scripts/api.js";

// Upload file to WaveSpeed server
export async function uploadToWaveSpeed(data, type, filename = null) {
    try {
        const formData = new FormData();
        formData.append('type', type);

        if (type === 'url') {
            formData.append('url', data);
        } else {
            formData.append('file', data, filename || 'uploaded_file');
        }

        const baseUrl = window.location.origin;
        const uploadUrl = `${baseUrl}/wavespeed/api/upload`;

        console.log('[WaveSpeed] Uploading to local proxy:', uploadUrl);

        const response = await fetch(uploadUrl, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('[WaveSpeed Upload] Server response:', response.status, errorText);
            throw new Error(`Upload failed: ${response.statusText}`);
        }

        const result = await response.json();
        if (result.success && result.data?.url) {
            console.log('[WaveSpeed] Upload successful:', result.data.url);
            return { success: true, url: result.data.url, filename: result.data.filename };
        } else {
            return { success: false, error: result.error || 'Upload failed' };
        }
    } catch (error) {
        console.error('[WaveSpeed Upload] Error:', error);
        return { success: false, error: error.message };
    }
}

// Create file preview element
export function createFilePreview(url, mediaType, onDelete) {
    const previewContainer = document.createElement('div');
    previewContainer.style.position = 'relative';
    previewContainer.style.display = 'inline-block';
    previewContainer.style.marginRight = '2px';
    previewContainer.style.borderRadius = '3px';
    previewContainer.style.overflow = 'hidden';
    previewContainer.style.border = '1px solid #444';
    previewContainer.style.cursor = 'pointer';
    previewContainer.style.transition = 'all 0.2s ease';
    previewContainer.style.flexShrink = '0';
    previewContainer.style.width = '40px';
    previewContainer.style.height = '40px';

    let previewElement;
    if (mediaType === 'image') {
        previewElement = document.createElement('img');
        previewElement.src = url;
        previewElement.style.width = '40px';
        previewElement.style.height = '40px';
        previewElement.style.objectFit = 'cover';
        previewElement.style.display = 'block';

        previewElement.addEventListener('error', function(e) {
            console.error('[WaveSpeed] Image preview load error:', e);
            const errorPlaceholder = document.createElement('div');
            errorPlaceholder.style.width = '40px';
            errorPlaceholder.style.height = '40px';
            errorPlaceholder.style.backgroundColor = '#2a2a2a';
            errorPlaceholder.style.display = 'flex';
            errorPlaceholder.style.alignItems = 'center';
            errorPlaceholder.style.justifyContent = 'center';
            errorPlaceholder.style.fontSize = '16px';
            errorPlaceholder.textContent = '🖼️';
            errorPlaceholder.title = 'Image (preview unavailable)';
            previewElement.replaceWith(errorPlaceholder);
        });
    } else if (mediaType === 'video') {
        previewElement = document.createElement('video');
        previewElement.src = url;
        previewElement.style.width = '40px';
        previewElement.style.height = '40px';
        previewElement.style.objectFit = 'cover';
        previewElement.style.display = 'block';
        previewElement.muted = true;
        previewElement.preload = 'metadata';

        previewElement.addEventListener('loadeddata', function() {
            this.currentTime = 0.1;
        });

        previewElement.addEventListener('error', function(e) {
            console.error('[WaveSpeed] Video preview load error:', e);
            const errorPlaceholder = document.createElement('div');
            errorPlaceholder.style.width = '40px';
            errorPlaceholder.style.height = '40px';
            errorPlaceholder.style.backgroundColor = '#2a2a2a';
            errorPlaceholder.style.display = 'flex';
            errorPlaceholder.style.alignItems = 'center';
            errorPlaceholder.style.justifyContent = 'center';
            errorPlaceholder.style.fontSize = '16px';
            errorPlaceholder.textContent = '🎬';
            errorPlaceholder.title = 'Video (preview unavailable)';
            previewElement.replaceWith(errorPlaceholder);
        });
    } else if (mediaType === 'audio') {
        previewElement = document.createElement('div');
        previewElement.style.width = '40px';
        previewElement.style.height = '40px';
        previewElement.style.backgroundColor = '#2a2a2a';
        previewElement.style.display = 'flex';
        previewElement.style.alignItems = 'center';
        previewElement.style.justifyContent = 'center';
        previewElement.style.fontSize = '16px';
        previewElement.textContent = '🎵';
    } else {
        // Default placeholder for unknown media types (e.g., lora, file)
        previewElement = document.createElement('div');
        previewElement.style.width = '40px';
        previewElement.style.height = '40px';
        previewElement.style.backgroundColor = '#2a2a2a';
        previewElement.style.display = 'flex';
        previewElement.style.alignItems = 'center';
        previewElement.style.justifyContent = 'center';
        previewElement.style.fontSize = '16px';
        previewElement.textContent = '📄';
        previewElement.title = 'File';
    }

    // CRITICAL FIX: Ensure previewElement is always a valid DOM node before appending
    if (previewElement && previewElement instanceof Node) {
        previewContainer.appendChild(previewElement);
    } else {
        console.error('[WaveSpeed] Failed to create previewElement for mediaType:', mediaType);
        // Create fallback placeholder
        const fallbackElement = document.createElement('div');
        fallbackElement.style.width = '40px';
        fallbackElement.style.height = '40px';
        fallbackElement.style.backgroundColor = '#2a2a2a';
        fallbackElement.style.display = 'flex';
        fallbackElement.style.alignItems = 'center';
        fallbackElement.style.justifyContent = 'center';
        fallbackElement.style.fontSize = '16px';
        fallbackElement.textContent = '📄';
        previewContainer.appendChild(fallbackElement);
    }

    // Delete button - show on hover
    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = '×';
    deleteBtn.style.position = 'absolute';
    deleteBtn.style.top = '-2px';
    deleteBtn.style.right = '-2px';
    deleteBtn.style.width = '16px';
    deleteBtn.style.height = '16px';
    deleteBtn.style.backgroundColor = '#ff4444';
    deleteBtn.style.color = 'white';
    deleteBtn.style.border = 'none';
    deleteBtn.style.borderRadius = '50%';
    deleteBtn.style.cursor = 'pointer';
    deleteBtn.style.fontSize = '12px';
    deleteBtn.style.lineHeight = '1';
    deleteBtn.style.padding = '0';
    deleteBtn.style.opacity = '0';
    deleteBtn.style.transition = 'opacity 0.2s ease';

    previewContainer.appendChild(deleteBtn);

    // Hover effect
    previewContainer.onmouseenter = () => {
        previewContainer.style.borderColor = '#4a9eff';
        deleteBtn.style.opacity = '1';
    };
    previewContainer.onmouseleave = () => {
        previewContainer.style.borderColor = '#444';
        deleteBtn.style.opacity = '0';
    };

    // Delete handler
    deleteBtn.onclick = (e) => {
        e.stopPropagation();
        if (onDelete) onDelete();
    };

    // Click to enlarge
    previewContainer.onclick = () => {
        showMediaModal(url, mediaType);
    };

    previewContainer.deleteBtn = deleteBtn;
    previewContainer.onClickHandler = () => showMediaModal(url, mediaType);

    return previewContainer;
}

// Show media modal
export function showMediaModal(url, mediaType) {
    const modal = document.createElement('div');
    modal.style.position = 'fixed';
    modal.style.top = '0';
    modal.style.left = '0';
    modal.style.width = '100%';
    modal.style.height = '100%';
    modal.style.backgroundColor = 'rgba(0, 0, 0, 0.9)';
    modal.style.display = 'flex';
    modal.style.alignItems = 'center';
    modal.style.justifyContent = 'center';
    modal.style.zIndex = '10000';
    modal.style.cursor = 'pointer';

    let mediaElement;
    if (mediaType === 'image') {
        mediaElement = document.createElement('img');
        mediaElement.src = url;
        mediaElement.style.maxWidth = '90%';
        mediaElement.style.maxHeight = '90%';
        mediaElement.style.objectFit = 'contain';
    } else if (mediaType === 'video') {
        mediaElement = document.createElement('video');
        mediaElement.src = url;
        mediaElement.style.maxWidth = '90%';
        mediaElement.style.maxHeight = '90%';
        mediaElement.controls = true;
        mediaElement.autoplay = true;
    } else if (mediaType === 'audio') {
        mediaElement = document.createElement('audio');
        mediaElement.src = url;
        mediaElement.style.width = '80%';
        mediaElement.controls = true;
        mediaElement.autoplay = true;
    }

    mediaElement.style.cursor = 'default';
    mediaElement.onclick = (e) => e.stopPropagation();

    modal.appendChild(mediaElement);

    modal.onclick = () => {
        document.body.removeChild(modal);
    };

    document.body.appendChild(modal);
}

// Create loading preview element
export function createLoadingPreview(filename) {
    const loadingContainer = document.createElement('div');
    loadingContainer.style.position = 'relative';
    loadingContainer.style.display = 'inline-flex';
    loadingContainer.style.marginRight = '2px';
    loadingContainer.style.borderRadius = '3px';
    loadingContainer.style.overflow = 'hidden';
    loadingContainer.style.border = '1px solid #4a9eff';
    loadingContainer.style.backgroundColor = '#2a2a2a';
    loadingContainer.style.width = '40px';
    loadingContainer.style.height = '40px';
    loadingContainer.style.alignItems = 'center';
    loadingContainer.style.justifyContent = 'center';

    // Spinner
    const spinner = document.createElement('div');
    spinner.style.width = '20px';
    spinner.style.height = '20px';
    spinner.style.border = '2px solid #444';
    spinner.style.borderTop = '2px solid #4a9eff';
    spinner.style.borderRadius = '50%';
    spinner.style.animation = 'spin 1s linear infinite';

    // Add CSS animation
    if (!document.getElementById('wavespeed-spinner-style')) {
        const style = document.createElement('style');
        style.id = 'wavespeed-spinner-style';
        style.textContent = `
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        `;
        document.head.appendChild(style);
    }

    loadingContainer.appendChild(spinner);
    loadingContainer.title = `Uploading ${filename}`;

    return loadingContainer;
}

// Create error preview element
export function createErrorPreview(errorMessage) {
    const errorContainer = document.createElement('div');
    errorContainer.style.position = 'relative';
    errorContainer.style.display = 'inline-flex';
    errorContainer.style.marginRight = '2px';
    errorContainer.style.borderRadius = '3px';
    errorContainer.style.overflow = 'hidden';
    errorContainer.style.border = '1px solid #ff4444';
    errorContainer.style.backgroundColor = '#2a2a2a';
    errorContainer.style.width = '40px';
    errorContainer.style.height = '40px';
    errorContainer.style.alignItems = 'center';
    errorContainer.style.justifyContent = 'center';
    errorContainer.style.cursor = 'pointer';

    const icon = document.createElement('div');
    icon.style.fontSize = '16px';
    icon.textContent = '⚠️';

    errorContainer.appendChild(icon);
    errorContainer.title = `Error: ${errorMessage}`;

    errorContainer.onclick = () => {
        errorContainer.remove();
    };

    return errorContainer;
}

// Create upload button
export function createUploadButton(onFileSelected, mediaType = 'file') {
    const uploadBtn = document.createElement('button');
    uploadBtn.textContent = '📁';
    uploadBtn.title = 'Upload file';
    uploadBtn.style.padding = '4px 8px';
    uploadBtn.style.backgroundColor = '#4a9eff';
    uploadBtn.style.color = 'white';
    uploadBtn.style.border = 'none';
    uploadBtn.style.borderRadius = '4px';
    uploadBtn.style.cursor = 'pointer';
    uploadBtn.style.fontSize = '14px';
    uploadBtn.style.marginLeft = '4px';
    uploadBtn.style.transition = 'background-color 0.2s ease';

    uploadBtn.onmouseenter = () => {
        uploadBtn.style.backgroundColor = '#3a8eef';
    };
    uploadBtn.onmouseleave = () => {
        uploadBtn.style.backgroundColor = '#4a9eff';
    };

    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.style.display = 'none';

    if (mediaType === 'image') {
        fileInput.accept = 'image/*';
    } else if (mediaType === 'video') {
        fileInput.accept = 'video/*';
    } else if (mediaType === 'audio') {
        fileInput.accept = 'audio/*';
    } else {
        fileInput.accept = 'image/*,video/*,audio/*';
    }

    uploadBtn.onclick = () => {
        fileInput.click();
    };

    fileInput.onchange = async (e) => {
        const file = e.target.files[0];
        if (file && onFileSelected) {
            await onFileSelected(file);
        }
        fileInput.value = '';
    };

    uploadBtn.appendChild(fileInput);

    return uploadBtn;
}