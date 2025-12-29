/**
 * ComfyUI-GeometryPack UV Unwrap Dynamic Parameter Management
 *
 * This extension manages the visibility of method-specific parameters
 * in the UV Unwrap node based on the selected method.
 */

import { app } from "../../../scripts/app.js";

const DEBUG = false; // Set to true to enable console logging

function log(...args) {
    if (DEBUG) {
        console.log("[GeomPack-UVUnwrap]", ...args);
    }
}

// Helper function to hide a widget (removes from array)
function hideWidget(node, widget) {
    if (!widget) return;
    if (widget._hidden) {
        log("Widget already hidden:", widget.name);
        return;
    }

    log("Hiding widget:", widget.name);

    const index = node.widgets.indexOf(widget);
    if (index === -1) {
        log("  ERROR: Widget not found in array!");
        return;
    }

    // Store original properties for restoration
    if (!widget.origType) {
        widget.origType = widget.type;
        widget.origComputeSize = widget.computeSize;
        widget.origSerializeValue = widget.serializeValue;
    }

    widget._originalIndex = index;
    widget._hidden = true;

    // Remove widget from array
    node.widgets.splice(index, 1);

    log("  Widget removed from index:", index);

    // Hide linked widgets if any
    if (widget.linkedWidgets) {
        widget.linkedWidgets.forEach(w => hideWidget(node, w));
    }
}

// Helper function to show a widget (re-inserts into array)
function showWidget(node, widget) {
    if (!widget) return;
    if (!widget._hidden) {
        log("Widget already visible:", widget.name);
        return;
    }

    log("Showing widget:", widget.name);

    // Restore original properties
    if (widget.origType) {
        widget.type = widget.origType;
        widget.computeSize = widget.origComputeSize;
        if (widget.origSerializeValue) {
            widget.serializeValue = widget.origSerializeValue;
        }
    }

    // Re-insert widget at original position
    const insertIndex = Math.min(widget._originalIndex, node.widgets.length);
    node.widgets.splice(insertIndex, 0, widget);

    log("  Widget restored to index:", insertIndex);

    widget._hidden = false;

    // Show linked widgets if any
    if (widget.linkedWidgets) {
        widget.linkedWidgets.forEach(w => showWidget(node, w));
    }
}

// Helper to update node size and redraw
function updateNodeSize(node) {
    node.setDirtyCanvas(true, true);
    if (app.graph) {
        app.graph.setDirtyCanvas(true, true);
    }

    requestAnimationFrame(() => {
        const newSize = node.computeSize();
        node.setSize([node.size[0], newSize[1]]);
        node.setDirtyCanvas(true, true);

        if (app.canvas) {
            app.canvas.setDirty(true, true);
        }

        requestAnimationFrame(() => {
            if (app.canvas) {
                app.canvas.draw(true, true);
            }
            log("Widget visibility update complete");
        });
    });
}

// Main extension registration
app.registerExtension({
    name: "geompack.uv_unwrap_dynamic",

    async nodeCreated(node) {
        // Handle UV Unwrap node
        if (node.comfyClass === "GeomPackUVUnwrap") {
            setTimeout(() => {
                this.setupUVUnwrapNode(node);
            }, 100);
        }
    },

    setupUVUnwrapNode(node) {
        log("Setting up GeomPackUVUnwrap node");

        // Find the method selector widget
        const methodWidget = node.widgets?.find(w => w.name === "method");
        if (!methodWidget) {
            log("ERROR: Method widget not found!");
            return;
        }

        log("Method widget found, current value:", methodWidget.value);

        // Find all parameter widgets
        const iterationsWidget = node.widgets?.find(w => w.name === "iterations");
        const angleLimitWidget = node.widgets?.find(w => w.name === "angle_limit");
        const islandMarginWidget = node.widgets?.find(w => w.name === "island_margin");
        const scaleToBoundsWidget = node.widgets?.find(w => w.name === "scale_to_bounds");
        const cubeSizeWidget = node.widgets?.find(w => w.name === "cube_size");
        const cylinderRadiusWidget = node.widgets?.find(w => w.name === "cylinder_radius");

        log("Found parameter widgets:", {
            iterations: !!iterationsWidget,
            angle_limit: !!angleLimitWidget,
            island_margin: !!islandMarginWidget,
            scale_to_bounds: !!scaleToBoundsWidget,
            cube_size: !!cubeSizeWidget,
            cylinder_radius: !!cylinderRadiusWidget
        });

        // Function to update widget visibility based on selected method
        const updateWidgetVisibility = (selectedMethod) => {
            log("Updating widget visibility for method:", selectedMethod);

            // Hide all optional parameters first
            hideWidget(node, iterationsWidget);
            hideWidget(node, angleLimitWidget);
            hideWidget(node, islandMarginWidget);
            hideWidget(node, scaleToBoundsWidget);
            hideWidget(node, cubeSizeWidget);
            hideWidget(node, cylinderRadiusWidget);

            // Show relevant parameters based on method
            if (selectedMethod === "xatlas" ||
                selectedMethod === "libigl_lscm" ||
                selectedMethod === "libigl_harmonic") {
                // No parameters for these methods
                log("  No parameters needed for", selectedMethod);
            } else if (selectedMethod === "libigl_arap") {
                showWidget(node, iterationsWidget);
            } else if (selectedMethod === "blender_smart") {
                showWidget(node, angleLimitWidget);
                showWidget(node, islandMarginWidget);
                showWidget(node, scaleToBoundsWidget);
            } else if (selectedMethod === "blender_cube") {
                showWidget(node, cubeSizeWidget);
                showWidget(node, scaleToBoundsWidget);
            } else if (selectedMethod === "blender_cylinder") {
                showWidget(node, cylinderRadiusWidget);
                showWidget(node, scaleToBoundsWidget);
            } else if (selectedMethod === "blender_sphere") {
                showWidget(node, scaleToBoundsWidget);
            }

            updateNodeSize(node);
        };

        // Store original callback
        const origCallback = methodWidget.callback;

        // Override callback to update visibility when method changes
        methodWidget.callback = function(value) {
            log("Method changed to:", value);

            // Call original callback if it exists
            const result = origCallback?.apply(this, arguments);

            // Update widget visibility
            updateWidgetVisibility(value);

            return result;
        };

        // Initialize visibility on node creation
        log("Initializing widget visibility...");
        updateWidgetVisibility(methodWidget.value);
    }
});

log("GeomPack UV Unwrap dynamic parameters extension loaded");
