let activeGridContainer = null;

function isLiveElement(element) {
    if (!element) return false;
    if (typeof element?.isConnected === "boolean") return element.isConnected;
    return true;
}

function findGridInDom() {
    try {
        const grid =
            document.getElementById("mjr-assets-grid") || document.querySelector(".mjr-grid");
        if (isLiveElement(grid)) return grid;
    } catch (e) {
        console.debug?.(e);
    }
    return null;
}

export function setActiveGridContainer(container) {
    activeGridContainer = container || null;
    return activeGridContainer;
}

export function clearActiveGridContainer(container = null) {
    if (!container || activeGridContainer === container) {
        activeGridContainer = null;
    }
}

export function getActiveGridContainer() {
    if (isLiveElement(activeGridContainer)) {
        return activeGridContainer;
    }
    activeGridContainer = null;
    return findGridInDom();
}
