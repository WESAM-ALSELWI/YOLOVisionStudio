// Basic feature maps mock
window.renderFeatureMaps = function(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = `<div style="text-align:center; padding: 40px; color: var(--text-muted);">Feature map data not loaded.</div>`;
}
