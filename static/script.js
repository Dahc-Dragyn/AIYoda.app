// /static/script.js

// NO DOMContentLoaded needed because of 'defer'

// Set the current year in the footer.
document.getElementById("currentYear").textContent = new Date().getFullYear();

// Initialize the Ko-Fi widget.
if (typeof kofiWidgetOverlay !== 'undefined') {
    kofiWidgetOverlay.draw('aiyoda', {
        'type': 'floating-button',
        'floating-button.donateButton.text': 'Support Us',
        'floating-button.donateButton.background-color': '#5bc0de',
        'floating-button.donateButton.text-color': '#323842'
    });
} else {
    console.error('kofiWidgetOverlay is not defined. Ensure the Ko-Fi script is loaded correctly.');
}