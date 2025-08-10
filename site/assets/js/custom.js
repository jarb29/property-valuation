// Custom JavaScript for GitHub Pages compatibility
document.addEventListener('DOMContentLoaded', function() {
    // Ensure CSS variables are applied
    const root = document.documentElement;
    
    // Force CSS custom properties to be recognized
    if (getComputedStyle(root).getPropertyValue('--primary') === '') {
        root.style.setProperty('--primary', '#6366f1');
        root.style.setProperty('--secondary', '#8b5cf6');
        root.style.setProperty('--accent', '#06b6d4');
    }
    
    // Add a class to indicate custom styles are loaded
    document.body.classList.add('custom-styles-loaded');
});