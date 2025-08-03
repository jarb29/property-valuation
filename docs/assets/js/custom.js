// Modern interactions
document.addEventListener('DOMContentLoaded', function() {
    // Smooth scroll
    document.querySelectorAll('a[href^="#"]').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
    
    // Progress bar
    const progress = document.createElement('div');
    progress.style.cssText = `
        position: fixed; top: 0; left: 0; width: 0%; height: 2px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        z-index: 9999; transition: width 0.1s;
    `;
    document.body.appendChild(progress);
    
    window.addEventListener('scroll', () => {
        const scrolled = (window.pageYOffset / (document.body.scrollHeight - window.innerHeight)) * 100;
        progress.style.width = scrolled + '%';
    });
});