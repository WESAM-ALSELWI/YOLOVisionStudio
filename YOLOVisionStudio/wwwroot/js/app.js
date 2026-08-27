// Global App JavaScript
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Feather Icons
    if (typeof feather !== 'undefined') {
        feather.replace();
    }

    // Sidebar Toggle
    const sidebarToggle = document.querySelector('.sidebar-toggle');
    const sidebar = document.querySelector('.sidebar');
    if (sidebarToggle && sidebar) {
        sidebarToggle.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
            localStorage.setItem('sidebar-collapsed', sidebar.classList.contains('collapsed'));
        });
        if (localStorage.getItem('sidebar-collapsed') === 'true') {
            sidebar.classList.add('collapsed');
        }
    }

    // Theme Toggle
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            updateThemeIcon(newTheme);
        });
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme) {
            document.documentElement.setAttribute('data-theme', savedTheme);
            updateThemeIcon(savedTheme);
        }
    }

    function updateThemeIcon(theme) {
        if (!themeToggle) return;
        const icon = themeToggle.querySelector('i');
        if (icon && typeof feather !== 'undefined') {
            icon.setAttribute('data-feather', theme === 'dark' ? 'sun' : 'moon');
            feather.replace();
        }
    }

    // Count Up Animation
    const counters = document.querySelectorAll('.counter');
    counters.forEach(counter => {
        const target = +counter.getAttribute('data-target');
        const duration = 2000; // ms
        const increment = target / (duration / 16); // 60fps
        let current = 0;
        const updateCounter = () => {
            current += increment;
            if (current < target) {
                counter.innerText = Math.ceil(current);
                requestAnimationFrame(updateCounter);
            } else {
                counter.innerText = target;
            }
        };
        updateCounter();
    });

    // Toast System
    window.showToast = (title, message, type = 'info') => {
        let container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            document.body.appendChild(container);
        }
        
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        let iconName = 'info';
        if (type === 'success') iconName = 'check-circle';
        else if (type === 'error') iconName = 'alert-circle';
        else if (type === 'warning') iconName = 'alert-triangle';

        toast.innerHTML = `
            <div class="toast-icon"><i data-feather="${iconName}"></i></div>
            <div class="toast-content">
                <div class="toast-title">${title}</div>
                <div class="toast-message">${message}</div>
            </div>
        `;
        
        container.appendChild(toast);
        if (typeof feather !== 'undefined') feather.replace();
        
        // Trigger reflow for animation
        toast.offsetHeight;
        toast.classList.add('show');
        
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    };

    // Keyboard Shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.querySelector('.topbar-search input');
            if (searchInput) searchInput.focus();
        }
    });

    // SignalR Setup (Mocked for Demo if not fully integrated)
    setupSignalR();
});

function setupSignalR() {
    // If SignalR is available
    if (typeof signalR !== 'undefined') {
        const connection = new signalR.HubConnectionBuilder()
            .withUrl("/hubs/training")
            .withAutomaticReconnect()
            .build();

        const statusDot = document.getElementById('connection-status');

        connection.on("ReceiveMetrics", (metrics) => {
            document.dispatchEvent(new CustomEvent('TrainingMetricsReceived', { detail: metrics }));
        });

        connection.start().then(() => {
            if(statusDot) statusDot.classList.add('pulse-dot');
        }).catch(err => {
            console.error(err.toString());
            if(statusDot) statusDot.style.backgroundColor = 'var(--color-error)';
        });
    } else {
        // Mock SignalR for demo purposes
        console.log("SignalR not found. Running mock metrics.");
        setInterval(() => {
            const mockMetrics = {
                epoch: Math.floor(Math.random() * 100),
                box_loss: Math.random() * 2,
                cls_loss: Math.random() * 2,
                dfl_loss: Math.random() * 1.5,
                map50: 0.5 + Math.random() * 0.4,
                map5095: 0.3 + Math.random() * 0.4
            };
            document.dispatchEvent(new CustomEvent('TrainingMetricsReceived', { detail: mockMetrics }));
        }, 2000);
    }
}
