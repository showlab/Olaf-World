// Synchronized video playback
document.addEventListener('DOMContentLoaded', function() {
    // Collect all sync groups
    const syncGroups = [];
    
    // Handle parent containers with sync-group class (for video-grid sections)
    document.querySelectorAll('.sync-group').forEach(group => {
        const videos = Array.from(group.querySelectorAll('video'));
        if (videos.length > 0) {
            syncGroups.push({
                name: group.getAttribute('data-group') || 'group-' + syncGroups.length,
                videos: videos,
                container: group
            });
        }
    });
    
    // Handle individual containers with data-group attribute (for comparison-grid)
    const dataGroups = {};
    document.querySelectorAll('[data-group]:not(.sync-group)').forEach(element => {
        const groupName = element.getAttribute('data-group');
        if (!dataGroups[groupName]) {
            dataGroups[groupName] = [];
        }
        const video = element.querySelector('video');
        if (video) {
            dataGroups[groupName].push(video);
        }
    });
    
    // Handle videos with data-group attribute directly
    document.querySelectorAll('video[data-group]').forEach(video => {
        const groupName = video.getAttribute('data-group');
        if (!dataGroups[groupName]) {
            dataGroups[groupName] = [];
        }
        dataGroups[groupName].push(video);
    });
    
    // Add data-groups to syncGroups
    Object.entries(dataGroups).forEach(([name, videos]) => {
        if (videos.length > 0) {
            // Find common parent for button placement
            let container = null;
            if (videos.length >= 2) {
                const firstVideo = videos[0];
                const secondVideo = videos[1];
                const firstParent = firstVideo.closest('.video-container');
                const secondParent = secondVideo.closest('.video-container');
                if (firstParent && secondParent && firstParent.parentElement === secondParent.parentElement) {
                    container = firstParent.parentElement;
                }
            }
            
            syncGroups.push({
                name: name,
                videos: videos,
                container: container,
                isVideoPair: true
            });
        }
    });
    
    // Create control buttons for each group
    syncGroups.forEach((group, index) => {
        if (group.container) {
            // For sync-groups with display: contents, find the first video container
            let buttonParent = group.container;
            const containerStyle = window.getComputedStyle(group.container);
            
            if (containerStyle.display === 'contents' && group.videos.length >= 2) {
                // For display: contents, attach button to first video container
                buttonParent = group.videos[0].closest('.video-container');
            }
            
            if (buttonParent) {
                // Add position relative to container if not already set
                const parentStyle = window.getComputedStyle(buttonParent);
                if (parentStyle.position === 'static') {
                    buttonParent.style.position = 'relative';
                }
                
                const controls = document.createElement('div');
                controls.className = 'sync-controls';
                
                // For video pairs with display: contents, position after first video
                if (containerStyle.display === 'contents' && group.videos.length === 2) {
                    controls.style.position = 'absolute';
                    controls.style.bottom = '-20px';
                    controls.style.left = '100%';
                    controls.style.marginLeft = '10px';
                    controls.style.transform = 'none';
                    controls.style.zIndex = '10';
                }
                
                controls.innerHTML = `
                    <button class="sync-btn sync-play" data-group-idx="${index}" title="Play videos together">▶</button>
                `;
                buttonParent.appendChild(controls);
            }
        }
    });
    
    // Apply sync listeners to each group
    syncGroups.forEach((group) => {
        const videos = group.videos;
        
        videos.forEach(video => {
            // Sync play
            video.addEventListener('play', function() {
                videos.forEach(v => {
                    if (v !== video && v.paused) {
                        v.play().catch(e => {});
                    }
                });
            });
            
            // Sync pause
            video.addEventListener('pause', function() {
                videos.forEach(v => {
                    if (v !== video && !v.paused) {
                        v.pause();
                    }
                });
            });
            
            // Sync seeking
            video.addEventListener('seeked', function() {
                const currentTime = video.currentTime;
                videos.forEach(v => {
                    if (v !== video && Math.abs(v.currentTime - currentTime) > 0.1) {
                        v.currentTime = currentTime;
                    }
                });
            });
        });
    });
    
    // Control button handlers
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('sync-play')) {
            const idx = parseInt(e.target.getAttribute('data-group-idx'));
            syncGroups[idx].videos.forEach(v => v.play().catch(e => {}));
        }
    });
    
    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    const dropdownTriggers = document.querySelectorAll('.navbar-item.has-dropdown .navbar-link');
    const dropdownParents = document.querySelectorAll('.navbar-item.has-dropdown');
    const dropdownCloseTimers = new Map();

    const closeDropdown = (parent) => {
        parent.classList.remove('is-active');
        const btn = parent.querySelector('.navbar-link');
        if (btn) {
            btn.setAttribute('aria-expanded', 'false');
        }
    };

    const closeAllDropdowns = () => {
        dropdownParents.forEach(parent => {
            clearTimeout(dropdownCloseTimers.get(parent));
            closeDropdown(parent);
        });
    };

    const openDropdown = (parent) => {
        clearTimeout(dropdownCloseTimers.get(parent));
        closeAllDropdowns();
        parent.classList.add('is-active');
        const btn = parent.querySelector('.navbar-link');
        if (btn) {
            btn.setAttribute('aria-expanded', 'true');
        }
    };

    const scheduleClose = (parent) => {
        clearTimeout(dropdownCloseTimers.get(parent));
        const timer = setTimeout(() => {
            closeDropdown(parent);
        }, 160);
        dropdownCloseTimers.set(parent, timer);
    };

    dropdownParents.forEach(parent => {
        const dropdown = parent.querySelector('.navbar-dropdown');

        parent.addEventListener('mouseenter', () => {
            openDropdown(parent);
        });

        parent.addEventListener('mouseleave', () => {
            scheduleClose(parent);
        });

        if (dropdown) {
            dropdown.addEventListener('mouseenter', () => {
                clearTimeout(dropdownCloseTimers.get(parent));
            });

            dropdown.addEventListener('mouseleave', () => {
                scheduleClose(parent);
            });
        }
    });

    dropdownTriggers.forEach(trigger => {
        trigger.addEventListener('click', (event) => {
            event.stopPropagation();
            const parent = trigger.closest('.navbar-item.has-dropdown');
            const isActive = parent.classList.contains('is-active');
            if (isActive) {
                closeDropdown(parent);
            } else {
                openDropdown(parent);
            }
        });

        trigger.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                trigger.click();
            }
        });
    });

    document.addEventListener('click', () => {
        closeAllDropdowns();
    });

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            closeAllDropdowns();
        }
    });
});

// Copy BibTeX to clipboard
function copyBibtex() {
    const bibtexText = `@misc{jiang2026olafworldorientinglatentactions,
      title={Olaf-World: Orienting Latent Actions for Video World Modeling}, 
      author={Yuxin Jiang and Yuchao Gu and Ivor W. Tsang and Mike Zheng Shou},
      year={2026},
      eprint={2602.10104},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.10104}, 
}`;
    
    navigator.clipboard.writeText(bibtexText).then(() => {
        const btn = document.querySelector('.copy-btn');
        const originalText = btn.textContent;
        btn.textContent = '✓ Copied!';
        btn.style.background = '#27ae60';
        
        setTimeout(() => {
            btn.textContent = originalText;
            btn.style.background = '#4a90e2';
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
        alert('Failed to copy BibTeX. Please copy manually.');
    });
}
