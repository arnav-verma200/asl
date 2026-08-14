let lastWord = '';

// Main polling interval for state
setInterval(() => {
    fetch('/state')
        .then(r => r.json())
        .then(d => {
            // Update predicted letter
            document.getElementById('letter-display').textContent = d.predicted_letter || '—';
            
            // Update confidence percentage and progress bar
            document.getElementById('conf-text').textContent = d.confidence + '%';
            document.getElementById('confidence-bar').style.width = d.confidence + '%';
            
            // Update word composition
            const currentWord = d.current_word || '';
            document.getElementById('word').textContent = currentWord || '_';
            
            // Only fetch word suggestions if the current word has changed
            if (currentWord !== lastWord) {
                lastWord = currentWord;
                fetchSuggestions(currentWord);
            }
            
            // Update sentence constructor
            document.getElementById('sentence').textContent = d.sentence || '...';

            // Update hand status indicator badge
            const statusBadge = document.getElementById('hand-status');
            if (d.hand_detected) {
                statusBadge.textContent = 'Hand Detected';
                statusBadge.classList.add('ok');
            } else {
                statusBadge.textContent = 'No Hand Detected';
                statusBadge.classList.remove('ok');
            }
        })
        .catch(err => console.error('Failed to poll state:', err));
}, 300);

// Basic translation commands
function addWord()      { fetch('/add_word',      { method: 'POST' }); }
function deleteLetter() { fetch('/delete_letter', { method: 'POST' }); }
function speakSentence(){ fetch('/speak',         { method: 'POST' }); }
function clearAll()     { fetch('/clear',         { method: 'POST' }); }

// Switch camera mode
function switchCamera(mode) {
    fetch('/switch_camera', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ mode })
    })
    .then(() => {
        document.getElementById('btn-laptop').classList.toggle('active', mode === 'laptop');
        document.getElementById('btn-phone').classList.toggle('active',  mode === 'phone');
    })
    .catch(err => console.error('Failed to switch camera:', err));
}

// Global hotkeys matching interface controls
document.addEventListener('keydown', e => {
    // Avoid triggering controls if user is typing elsewhere (if inputs are ever added)
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        return;
    }
    
    if (e.code === 'Space') {
        e.preventDefault();
        addWord();
    }
    if (e.code === 'Enter') {
        e.preventDefault();
        speakSentence();
    }
    if (e.code === 'KeyC') {
        clearAll();
    }
    if (e.code === 'Backspace') {
        deleteLetter();
    }
});

// Suggestions retrieval
function fetchSuggestions(prefix) {
    if (!prefix) {
        document.getElementById('suggestions').innerHTML = '';
        return;
    }

    fetch('/suggestions?prefix=' + prefix)
        .then(r => r.json())
        .then(d => {
            const container = document.getElementById('suggestions');
            container.innerHTML = '';

            if (d.suggestions && d.suggestions.length > 0) {
                d.suggestions.forEach(word => {
                    const btn = document.createElement('button');
                    btn.textContent = word;
                    btn.onclick = () => useSuggestion(word);
                    container.appendChild(btn);
                });
            }
        })
        .catch(err => console.error('Failed to fetch suggestions:', err));
}

// Use suggestion
function useSuggestion(word) {
    fetch('/use_suggestion', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ word })
    })
    .then(() => {
        document.getElementById('word').textContent = '_';
        document.getElementById('suggestions').innerHTML = '';
        lastWord = ''; // Reset word tracker after word is added
    })
    .catch(err => console.error('Failed to use suggestion:', err));
}