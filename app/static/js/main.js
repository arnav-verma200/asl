// Poll server for state every 300ms
setInterval(updateState, 300);

function updateState() {
    fetch('/state')
        .then(res => res.json())
        .then(data => {

            // Update predicted letter
            const letterEl = document.getElementById('predicted-letter');
            letterEl.textContent = data.predicted_letter || '—';

            // Update confidence bar
            const bar  = document.getElementById('confidence-bar');
            const text = document.getElementById('confidence-text');
            bar.style.width    = data.confidence + '%';
            text.textContent   = `Confidence: ${data.confidence}%`;

            // Update word
            const wordEl = document.getElementById('current-word');
            wordEl.textContent = data.current_word || '_';

            // Update sentence
            const sentenceEl = document.getElementById('sentence');
            sentenceEl.textContent = data.sentence ||
                                     'Your sentence will appear here...';

            // Update hand status
            const statusEl = document.getElementById('hand-status');
            if (data.hand_detected) {
                statusEl.textContent  = '✋ Hand Detected';
                statusEl.className    = 'hand-status hand-detected';
            } else {
                statusEl.textContent  = 'No Hand Detected';
                statusEl.className    = 'hand-status no-hand';
            }

            // Flash letter when new letter added
            if (data.predicted_letter) {
                letterEl.style.color = '#00ff88';
            } else {
                letterEl.style.color = '#333333';
            }
        })
        .catch(err => console.log('State update error:', err));
}

function addWord() {
    fetch('/add_word', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            document.getElementById('sentence').textContent =
                data.sentence || 'Your sentence will appear here...';
            document.getElementById('current-word').textContent = '_';
            showFeedback('Word added! 🎉');
        });
}

function speakSentence() {
    fetch('/speak', { method: 'POST' })
        .then(() => showFeedback('Speaking... 🔊'));
}

function clearAll() {
    fetch('/clear', { method: 'POST' })
        .then(() => {
            document.getElementById('current-word').textContent  = '_';
            document.getElementById('sentence').textContent =
                'Your sentence will appear here...';
            document.getElementById('predicted-letter').textContent = '—';
            document.getElementById('confidence-bar').style.width   = '0%';
            showFeedback('Cleared! 🗑️');
        });
}

function deleteLetter() {
    fetch('/delete_letter', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            document.getElementById('current-word').textContent =
                data.word || '_';
        });
}

function switchCamera(mode) {
    fetch('/switch_camera', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: mode })
    })
    .then(() => {
        // Update toggle buttons
        document.getElementById('btn-laptop').classList.remove('active');
        document.getElementById('btn-phone').classList.remove('active');
        document.getElementById(`btn-${mode}`).classList.add('active');
        showFeedback(`Switched to ${mode} camera 📷`);
    });
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    if (e.code === 'Space') {
        e.preventDefault();
        addWord();
    } else if (e.code === 'Enter') {
        speakSentence();
    } else if (e.code === 'KeyC') {
        clearAll();
    } else if (e.code === 'Backspace') {
        deleteLetter();
    }
});

// Feedback toast
function showFeedback(message) {
    let toast = document.getElementById('toast');
    if (!toast) {
        toast           = document.createElement('div');
        toast.id        = 'toast';
        toast.style.cssText = `
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: #00ff8820;
            color: #00ff88;
            border: 1px solid #00ff88;
            padding: 12px 24px;
            border-radius: 10px;
            font-weight: 600;
            z-index: 9999;
            transition: opacity 0.3s;
        `;
        document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.style.opacity = '1';
    setTimeout(() => { toast.style.opacity = '0'; }, 2000);
}