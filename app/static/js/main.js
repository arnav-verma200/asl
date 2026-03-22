setInterval(() => {
    fetch('/state')
        .then(r => r.json())
        .then(d => {
            document.getElementById('letter-display').textContent = d.predicted_letter || '—';
            document.getElementById('conf').textContent     = 'Confidence: ' + d.confidence + '%';
            document.getElementById('word').textContent     = d.current_word || '_';
            document.getElementById('sentence').textContent = d.sentence     || '...';

            const s     = document.getElementById('hand-status');
            s.textContent = d.hand_detected ? 'Hand Detected' : 'No Hand';
            s.className   = d.hand_detected ? 'ok' : '';
        });
}, 300);

function addWord()      { fetch('/add_word',      { method: 'POST' }); }
function deleteLetter() { fetch('/delete_letter', { method: 'POST' }); }
function speakSentence(){ fetch('/speak',         { method: 'POST' }); }
function clearAll()     { fetch('/clear',         { method: 'POST' }); }

function switchCamera(mode) {
    fetch('/switch_camera', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ mode })
    });
    document.getElementById('btn-laptop').classList.toggle('active', mode === 'laptop');
    document.getElementById('btn-phone').classList.toggle('active',  mode === 'phone');
}

document.addEventListener('keydown', e => {
    if (e.code === 'Space')     { e.preventDefault(); addWord(); }
    if (e.code === 'Enter')     speakSentence();
    if (e.code === 'KeyC')      clearAll();
    if (e.code === 'Backspace') deleteLetter();
});