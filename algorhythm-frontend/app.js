/* ═══════════════════════════════════════════════
   ALGORHYTHM — Frontend Logic
   ═══════════════════════════════════════════════ */

const API_BASE = 'http://localhost:8000';

// DOM refs
const stepInput = document.getElementById('step-input');
const stepTracks = document.getElementById('step-tracks');
const stepResults = document.getElementById('step-results');

const playlistForm = document.getElementById('playlist-form');
const playlistUrlInput = document.getElementById('playlist-url');
const analyzeBtn = document.getElementById('analyze-btn');
const errorToast = document.getElementById('error-toast');
const errorMessage = document.getElementById('error-message');

const playlistTitle = document.getElementById('playlist-title');
const trackCountLabel = document.getElementById('track-count-label');
const trackList = document.getElementById('track-list');
const selectedCount = document.getElementById('selected-count');
const selectAllBtn = document.getElementById('select-all-btn');
const deselectAllBtn = document.getElementById('deselect-all-btn');
const backBtn = document.getElementById('back-btn');
const buildDnaBtn = document.getElementById('build-dna-btn');

const dnaPlaylistName = document.getElementById('dna-playlist-name');
const dnaResults = document.getElementById('dna-results');
const restartBtn = document.getElementById('restart-btn');

// State
let tracks = [];

// ─── Key name mapping ───
const KEY_NAMES = {
    0: 'C', 1: 'C♯/D♭', 2: 'D', 3: 'D♯/E♭', 4: 'E', 5: 'F',
    6: 'F♯/G♭', 7: 'G', 8: 'G♯/A♭', 9: 'A', 10: 'A♯/B♭', 11: 'B'
};

const MODE_NAMES = { 0: 'Minor', 1: 'Major' };

// ─── Helpers ───

function showError(msg) {
    errorMessage.textContent = msg;
    errorToast.hidden = false;
    setTimeout(() => { errorToast.hidden = true; }, 6000);
}

function hideError() {
    errorToast.hidden = true;
}

function setLoading(btn, loading) {
    const text = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.btn-loader');
    if (loading) {
        text.hidden = true;
        loader.hidden = false;
        btn.disabled = true;
    } else {
        text.hidden = false;
        loader.hidden = true;
        btn.disabled = false;
    }
}

function showStep(step) {
    stepInput.hidden = true;
    stepTracks.hidden = true;
    stepResults.hidden = true;
    step.hidden = false;
    // Re-trigger animation
    step.style.animation = 'none';
    step.offsetHeight; // force reflow
    step.style.animation = '';
}

function updateSelectionCount() {
    const checked = document.querySelectorAll('.track-checkbox:checked').length;
    selectedCount.textContent = checked;
}

// ─── Step 1: Analyze Playlist ───

playlistForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    hideError();

    const url = playlistUrlInput.value.trim();
    if (!url) {
        showError('Please paste a Spotify playlist URL.');
        return;
    }

    if (!url.includes('spotify.com/playlist')) {
        showError('That doesn\'t look like a Spotify playlist link.');
        return;
    }

    setLoading(analyzeBtn, true);

    try {
        const res = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url }),
        });

        const data = await res.json();

        if (data.error) {
            showError(data.error);
            return;
        }

        // Store tracks and render
        tracks = data.data || [];
        playlistTitle.textContent = data.playlist_name || 'Untitled Playlist';
        trackCountLabel.textContent = `${data.track_count} tracks found`;

        renderTrackList(tracks);
        showStep(stepTracks);

    } catch (err) {
        showError('Failed to connect to backend. Is it running on port 8000?');
        console.error(err);
    } finally {
        setLoading(analyzeBtn, false);
    }
});

// ─── Render Track List ───

function renderTrackList(tracks) {
    trackList.innerHTML = '';

    tracks.forEach((track, i) => {
        const pos = track.position || i + 1;
        const item = document.createElement('label');
        item.className = 'track-item';
        item.innerHTML = `
      <input type="checkbox" class="track-checkbox" data-position="${pos}" checked />
      <span class="track-number">${pos}</span>
      <div class="track-info">
        <div class="track-name">${escapeHtml(track.name)}</div>
        <div class="track-artist">${escapeHtml(track.artist)}</div>
      </div>
      <span class="track-popularity">${track.popularity ?? '—'}</span>
    `;

        // Toggle excluded visual state
        const checkbox = item.querySelector('.track-checkbox');
        checkbox.addEventListener('change', () => {
            item.classList.toggle('excluded', !checkbox.checked);
            updateSelectionCount();
        });

        trackList.appendChild(item);
    });

    updateSelectionCount();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ─── Select / Deselect All ───

selectAllBtn.addEventListener('click', () => {
    document.querySelectorAll('.track-checkbox').forEach(cb => {
        cb.checked = true;
        cb.closest('.track-item').classList.remove('excluded');
    });
    updateSelectionCount();
});

deselectAllBtn.addEventListener('click', () => {
    document.querySelectorAll('.track-checkbox').forEach(cb => {
        cb.checked = false;
        cb.closest('.track-item').classList.add('excluded');
    });
    updateSelectionCount();
});

// ─── Back Button ───

backBtn.addEventListener('click', () => {
    showStep(stepInput);
});

// ─── Step 2: Exclude & Build DNA ───

buildDnaBtn.addEventListener('click', async () => {
    setLoading(buildDnaBtn, true);

    try {
        // Collect positions of UNCHECKED tracks (the ones to exclude)
        const excludePositions = [];
        document.querySelectorAll('.track-checkbox').forEach(cb => {
            if (!cb.checked) {
                excludePositions.push(parseInt(cb.dataset.position));
            }
        });

        // Step A: Exclude tracks (if any unchecked)
        if (excludePositions.length > 0) {
            const excludeRes = await fetch(`${API_BASE}/exclude`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ positions: excludePositions }),
            });

            const excludeData = await excludeRes.json();
            if (excludeData.error || excludeData.detail) {
                showError(excludeData.error || excludeData.detail);
                return;
            }
        }

        // Step B: Build DNA
        const dnaRes = await fetch(`${API_BASE}/build-dna`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });

        const dnaData = await dnaRes.json();

        if (dnaData.error || dnaData.detail) {
            showError(dnaData.error || dnaData.detail);
            return;
        }

        // Render DNA results
        renderDnaResults(dnaData);
        showStep(stepResults);

    } catch (err) {
        showError('Failed during DNA analysis. Check the backend.');
        console.error(err);
    } finally {
        setLoading(buildDnaBtn, false);
    }
});

// ─── Natural ranges for each audio feature (for proper bar scaling) ───
const FEATURE_RANGES = {
    danceability: { min: 0, max: 1 },
    energy: { min: 0, max: 1 },
    loudness: { min: -60, max: 0 },
    speechiness: { min: 0, max: 1 },
    acousticness: { min: 0, max: 1 },
    instrumentalness: { min: 0, max: 1 },
    liveness: { min: 0, max: 1 },
    valence: { min: 0, max: 1 },
    tempo: { min: 0, max: 250 },
};

// ─── Render DNA Results ───

function renderDnaResults(data) {
    dnaPlaylistName.textContent = `${data.playlist_name} · ${data.track_count} tracks`;
    dnaResults.innerHTML = '';

    const profile = data.feature_profile || {};
    const features = Object.keys(profile);

    // ── 1. Cohesion Score ──
    if (data.cohesion_score != null) {
        const score = data.cohesion_score;
        const hue = score >= 70 ? 160 : score >= 40 ? 45 : 10; // green / yellow / red
        const cohesionCard = document.createElement('div');
        cohesionCard.className = 'dna-card full-width';
        cohesionCard.innerHTML = `
          <div class="dna-label">Playlist Cohesion</div>
          <div style="display: flex; align-items: center; gap: 20px;">
            <div class="dna-value" style="font-size: 2.2rem; color: hsl(${hue}, 70%, 55%);">${score}</div>
            <div>
              <div style="font-size: 0.85rem; color: var(--text-secondary);">
                ${score >= 70 ? 'Tight cluster — very cohesive sound' : score >= 40 ? 'Moderate variety in sound' : 'Wide range — eclectic playlist'}
              </div>
              <div class="dna-bar-track" style="margin-top: 8px; height: 10px;">
                <div class="dna-bar-fill" style="width: 0%; background: hsl(${hue}, 70%, 55%);" data-width="${score}%"></div>
              </div>
            </div>
          </div>
        `;
        dnaResults.appendChild(cohesionCard);
        requestAnimationFrame(() => {
            cohesionCard.querySelector('.dna-bar-fill').style.width = `${score}%`;
        });
    }

    // ── 2. Audio Feature Profile (per-feature scaling) ──
    if (features.length > 0) {
        const featCard = document.createElement('div');
        featCard.className = 'dna-card full-width';

        featCard.innerHTML = `
          <div class="dna-label">Audio Feature Profile</div>
          <div class="dna-bar-container">
            ${features.map(feat => {
            const fp = profile[feat];
            const mean = fp.mean;
            const std = fp.std;
            const range = FEATURE_RANGES[feat] || { min: 0, max: 1 };
            // Scale mean to 0-100% within its natural range
            const pct = Math.min(100, Math.max(0, ((mean - range.min) / (range.max - range.min)) * 100));
            // Format display value
            const displayVal = feat === 'tempo' ? mean.toFixed(1) + ' bpm'
                : feat === 'loudness' ? mean.toFixed(1) + ' dB'
                    : mean.toFixed(3);
            const stdDisplay = feat === 'tempo' ? `±${std.toFixed(1)}`
                : feat === 'loudness' ? `±${std.toFixed(1)}`
                    : `±${std.toFixed(3)}`;
            return `
                  <div class="dna-bar-row">
                    <span class="dna-bar-label">${feat}</span>
                    <div class="dna-bar-track">
                      <div class="dna-bar-fill" style="width: 0%;" data-width="${pct.toFixed(1)}%"></div>
                    </div>
                    <span class="dna-bar-value">${displayVal} <span style="color: var(--text-muted); font-size: 0.65rem;">${stdDisplay}</span></span>
                  </div>
                `;
        }).join('')}
          </div>
        `;

        dnaResults.appendChild(featCard);

        requestAnimationFrame(() => {
            featCard.querySelectorAll('.dna-bar-fill').forEach(bar => {
                bar.style.width = bar.dataset.width;
            });
        });
    }

    // ── 3. Dominant Traits ──
    const traits = data.dominant_traits || [];
    if (traits.length > 0) {
        const traitCard = document.createElement('div');
        traitCard.className = 'dna-card full-width';
        traitCard.innerHTML = `
          <div class="dna-label">Standout Traits</div>
          <div class="dna-pills" style="gap: 8px;">
            ${traits.map(t => {
            const arrow = t.direction === 'high' ? '↑' : '↓';
            const color = t.direction === 'high' ? 'var(--accent-2)' : 'var(--red)';
            return `<span class="dna-pill" style="border-color: ${color}; color: ${color};">
                  ${arrow} ${t.feature} ${Math.abs(t.deviation_pct).toFixed(0)}%
                </span>`;
        }).join('')}
          </div>
        `;
        dnaResults.appendChild(traitCard);
    }

    // ── 4. Key & Mode Distribution ──
    const keyDist = data.key_distribution || {};
    const modeDist = data.mode_distribution || {};

    if (Object.keys(keyDist).length > 0) {
        const keyCard = document.createElement('div');
        keyCard.className = 'dna-card';
        keyCard.innerHTML = `
          <div class="dna-label">Key Distribution</div>
          <div class="dna-pills">
            ${Object.entries(keyDist)
                .sort((a, b) => b[1] - a[1])
                .map(([key, count]) => `<span class="dna-pill">${KEY_NAMES[key] || key}: ${count}</span>`)
                .join('')}
          </div>
        `;
        dnaResults.appendChild(keyCard);
    }

    if (Object.keys(modeDist).length > 0) {
        const modeCard = document.createElement('div');
        modeCard.className = 'dna-card';
        modeCard.innerHTML = `
          <div class="dna-label">Mode Distribution</div>
          <div class="dna-pills">
            ${Object.entries(modeDist)
                .sort((a, b) => b[1] - a[1])
                .map(([mode, count]) => `<span class="dna-pill">${MODE_NAMES[mode] || mode}: ${count}</span>`)
                .join('')}
          </div>
        `;
        dnaResults.appendChild(modeCard);
    }

    // ── 5. Metadata ──
    const metaCard = document.createElement('div');
    metaCard.className = 'dna-card full-width';
    metaCard.innerHTML = `
      <div class="dna-label">Analysis Metadata</div>
      <div style="display: flex; gap: 32px; flex-wrap: wrap;">
        <div>
          <div class="dna-value">${data.track_count}</div>
          <div class="dna-label" style="margin-top: 4px; margin-bottom: 0;">Tracks Analyzed</div>
        </div>
        <div>
          <div class="dna-value">${(data.features_used || []).length}D</div>
          <div class="dna-label" style="margin-top: 4px; margin-bottom: 0;">Feature Space</div>
        </div>
        <div>
          <div class="dna-value">${data.cohesion_score ?? '—'}</div>
          <div class="dna-label" style="margin-top: 4px; margin-bottom: 0;">Cohesion Score</div>
        </div>
      </div>
    `;
    dnaResults.appendChild(metaCard);
}

// ─── Restart ───

restartBtn.addEventListener('click', () => {
    playlistUrlInput.value = '';
    tracks = [];
    trackList.innerHTML = '';
    dnaResults.innerHTML = '';
    showStep(stepInput);
});
