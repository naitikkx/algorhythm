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

const buildSemanticBtn = document.getElementById('build-semantic-btn');
const semanticActions = document.getElementById('semantic-actions');
const semanticResults = document.getElementById('semantic-results');

const scoreForm = document.getElementById('score-form');
const scoreUrlInput = document.getElementById('score-url');
const scoreBtn = document.getElementById('score-btn');
const scoreResults = document.getElementById('score-results');

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
  semanticResults.innerHTML = '';
  semanticActions.hidden = false;
  scoreUrlInput.value = '';
  scoreResults.innerHTML = '';
  showStep(stepInput);
});

// ─── Semantic DNA & Track Scoring ───

buildSemanticBtn.addEventListener('click', async () => {
  setLoading(buildSemanticBtn, true);

  // Show progress UI
  semanticResults.innerHTML = `
    <div class="dna-card full-width" id="semantic-progress-card">
      <div class="dna-label">🧠 Analyzing Lyrics...</div>
      <div style="margin-top: 8px;">
        <div class="dna-bar-track" style="height: 12px;">
          <div class="dna-bar-fill" id="semantic-progress-bar" style="width: 0%; background: var(--accent); transition: width 0.3s;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 0.8rem; color: var(--text-muted);">
          <span id="semantic-progress-text">Starting...</span>
          <span id="semantic-progress-pct">0%</span>
        </div>
        <div id="semantic-progress-track" style="margin-top: 6px; font-size: 0.75rem; color: var(--text-secondary); min-height: 1.2em;"></div>
      </div>
    </div>
  `;

  // Start polling progress
  const progressInterval = setInterval(async () => {
    try {
      const pRes = await fetch(`${API_BASE}/semantic-progress`);
      const p = await pRes.json();
      const bar = document.getElementById('semantic-progress-bar');
      const text = document.getElementById('semantic-progress-text');
      const pct = document.getElementById('semantic-progress-pct');
      const track = document.getElementById('semantic-progress-track');

      if (bar) {
        if (p.status === 'fetching_lyrics' && p.total > 0) {
          const percent = Math.round((p.current / p.total) * 100);
          bar.style.width = `${percent}%`;
          pct.textContent = `${percent}%`;
          text.textContent = `Fetching lyrics: ${p.current}/${p.total} (✅ ${p.found} found, ❌ ${p.missing} missing)`;
        } else if (p.status === 'analyzing' && p.gemini_total > 0) {
          const percent = Math.round((p.gemini_batch / p.gemini_total) * 100);
          bar.style.width = `${percent}%`;
          bar.style.background = 'var(--accent-secondary, #a855f7)';
          pct.textContent = `${percent}%`;
          text.textContent = `🧠 Gemini analyzing: batch ${p.gemini_batch}/${p.gemini_total}`;
        } else if (p.status === 'analyzing') {
          bar.style.width = '100%';
          pct.textContent = '100%';
          text.textContent = '🧠 Running Gemini sentiment analysis...';
        }
        if (track) track.textContent = p.current_track || '';
      }
    } catch (e) { /* ignore polling errors */ }
  }, 800);

  try {
    const res = await fetch(`${API_BASE}/build-semantic-dna`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    const data = await res.json();
    clearInterval(progressInterval);

    if (data.error || data.detail) {
      showError(data.error || data.detail);
      semanticResults.innerHTML = '';
      return;
    }
    renderSemanticResults(data);
    semanticActions.hidden = true;
  } catch (err) {
    clearInterval(progressInterval);
    showError('Failed during Semantic Analysis. Check backend logs.');
    console.error(err);
    semanticResults.innerHTML = '';
  } finally {
    setLoading(buildSemanticBtn, false);
  }
});

function renderSemanticResults(data) {
  semanticResults.innerHTML = '';

  if (data.semantic_cohesion != null) {
    const score = data.semantic_cohesion;
    const hue = score >= 70 ? 160 : score >= 40 ? 45 : 10;
    const cohesionCard = document.createElement('div');
    cohesionCard.className = 'dna-card full-width';
    cohesionCard.innerHTML = `
          <div class="dna-label">Semantic Cohesion (Thematic Unity)</div>
          <div style="display: flex; align-items: center; gap: 20px;">
            <div class="dna-value" style="font-size: 2.2rem; color: hsl(${hue}, 70%, 55%);">${score}</div>
            <div style="flex-grow: 1;">
               <div style="font-size: 0.85rem; color: var(--text-secondary);">
                ${score >= 70 ? 'Strong thematic unity' : score >= 40 ? 'Mixed lyrical themes' : 'Eclectic lyrical content'}
               </div>
               <div class="dna-bar-track" style="margin-top: 8px; height: 10px;">
                 <div class="dna-bar-fill" style="width: 0%; background: hsl(${hue}, 70%, 55%);" data-width="${score}%"></div>
               </div>
            </div>
            <div style="font-size: 0.85rem; color: var(--text-muted); text-align: right;">
               Based on ${data.tracks_analyzed} / ${data.lyrics_coverage.total_tracks} tracks with lyrics
            </div>
          </div>
        `;
    semanticResults.appendChild(cohesionCard);
    requestAnimationFrame(() => {
      cohesionCard.querySelector('.dna-bar-fill').style.width = `${score}%`;
    });
  }

  const agg = data.aggregate || {};
  const themes = agg.dominant_themes || [];
  const moods = agg.dominant_moods || [];

  if (themes.length > 0 || moods.length > 0) {
    const tCard = document.createElement('div');
    tCard.className = 'dna-card full-width';
    tCard.innerHTML = `
          <div style="display: flex; gap: 2rem; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 200px;">
              <div class="dna-label">Dominant Themes</div>
              <div class="dna-pills" style="gap: 8px;">
                ${themes.map(t => `<span class="dna-pill">${t[0]} (${(t[1] * 100).toFixed(0)}%)</span>`).join('')}
              </div>
            </div>
            <div style="flex: 1; min-width: 200px;">
              <div class="dna-label">Dominant Moods</div>
              <div class="dna-pills" style="gap: 8px;">
                ${moods.map(m => `<span class="dna-pill">${m[0]} (${(m[1] * 100).toFixed(0)}%)</span>`).join('')}
              </div>
            </div>
          </div>
        `;
    semanticResults.appendChild(tCard);
  }

  // ── Missing Lyrics ──
  const missing = data.missing_lyrics || [];
  if (missing.length > 0) {
    const missCard = document.createElement('div');
    missCard.className = 'dna-card full-width';
    missCard.innerHTML = `
      <div class="dna-label" style="cursor: pointer; display: flex; align-items: center; gap: 8px;" onclick="this.parentElement.querySelector('.missing-list').hidden = !this.parentElement.querySelector('.missing-list').hidden; this.querySelector('.chevron').textContent = this.parentElement.querySelector('.missing-list').hidden ? '▸' : '▾';">
        <span class="chevron">▸</span> Missing Lyrics (${missing.length} tracks)
      </div>
      <div class="missing-list" hidden style="margin-top: 8px; max-height: 200px; overflow-y: auto;">
        ${missing.map(t => `<div style="padding: 4px 0; font-size: 0.8rem; color: var(--text-muted); border-bottom: 1px solid rgba(255,255,255,0.05);">${escapeHtml(t)}</div>`).join('')}
      </div>
    `;
    semanticResults.appendChild(missCard);
  }

  // Show cached badge if applicable
  if (data.cached) {
    const badge = document.createElement('div');
    badge.style.cssText = 'text-align: center; font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem;';
    badge.textContent = '⚡ Loaded from cache';
    semanticResults.appendChild(badge);
  }
}

scoreForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  hideError();
  const url = scoreUrlInput.value.trim();
  if (!url || !url.includes('spotify.com/track')) {
    showError('Please paste a valid Spotify track URL.');
    return;
  }
  setLoading(scoreBtn, true);
  scoreResults.innerHTML = '';

  try {
    const audioRes = fetch(`${API_BASE}/score`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    }).then(r => r.json()).catch(() => null);

    const semanticRes = fetch(`${API_BASE}/semantic-score`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    }).then(r => r.json()).catch(() => null);

    const [audioData, semanticData] = await Promise.all([audioRes, semanticRes]);

    const audioOk = audioData && !audioData.error && !audioData.detail;
    const semanticOk = semanticData && !semanticData.error && !semanticData.detail;

    if (!audioOk && !semanticOk) {
      const audioErr = audioData?.error || audioData?.detail || '';
      const semErr = semanticData?.error || semanticData?.detail || '';
      showError(audioErr || semErr || 'Could not score this track.');
      return;
    }

    const trackName = audioOk ? audioData.track : (semanticData?.track || 'Unknown');
    const artistName = audioOk ? audioData.artist : (semanticData?.artist || 'Unknown');

    let html = `
      <div class="dna-card full-width" style="border-left: 4px solid var(--accent); padding-left: 1rem; background: var(--surface);">
        <h4 style="margin: 0; color: var(--text-primary); font-size: 1.1rem;">${escapeHtml(trackName)} <span style="color: var(--text-secondary); font-weight: 400;">by ${escapeHtml(artistName)}</span></h4>
        <div style="margin-top: 1.5rem; display: flex; gap: 3rem; flex-wrap: wrap;">
    `;

    if (audioOk) {
      html += `
          <div>
            <div class="dna-label">Audio Fit (${(audioData.scores.composite * 100).toFixed(0)}%)</div>
            <div class="dna-value" style="font-size: 1.5rem; color: ${audioData.verdict.includes('ADD') ? 'var(--accent)' : 'var(--text-primary)'}">${audioData.verdict}</div>
          </div>
      `;
    } else {
      const audioMsg = audioData?.error || 'Audio features not available';
      html += `
          <div>
            <div class="dna-label">Audio Fit</div>
            <div style="color: var(--text-muted); font-size: 0.85rem; margin-top: 4px;">⚠️ ${escapeHtml(audioMsg)}</div>
          </div>
      `;
    }

    if (semanticOk) {
      const semScore = semanticData.semantic_score || 0;
      const verdict = semScore >= 70 ? '✅ HIGH MATCH' : semScore >= 40 ? '🟡 MED MATCH' : '❌ LOW MATCH';
      const trackThemes = (semanticData.track_sentiment?.themes || []).join(', ');
      html += `
          <div>
            <div class="dna-label">Semantic Fit (${semScore}%)</div>
            <div class="dna-value" style="font-size: 1.5rem; color: ${semScore >= 70 ? 'var(--accent)' : 'var(--text-primary)'}">${verdict}</div>
            <div style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 4px;">Themes: ${trackThemes || 'None'}</div>
          </div>
      `;
    } else if (semanticData && (semanticData.error || semanticData.detail)) {
      const msg = semanticData.error || semanticData.detail;
      let displayMsg = msg;
      if (msg.includes('No Semantic DNA built yet')) {
        displayMsg = 'Semantic DNA not built yet';
      } else if (msg.includes('No lyrics found')) {
        displayMsg = 'Instrumental / No Lyrics found to score';
      }
      html += `
          <div>
            <div class="dna-label">Semantic Fit</div>
            <div style="color: var(--text-muted); font-size: 0.85rem; margin-top: 4px;">${displayMsg}</div>
          </div>
      `;
    }

    html += `</div></div>`;
    scoreResults.innerHTML = html;
    scoreUrlInput.value = '';

  } catch (err) {
    showError('Failed to score track. Check logs.');
    console.error(err);
  } finally {
    setLoading(scoreBtn, false);
  }
});
