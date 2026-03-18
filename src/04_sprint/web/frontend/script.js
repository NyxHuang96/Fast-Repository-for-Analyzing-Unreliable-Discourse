// === Handler for functions.html ===
document.addEventListener('DOMContentLoaded', () => {
    const searchBtn = document.getElementById('search-btn');
    if (!searchBtn) return;

    const queryInput = document.getElementById('search-query');
    const labelInput = document.getElementById('search-label');
    const annotatedInput = document.getElementById('search-annotated');
    const resultsContainer = document.getElementById('search-results');
    const statsContainer = document.getElementById('search-stats');

    searchBtn.addEventListener('click', performSearch);
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performSearch();
    });

    async function performSearch() {
        const q = queryInput.value.trim();
        if (!q) {
            alert("Please enter a search term.");
            return;
        }

        const url = new URL(`${window.location.origin}/search`);
        url.searchParams.append('q', q);

        if (labelInput && labelInput.value) {
            url.searchParams.append('label', labelInput.value);
        }
        if (annotatedInput && annotatedInput.checked) {
            url.searchParams.append('annotated_only', 'true');
        }

        try {
            statsContainer.innerHTML = 'Searching...';
            resultsContainer.innerHTML = '';

            const response = await fetch(url);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

            const data = await response.json();
            renderResults(data);
        } catch (error) {
            console.error("Search failed:", error);
            statsContainer.innerHTML = `<span style="color:red;">Error connecting to backend. Is the Python server running?</span>`;
        }
    }

    function renderResults(data) {
        statsContainer.innerHTML = `Found <strong>${data.total_hits}</strong> results in ${data.search_time_ms}ms`;

        if (data.results.length === 0) {
            resultsContainer.innerHTML = '<p>No matching documents found.</p>';
            return;
        }

        const html = data.results.map(res => `
            <div style="border: 1px solid #444; padding: 15px; margin-bottom: 10px; border-radius: 8px; background: #1e1e24; color: #fff;">
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #333; padding-bottom: 10px; margin-bottom: 10px;">
                    <strong>ID: ${res.doc_id}</strong>
                    <span style="background: #6c63ff; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; color: white;">
                        ${res.label.toUpperCase()} ${res.is_annotated ? '✓ Annotated' : ''}
                    </span>
                </div>
                <p style="margin-bottom: 8px;"><strong>EN:</strong> ${res.snippet || '<em>No English text</em>'}</p>
                <p style="margin: 0; color: #aaa;"><strong>ZH:</strong> ${res.snippet_zh || '<em>No Chinese text</em>'}</p>
            </div>
        `).join('');

        resultsContainer.innerHTML = html;
    }
});

// === Handler for index.html ===
document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('search-form');
    if (!searchForm) return;

    const API = window.location.origin;
    const RECENT_KEY = 'fraud_corpus_recent_searches';
    const MAX_RECENT = 8;

    // --- Recent Searches helpers ---
    function getRecentSearches() {
        try { return JSON.parse(localStorage.getItem(RECENT_KEY) || '[]'); }
        catch { return []; }
    }

    function saveRecentSearch(query) {
        let recents = getRecentSearches().filter(q => q !== query);
        recents.unshift(query);
        if (recents.length > MAX_RECENT) recents = recents.slice(0, MAX_RECENT);
        localStorage.setItem(RECENT_KEY, JSON.stringify(recents));
    }

    function clearRecentSearches() {
        localStorage.removeItem(RECENT_KEY);
        renderDropdown();
    }

    // --- Dropdown UI ---
    const dropdownTrigger = document.querySelector('.right-links .dropdown');
    const dropdownPanel = document.createElement('div');
    dropdownPanel.className = 'recent-searches-panel';
    dropdownPanel.style.display = 'none';
    dropdownTrigger.style.position = 'relative';
    dropdownTrigger.appendChild(dropdownPanel);

    function renderDropdown() {
        const recents = getRecentSearches();
        if (recents.length === 0) {
            dropdownPanel.innerHTML = '<div class="recent-empty">No recent searches</div>';
        } else {
            dropdownPanel.innerHTML =
                recents.map(q => `<div class="recent-item" data-query="${q.replace(/"/g, '&quot;')}">${q}</div>`).join('') +
                '<div class="recent-clear">Clear all</div>';

            dropdownPanel.querySelectorAll('.recent-item').forEach(item => {
                item.addEventListener('click', (e) => {
                    e.stopPropagation();
                    document.getElementById('query-input').value = item.dataset.query;
                    closeDropdown();
                    searchForm.requestSubmit();
                });
            });

            dropdownPanel.querySelector('.recent-clear').addEventListener('click', (e) => {
                e.stopPropagation();
                clearRecentSearches();
            });
        }
    }

    function openDropdown() {
        renderDropdown();
        dropdownPanel.style.display = 'block';
    }

    function closeDropdown() {
        dropdownPanel.style.display = 'none';
    }

    dropdownTrigger.addEventListener('click', (e) => {
        e.stopPropagation();
        dropdownPanel.style.display === 'none' ? openDropdown() : closeDropdown();
    });

    document.addEventListener('click', closeDropdown);

    // --- Corpus / Label filter state ---
    let activeCorpus = 'all';   // 'all' | 'annotated' | 'raw'
    let activeLabel = '';       // '' | 'ham' | 'spam' | 'phish'
    let lastQuery = '';

    const annotatedCheckbox = document.getElementById('annotated-only');

    // Sidebar link elements
    const corpusLinks = {
        all:       document.querySelector('#corpus-stats-list li:nth-child(1) a'),
        annotated: document.querySelector('#corpus-stats-list li:nth-child(2) a'),
        raw:       document.querySelector('#corpus-stats-list li:nth-child(3) a'),
    };
    const labelLinks = {
        ham:   document.querySelector('#corpus-stats-list li a[title*="Ham"]'),
        spam:  document.querySelector('#corpus-stats-list li a[title*="Spam"]'),
        phish: document.querySelector('#corpus-stats-list li a[title*="Phishing"]'),
    };

    function setActiveCorpus(corpus) {
        activeCorpus = corpus;
        Object.entries(corpusLinks).forEach(([key, el]) => {
            if (!el) return;
            el.classList.toggle('active-corpus', key === corpus);
        });
        // Sync annotated-only checkbox
        if (annotatedCheckbox) annotatedCheckbox.checked = (corpus === 'annotated');
    }

    function setActiveLabel(label) {
        activeLabel = label;
        Object.entries(labelLinks).forEach(([key, el]) => {
            if (!el) return;
            el.classList.toggle('active-corpus', key === label);
        });
    }

    // Corpus sidebar clicks
    Object.entries(corpusLinks).forEach(([key, el]) => {
        if (!el) return;
        el.addEventListener('click', (e) => {
            e.preventDefault();
            setActiveCorpus(key);
            if (lastQuery) runSearch(lastQuery);
        });
    });

    // Label sidebar clicks
    Object.entries(labelLinks).forEach(([key, el]) => {
        if (!el) return;
        el.addEventListener('click', (e) => {
            e.preventDefault();
            // Toggle off if already active
            const next = activeLabel === key ? '' : key;
            setActiveLabel(next);
            if (lastQuery) runSearch(lastQuery);
        });
    });

    // Keep state in sync when user toggles the annotated checkbox
    if (annotatedCheckbox) {
        annotatedCheckbox.addEventListener('change', () => {
            if (annotatedCheckbox.checked) {
                setActiveCorpus('annotated');
            } else if (activeCorpus === 'annotated') {
                setActiveCorpus('all');
            }
        });
    }

    // --- Load sidebar counts from /stats ---
    async function loadStats() {
        try {
            const res = await fetch(`${API}/stats`);
            const data = await res.json();
            const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
            set('stat-total-docs',     data.total_docs);
            set('stat-annotated-docs', data.annotated_docs);
            set('stat-raw-docs',       data.raw_docs);
            set('stat-label-ham',      data.labels['ham']   ?? data.labels['Ham']   ?? 0);
            set('stat-label-spam',     data.labels['spam']  ?? data.labels['Spam']  ?? 0);
            set('stat-label-phish',    data.labels['phish'] ?? data.labels['Phish'] ?? data.labels['Phishing'] ?? 0);
        } catch { /* backend not running — leave counts as (...) */ }
    }
    loadStats();

    // --- Full-text modal ---
    const modal = document.getElementById('doc-modal');
    const modalTitle = document.getElementById('doc-modal-title');
    const modalBody = document.getElementById('doc-modal-body');
    const modalClose = document.getElementById('doc-modal-close');
    const modalBackdrop = modal.querySelector('.doc-modal-backdrop');

    function openModal(doc) {
        const lbl = doc.label.toLowerCase();
        modalTitle.innerHTML =
            `${doc.doc_id} <span class="badge badge-${lbl}">${doc.label.toUpperCase()}</span>` +
            (doc.is_annotated ? ' <span class="badge badge-annotated">✓ Annotated</span>' : '');
        modalBody.innerHTML =
            (doc.text ? `<div class="full-text-block"><div class="full-text-label">English</div><div class="full-text-content">${doc.text}</div></div>` : '') +
            (doc.text_zh ? `<div class="full-text-block"><div class="full-text-label">Chinese</div><div class="full-text-content">${doc.text_zh}</div></div>` : '');
        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
    }

    function closeModal() {
        modal.style.display = 'none';
        document.body.style.overflow = '';
    }

    modalClose.addEventListener('click', closeModal);
    modalBackdrop.addEventListener('click', closeModal);
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

    async function fetchAndOpenDoc(docId) {
        modalTitle.textContent = docId;
        modalBody.innerHTML = '<div class="loading">Loading...</div>';
        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
        try {
            const res = await fetch(`${API}/doc/${encodeURIComponent(docId)}`);
            if (!res.ok) throw new Error('Not found');
            const doc = await res.json();
            openModal(doc);
        } catch {
            modalBody.innerHTML = '<p style="color:var(--red);">Failed to load document.</p>';
        }
    }

    // --- Core search function ---
    async function runSearch(q) {
        const url = new URL(`${API}/search`);
        url.searchParams.append('q', q);
        if (activeCorpus === 'annotated') url.searchParams.append('annotated_only', 'true');
        if (activeCorpus === 'raw')       url.searchParams.append('raw_only', 'true');
        if (activeLabel)                  url.searchParams.append('label', activeLabel);

        const area = document.getElementById('results-area');
        area.innerHTML = '<div class="loading">Searching...</div>';

        try {
            const res = await fetch(url);
            const data = await res.json();

            area.innerHTML = `<p>Found <strong>${data.total_hits}</strong> results in ${data.search_time_ms}ms</p>`;

            data.results.forEach(r => {
                const item = document.createElement('div');
                item.className = 'result-item';
                item.title = 'Click to view full text';
                const lbl = r.label.toLowerCase();
                item.dataset.label = lbl;
                item.innerHTML = `
                    <div class="doc-id">
                        ${r.doc_id}
                        <span class="badge badge-${lbl}">${r.label.toUpperCase()}</span>
                        ${r.is_annotated ? '<span class="badge badge-annotated">✓ Annotated</span>' : ''}
                    </div>
                    <div class="snippet"><strong>EN:</strong> ${r.snippet || '<em>None</em>'}</div>
                    <div class="snippet" style="margin-top:0.4rem;"><strong>ZH:</strong> ${r.snippet_zh || '<em>None</em>'}</div>
                `;
                item.addEventListener('click', () => fetchAndOpenDoc(r.doc_id));
                area.appendChild(item);
            });
        } catch (err) {
            area.innerHTML = '<p style="color:red;">Search failed. Is the backend running?</p>';
        }
    }

    // --- Search form submit ---
    searchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const q = document.getElementById('query-input').value.trim();
        if (!q) return;
        lastQuery = q;
        saveRecentSearch(q);
        if (annotatedCheckbox && annotatedCheckbox.checked && activeCorpus !== 'annotated') setActiveCorpus('annotated');
        runSearch(q);
    });

    // Initialise active states
    setActiveCorpus('all');
});
