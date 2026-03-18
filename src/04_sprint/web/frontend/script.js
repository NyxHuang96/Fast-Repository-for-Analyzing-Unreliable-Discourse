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

    searchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const q = document.getElementById('query-input').value.trim();
        if (!q) return;

        const url = new URL(`${API}/search`);
        url.searchParams.append('q', q);

        const label = document.getElementById('label-filter');
        if (label && label.value) url.searchParams.append('label', label.value);

        const ann = document.getElementById('annotated-only');
        if (ann && ann.checked) url.searchParams.append('annotated_only', 'true');

        const area = document.getElementById('results-area');
        area.innerHTML = 'Searching...';

        try {
            const res = await fetch(url);
            const data = await res.json();
            area.innerHTML = `<p>Found <strong>${data.total_hits}</strong> results in ${data.search_time_ms}ms</p>` +
                data.results.map(r => `
                    <div style="border:1px solid #444; padding:15px; margin-bottom:10px; border-radius:8px; background:#1e1e24;">
                        <strong>${r.doc_id}</strong> — ${r.label.toUpperCase()}
                        <p><strong>EN:</strong> ${r.snippet || '<em>None</em>'}</p>
                        <p style="color:#aaa;"><strong>ZH:</strong> ${r.snippet_zh || '<em>None</em>'}</p>
                    </div>
                `).join('');
        } catch (err) {
            area.innerHTML = '<p style="color:red;">Search failed. Is the backend running?</p>';
        }
    });
});
