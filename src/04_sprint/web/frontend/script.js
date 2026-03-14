document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('search-form');
    const queryInput = document.getElementById('query-input');
    const annotatedCheckbox = document.getElementById('annotated-only');
    const resultsArea = document.getElementById('results-area');

    // Use a local FastAPI server for development
    const API_BASE_URL = 'http://127.0.0.1:8000';

    searchForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const query = queryInput.value.trim();
        if (!query) return;

        const isAnnotatedOnly = annotatedCheckbox.checked;

        // Show loading state
        resultsArea.innerHTML = '<div class="loading">Searching corpus...</div>';

        try {
            const response = await fetch(`${API_BASE_URL}/search?q=${encodeURIComponent(query)}&annotated_only=${isAnnotatedOnly}`);

            if (!response.ok) {
                throw new Error(`Server returned ${response.status}`);
            }

            const data = await response.json();
            renderResults(data, query);

        } catch (error) {
            console.error('Search error:', error);
            resultsArea.innerHTML = `<div class="placeholder-text" style="color: #dc3545;">
                <p>Error connecting to search server.</p>
                <p style="font-size: 0.85rem; margin-top: 0.5rem">Make sure the FastAPI backend is running on ${API_BASE_URL}</p>
            </div>`;
        }
    });

    function renderResults(data, query) {
        if (!data.results || data.results.length === 0) {
            resultsArea.innerHTML = `<div class="placeholder-text"><p>No relevant texts found for "${query}".</p></div>`;
            return;
        }

        const html = [
            `<div style="margin-bottom: 1.5rem; font-size: 0.9rem; color: #6c757d;">
                Found ${data.total_hits} results in ${data.search_time_ms}ms
            </div>`
        ];

        data.results.forEach(item => {
            // Whoosh now returns raw HTML syntax directly in snippet text with `<span class="highlight">...</span>`

            const badgeHtml = item.is_annotated
                ? '<span class="badge badge-annotated">✓ Annotated</span>'
                : '<span class="badge">Raw Text</span>';

            html.push(`
                <div class="result-item">
                    <div class="doc-id">Document ID: ${item.doc_id} ${badgeHtml}</div>
                    <div class="snippet">...${item.snippet}...</div>
                </div>
            `);
        });

        resultsArea.innerHTML = html.join('');
    }

    // Note: placeholder link interception removed — nav links now point to real pages.
});
