document.addEventListener('DOMContentLoaded', () => {
    // 1. Identify all the elements from functions.html
    const searchBtn = document.getElementById('search-btn');
    const queryInput = document.getElementById('search-query');
    const labelInput = document.getElementById('search-label');
    const annotatedInput = document.getElementById('search-annotated');
    const resultsContainer = document.getElementById('search-results');
    const statsContainer = document.getElementById('search-stats');

    // If we are on a page that doesn't have the search button, stop running the script
    if (!searchBtn) return; 

    const API_BASE_URL = 'http://127.0.0.1:8000';

    // 2. Attach Event Listeners
    searchBtn.addEventListener('click', performSearch);
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performSearch();
    });

    // 3. The Search Function
    async function performSearch() {
        const q = queryInput.value.trim();
        if (!q) {
            alert("Please enter a search term.");
            return;
        }

        // Build the URL to talk to the Python backend
        const url = new URL(`${API_BASE_URL}/search`);
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

            // Fetch the data from FastAPI
            const response = await fetch(url);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            
            const data = await response.json();
            renderResults(data);
            
        } catch (error) {
            console.error("Search failed:", error);
            statsContainer.innerHTML = `<span style="color:red;">Error connecting to backend. Is the Python server running?</span>`;
        }
    }

    // 4. Render the Results
    function renderResults(data) {
        statsContainer.innerHTML = `Found <strong>${data.total_hits}</strong> results in ${data.search_time_ms}ms`;

        if (data.results.length === 0) {
            resultsContainer.innerHTML = '<p>No matching documents found.</p>';
            return;
        }

        // Create HTML for each search result
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