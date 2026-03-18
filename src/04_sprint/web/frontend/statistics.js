document.addEventListener('DOMContentLoaded', () => {
    const API_BASE_URL = '';

    // =========================================================================
    // Color palette for charts
    // =========================================================================
    const COLORS = [
        '#6c63ff', '#34d399', '#fbbf24', '#60a5fa',
        '#f87171', '#a78bfa', '#fb923c', '#2dd4bf',
        '#e879f9', '#38bdf8',
    ];
    const COLORS_ALPHA = COLORS.map(c => c + '33');

    // =========================================================================
    // State
    // =========================================================================
    let currentCorpus = 'all';
    let labelChartInstance = null;
    let statusChartInstance = null;

    // =========================================================================
    // Corpus filter buttons
    // =========================================================================
    const filterBar = document.getElementById('corpus-filter');
    filterBar.addEventListener('click', (e) => {
        const btn = e.target.closest('.filter-btn');
        if (!btn) return;

        filterBar.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        currentCorpus = btn.dataset.corpus;
        fetchStats(currentCorpus);
    });

    // =========================================================================
    // Chart type selectors
    // =========================================================================
    document.getElementById('label-chart-type').addEventListener('change', () => rebuildCharts());
    document.getElementById('status-chart-type').addEventListener('change', () => rebuildCharts());

    // =========================================================================
    // Fetch stats from backend
    // =========================================================================
    let latestData = null;

    async function fetchStats(corpus = 'all') {
        try {
            const url = `${window.location.origin}/stats?corpus=${encodeURIComponent(corpus)}`;
            const res = await fetch(url);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            latestData = await res.json();
            renderCards(latestData);
            rebuildCharts();
            renderTable(latestData);
        } catch (err) {
            console.error('Failed to fetch stats:', err);
            // Show fallback demo data so the page isn't empty
            latestData = {
                total_docs: 0,
                annotated_docs: 0,
                raw_docs: 0,
                labels: {}
            };
            renderCards(latestData);
            rebuildCharts();
            renderTable(latestData);
        }
    }

    // =========================================================================
    // Render summary cards
    // =========================================================================
    function renderCards(data) {
        document.getElementById('total-docs').textContent = data.total_docs.toLocaleString();
        document.getElementById('annotated-docs').textContent = data.annotated_docs.toLocaleString();
        document.getElementById('raw-docs').textContent = data.raw_docs.toLocaleString();
        document.getElementById('label-count').textContent = Object.keys(data.labels).length;
    }

    // =========================================================================
    // Build / rebuild charts
    // =========================================================================
    function rebuildCharts() {
        if (!latestData) return;

        const labelType = document.getElementById('label-chart-type').value;
        const statusType = document.getElementById('status-chart-type').value;

        // --- Label distribution chart ---
        if (labelChartInstance) labelChartInstance.destroy();

        const labelNames = Object.keys(latestData.labels);
        const labelValues = Object.values(latestData.labels);

        const labelCtx = document.getElementById('label-chart').getContext('2d');
        labelChartInstance = new Chart(labelCtx, {
            type: labelType,
            data: {
                labels: labelNames.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
                datasets: [{
                    label: 'Documents',
                    data: labelValues,
                    backgroundColor: COLORS_ALPHA.slice(0, labelNames.length),
                    borderColor: COLORS.slice(0, labelNames.length),
                    borderWidth: 1.5,
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: labelType === 'bar' ? 'top' : 'right',
                        labels: { color: '#9aa0b0', font: { family: 'Inter', size: 12 } }
                    }
                },
                scales: labelType === 'bar' ? {
                    x: { ticks: { color: '#9aa0b0' }, grid: { color: 'rgba(255,255,255,0.04)' } },
                    y: { ticks: { color: '#9aa0b0' }, grid: { color: 'rgba(255,255,255,0.04)' } }
                } : undefined
            }
        });

        // --- Status chart ---
        if (statusChartInstance) statusChartInstance.destroy();

        const statusCtx = document.getElementById('status-chart').getContext('2d');
        statusChartInstance = new Chart(statusCtx, {
            type: statusType,
            data: {
                labels: ['Annotated', 'Raw'],
                datasets: [{
                    label: 'Documents',
                    data: [latestData.annotated_docs, latestData.raw_docs],
                    backgroundColor: [COLORS_ALPHA[1], COLORS_ALPHA[2]],
                    borderColor: [COLORS[1], COLORS[2]],
                    borderWidth: 1.5,
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: statusType === 'bar' ? 'top' : 'right',
                        labels: { color: '#9aa0b0', font: { family: 'Inter', size: 12 } }
                    }
                },
                scales: statusType === 'bar' ? {
                    x: { ticks: { color: '#9aa0b0' }, grid: { color: 'rgba(255,255,255,0.04)' } },
                    y: { ticks: { color: '#9aa0b0' }, grid: { color: 'rgba(255,255,255,0.04)' } }
                } : undefined
            }
        });
    }

    // =========================================================================
    // Render label breakdown table
    // =========================================================================
    function renderTable(data) {
        const tbody = document.getElementById('label-tbody');
        const total = data.total_docs || 1;
        const entries = Object.entries(data.labels).sort((a, b) => b[1] - a[1]);

        if (entries.length === 0) {
            tbody.innerHTML = `<tr><td colspan="4" style="padding:1.5rem; text-align:center; color:var(--text-muted);">No data available. Make sure the backend is running.</td></tr>`;
            return;
        }

        const maxVal = Math.max(...entries.map(e => e[1]));

        tbody.innerHTML = entries.map(([label, count], i) => {
            const pct = ((count / total) * 100).toFixed(1);
            const barWidth = ((count / maxVal) * 100).toFixed(1);
            const color = COLORS[i % COLORS.length];
            return `
                <tr style="border-bottom:1px solid var(--border-glass);">
                    <td style="padding:0.6rem 1rem; font-size:0.9rem; font-weight:500;">${label.charAt(0).toUpperCase() + label.slice(1)}</td>
                    <td style="padding:0.6rem 1rem; font-size:0.9rem; color:var(--text-secondary);">${count.toLocaleString()}</td>
                    <td style="padding:0.6rem 1rem; font-size:0.9rem; color:var(--text-secondary);">${pct}%</td>
                    <td style="padding:0.6rem 1rem; width:40%;">
                        <div style="background:rgba(255,255,255,0.04); border-radius:4px; height:8px; overflow:hidden;">
                            <div style="width:${barWidth}%; height:100%; background:${color}; border-radius:4px; transition:width 0.4s ease;"></div>
                        </div>
                    </td>
                </tr>`;
        }).join('');
    }

    // =========================================================================
    // Initial load
    // =========================================================================
    fetchStats('all');
});
