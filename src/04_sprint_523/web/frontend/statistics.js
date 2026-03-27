document.addEventListener('DOMContentLoaded', () => {
    const API_BASE_URL = '';

    // =========================================================================
    // Color system — matches the cyber threat-intel theme
    // Known labels get fixed, meaningful colors; extras cycle through fallbacks
    // =========================================================================
    const LABEL_PALETTE = {
        ham:      { solid: '#10b981', alpha: 'rgba(16,  185, 129, 0.18)' }, // green  — safe
        spam:     { solid: '#f59e0b', alpha: 'rgba(245, 158,  11, 0.18)' }, // amber  — warning
        phish:    { solid: '#f43f5e', alpha: 'rgba(244,  63,  94, 0.18)' }, // red    — danger
        phishing: { solid: '#f43f5e', alpha: 'rgba(244,  63,  94, 0.18)' },
    };

    const FALLBACK_PALETTE = [
        { solid: '#00d4ff', alpha: 'rgba(  0, 212, 255, 0.18)' }, // cyan
        { solid: '#6366f1', alpha: 'rgba( 99, 102, 241, 0.18)' }, // indigo
        { solid: '#8b5cf6', alpha: 'rgba(139,  92, 246, 0.18)' }, // violet
        { solid: '#14b8a6', alpha: 'rgba( 20, 184, 166, 0.18)' }, // teal
        { solid: '#ec4899', alpha: 'rgba(236,  72, 153, 0.18)' }, // pink
    ];

    // Status chart — annotated vs raw
    const STATUS_COLORS = {
        annotated: { solid: '#00d4ff', alpha: 'rgba(  0, 212, 255, 0.18)' },
        raw:       { solid: '#4b5875', alpha: 'rgba( 75,  88, 117, 0.35)' },
    };

    function colorFor(label, fallbackIndex) {
        return LABEL_PALETTE[label.toLowerCase()] ?? FALLBACK_PALETTE[fallbackIndex % FALLBACK_PALETTE.length];
    }

    // =========================================================================
    // Chart.js global defaults
    // =========================================================================
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";
    Chart.defaults.font.size = 12;
    Chart.defaults.animation.duration = 700;
    Chart.defaults.animation.easing = 'easeInOutQuart';

    // Shared tooltip style
    const TOOLTIP_DEFAULTS = {
        backgroundColor: '#0b1225',
        titleColor: '#e2e8f0',
        bodyColor: '#94a3b8',
        borderColor: 'rgba(0, 212, 255, 0.2)',
        borderWidth: 1,
        padding: 10,
        cornerRadius: 8,
        displayColors: true,
        boxWidth: 10,
        boxHeight: 10,
    };

    // Shared grid style for bar/line charts
    const GRID_DEFAULTS = {
        color: 'rgba(0, 212, 255, 0.05)',
        drawBorder: false,
    };

    const TICK_DEFAULTS = { color: '#64748b', padding: 6 };

    // =========================================================================
    // Helpers
    // =========================================================================
    function hexToRgba(hex, alpha) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(2)})`;
    }

    // =========================================================================
    // State
    // =========================================================================
    let currentCorpus = 'all';
    let labelChartInstance = null;
    let statusChartInstance = null;
    let wordFreqChartInstance = null;
    let latestData = null;

    // Word-freq local controls
    let wfLang  = 'en';
    let wfTop   = 20;
    let wfLabel = '';

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

    // Chart-type selectors
    document.getElementById('label-chart-type').addEventListener('change', () => rebuildCharts());
    document.getElementById('status-chart-type').addEventListener('change', () => rebuildCharts());

    // Word-freq control selectors (only rebuild word-freq chart, not all charts)
    document.getElementById('wf-lang').addEventListener('change', e => {
        wfLang = e.target.value;
        fetchWordFreq();
    });
    document.getElementById('wf-top').addEventListener('change', e => {
        wfTop = parseInt(e.target.value);
        fetchWordFreq();
    });
    document.getElementById('wf-label').addEventListener('change', e => {
        wfLabel = e.target.value;
        fetchWordFreq();
    });

    // =========================================================================
    // Fetch stats from backend
    // =========================================================================
    async function fetchStats(corpus = 'all') {
        try {
            const url = `${window.location.origin}/stats?corpus=${encodeURIComponent(corpus)}`;
            const res = await fetch(url);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            latestData = await res.json();
        } catch (err) {
            console.warn('Stats fetch failed — showing empty state:', err);
            latestData = { total_docs: 0, annotated_docs: 0, raw_docs: 0, labels: {} };
        }
        renderCards(latestData);
        rebuildCharts();
        renderTable(latestData);
        fetchWordFreq(); // keep word-freq in sync with corpus filter
    }

    // =========================================================================
    // Summary cards
    // =========================================================================
    function renderCards(data) {
        document.getElementById('total-docs').textContent     = data.total_docs.toLocaleString();
        document.getElementById('annotated-docs').textContent = data.annotated_docs.toLocaleString();
        document.getElementById('raw-docs').textContent       = data.raw_docs.toLocaleString();
        document.getElementById('label-count').textContent    = Object.keys(data.labels).length;
    }

    // =========================================================================
    // Chart builders
    // =========================================================================
    function buildLabelChart(ctx, type, labels, values, colors) {
        const isCartesian = type === 'bar';

        const dataset = {
            label: 'Documents',
            data: values,
            backgroundColor: colors.map(c => c.alpha),
            borderColor:     colors.map(c => c.solid),
            borderWidth: 1.5,
            hoverBackgroundColor: colors.map(c => c.solid + '33'),
            hoverBorderColor:     colors.map(c => c.solid),
            hoverBorderWidth: 2,
        };

        if (isCartesian) {
            dataset.borderRadius  = 6;
            dataset.borderSkipped = false;
        }

        if (type === 'doughnut' || type === 'pie') {
            dataset.hoverOffset = 8;
        }

        return new Chart(ctx, {
            type,
            data: {
                labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
                datasets: [dataset],
            },
            options: {
                responsive: true,
                cutout: type === 'doughnut' ? '62%' : undefined,
                plugins: {
                    tooltip: {
                        ...TOOLTIP_DEFAULTS,
                        callbacks: {
                            label: ctx => ` ${ctx.label}: ${Number(ctx.raw).toLocaleString()} docs`,
                        },
                    },
                    legend: {
                        position: isCartesian ? 'top' : 'right',
                        labels: {
                            color: '#94a3b8',
                            padding: 16,
                            usePointStyle: true,
                            pointStyle: 'circle',
                            font: { size: 12 },
                        },
                    },
                },
                scales: isCartesian ? {
                    x: { grid: GRID_DEFAULTS, ticks: TICK_DEFAULTS, border: { display: false } },
                    y: { grid: GRID_DEFAULTS, ticks: { ...TICK_DEFAULTS, callback: v => v.toLocaleString() }, border: { display: false } },
                } : undefined,
            },
        });
    }

    function buildStatusChart(ctx, type, annotated, raw) {
        const isCartesian = type === 'bar';
        const aColor = STATUS_COLORS.annotated;
        const rColor = STATUS_COLORS.raw;

        const dataset = {
            label: 'Documents',
            data: [annotated, raw],
            backgroundColor: [aColor.alpha, rColor.alpha],
            borderColor:     [aColor.solid, rColor.solid],
            borderWidth: 1.5,
            hoverOffset: 8,
        };

        if (isCartesian) {
            dataset.borderRadius  = 6;
            dataset.borderSkipped = false;
        }

        return new Chart(ctx, {
            type,
            data: {
                labels: ['Annotated', 'Raw'],
                datasets: [dataset],
            },
            options: {
                responsive: true,
                cutout: type === 'doughnut' ? '62%' : undefined,
                plugins: {
                    tooltip: {
                        ...TOOLTIP_DEFAULTS,
                        callbacks: {
                            label: ctx => ` ${ctx.label}: ${Number(ctx.raw).toLocaleString()} docs`,
                        },
                    },
                    legend: {
                        position: isCartesian ? 'top' : 'right',
                        labels: {
                            color: '#94a3b8',
                            padding: 16,
                            usePointStyle: true,
                            pointStyle: 'circle',
                            font: { size: 12 },
                        },
                    },
                },
                scales: isCartesian ? {
                    x: { grid: GRID_DEFAULTS, ticks: TICK_DEFAULTS, border: { display: false } },
                    y: { grid: GRID_DEFAULTS, ticks: { ...TICK_DEFAULTS, callback: v => v.toLocaleString() }, border: { display: false } },
                } : undefined,
            },
        });
    }

    // =========================================================================
    // Rebuild both charts from current latestData + selected types
    // =========================================================================
    function rebuildCharts() {
        if (!latestData) return;

        const labelType  = document.getElementById('label-chart-type').value;
        const statusType = document.getElementById('status-chart-type').value;

        // --- Label distribution ---
        if (labelChartInstance) { labelChartInstance.destroy(); labelChartInstance = null; }

        const labelNames  = Object.keys(latestData.labels);
        const labelValues = Object.values(latestData.labels);
        const labelColors = labelNames.map((name, i) => colorFor(name, i));

        labelChartInstance = buildLabelChart(
            document.getElementById('label-chart').getContext('2d'),
            labelType, labelNames, labelValues, labelColors
        );

        // --- Annotation status ---
        if (statusChartInstance) { statusChartInstance.destroy(); statusChartInstance = null; }

        statusChartInstance = buildStatusChart(
            document.getElementById('status-chart').getContext('2d'),
            statusType, latestData.annotated_docs, latestData.raw_docs
        );
    }

    // =========================================================================
    // Label breakdown table
    // =========================================================================
    function renderTable(data) {
        const tbody = document.getElementById('label-tbody');
        const total   = data.total_docs || 1;
        const entries = Object.entries(data.labels).sort((a, b) => b[1] - a[1]);

        if (entries.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="4" style="padding:2rem; text-align:center; color:var(--text-muted); font-size:0.87rem;">
                        No data available — make sure the backend is running.
                    </td>
                </tr>`;
            return;
        }

        const maxVal = Math.max(...entries.map(e => e[1]));

        tbody.innerHTML = entries.map(([label, count], i) => {
            const pct      = ((count / total) * 100).toFixed(1);
            const barWidth = ((count / maxVal) * 100).toFixed(1);
            const { solid } = colorFor(label, i);
            const capsLabel = label.charAt(0).toUpperCase() + label.slice(1);

            return `
                <tr style="border-bottom:1px solid var(--border-subtle); transition:background 0.18s;"
                    onmouseover="this.style.background='rgba(0,212,255,0.03)'"
                    onmouseout="this.style.background=''">
                    <td style="padding:0.65rem 1rem; font-size:0.88rem; font-weight:600; white-space:nowrap;">
                        <span style="
                            display:inline-block;
                            width:8px; height:8px;
                            border-radius:50%;
                            background:${solid};
                            box-shadow:0 0 6px ${solid}88;
                            margin-right:0.6rem;
                            vertical-align:middle;
                        "></span>${capsLabel}
                    </td>
                    <td style="padding:0.65rem 1rem; font-size:0.88rem; color:var(--text-secondary);
                               font-variant-numeric:tabular-nums; font-family:'SF Mono','Fira Code',monospace;">
                        ${count.toLocaleString()}
                    </td>
                    <td style="padding:0.65rem 1rem; font-size:0.88rem; color:var(--text-muted);
                               font-variant-numeric:tabular-nums;">
                        ${pct}%
                    </td>
                    <td style="padding:0.65rem 1rem; width:40%;">
                        <div style="background:rgba(255,255,255,0.04); border-radius:4px; height:6px; overflow:hidden;">
                            <div style="
                                width:${barWidth}%;
                                height:100%;
                                border-radius:4px;
                                background:linear-gradient(90deg, ${solid}, ${solid}88);
                                box-shadow:0 0 8px ${solid}55;
                                transition:width 0.5s cubic-bezier(0.4,0,0.2,1);
                            "></div>
                        </div>
                    </td>
                </tr>`;
        }).join('');
    }

    // =========================================================================
    // Word Frequency — fetch & chart
    // =========================================================================
    async function fetchWordFreq() {
        const loadingEl = document.getElementById('wf-loading');
        const canvasEl  = document.getElementById('word-freq-chart');
        const scannedEl = document.getElementById('wf-docs-scanned');

        loadingEl.textContent = 'Loading word frequencies...';
        loadingEl.style.display = 'flex';
        canvasEl.style.display  = 'none';

        const url = new URL(`${window.location.origin}/word-freq`);
        url.searchParams.set('corpus', currentCorpus);
        url.searchParams.set('lang',   wfLang);
        url.searchParams.set('limit',  wfTop);
        if (wfLabel) url.searchParams.set('label', wfLabel);

        try {
            const res = await fetch(url);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();

            if (scannedEl) {
                scannedEl.textContent = `${data.docs_scanned.toLocaleString()} docs scanned`;
            }

            if (!data.words || data.words.length === 0) {
                loadingEl.textContent = 'No words found for the selected filters.';
                return;
            }

            loadingEl.style.display = 'none';
            canvasEl.style.display  = 'block';
            renderWordFreqChart(data.words);
        } catch (err) {
            console.warn('Word freq fetch failed:', err);
            loadingEl.textContent = 'Backend not running — word frequency unavailable.';
        }
    }

    function renderWordFreqChart(words) {
        if (wordFreqChartInstance) {
            wordFreqChartInstance.destroy();
            wordFreqChartInstance = null;
        }

        const labels = words.map(w => w.word);
        const counts = words.map(w => w.count);
        const n = labels.length;

        // Bar color: label-specific when filtered, accent cyan otherwise
        let baseColor = '#00d4ff';
        if (wfLabel) baseColor = colorFor(wfLabel, 0).solid;

        // Fade bars by rank so #1 word is most vivid
        const bgColors     = counts.map((_, i) => hexToRgba(baseColor, 0.72 - (i / n) * 0.44));
        const borderColors = counts.map((_, i) => hexToRgba(baseColor, 0.90 - (i / n) * 0.40));

        const ctx = document.getElementById('word-freq-chart').getContext('2d');
        wordFreqChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Occurrences',
                    data: counts,
                    backgroundColor: bgColors,
                    borderColor: borderColors,
                    borderWidth: 1,
                    borderRadius: 4,
                    borderSkipped: false,
                }],
            },
            options: {
                indexAxis: 'y',   // horizontal bars — better for word labels
                responsive: true,
                animation: { duration: 550, easing: 'easeOutQuart' },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        ...TOOLTIP_DEFAULTS,
                        callbacks: {
                            title: ctx => ctx[0].label,
                            label: ctx => `  ${ctx.raw.toLocaleString()} occurrences`,
                        },
                    },
                },
                scales: {
                    x: {
                        grid: GRID_DEFAULTS,
                        ticks: { ...TICK_DEFAULTS, callback: v => v.toLocaleString() },
                        border: { display: false },
                    },
                    y: {
                        grid: { display: false },
                        ticks: {
                            color: '#94a3b8',
                            font: {
                                family: "'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
                                size: 11,
                            },
                            padding: 8,
                        },
                        border: { display: false },
                    },
                },
            },
        });
    }

    // =========================================================================
    // Boot
    // =========================================================================
    fetchStats('all');
});
