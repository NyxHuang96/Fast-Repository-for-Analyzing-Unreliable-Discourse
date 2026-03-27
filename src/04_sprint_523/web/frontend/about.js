document.addEventListener('DOMContentLoaded', () => {

    // =========================================================================
    // Data
    // =========================================================================
    const MEMBERS = {
        'tianhao-cao':  { name: 'Tianhao Cao',  role: 'Front-end Developer', img: './img/TianhaoCao.png'  },
        'darwin-zhang': { name: 'Darwin Zhang', role: 'Back-end Developer',   img: './img/DarwinZhang.png' },
        'marco-wang':   { name: 'Marco Wang',   role: 'Front-end Developer',  img: './img/MarcoWang.png'   },
        'yusen-huang':  { name: 'Yusen Huang',  role: 'Back-end Developer',   img: './img/YusenHuang.png'  },
    };

    // The one correct arrangement
    const CORRECT = {
        'top-left':     'tianhao-cao',
        'top-right':    'darwin-zhang',
        'bottom-left':  'marco-wang',
        'bottom-right': 'yusen-huang',
    };

    // =========================================================================
    // State
    // =========================================================================

    // slot key → memberId | null
    const slotState = { 'top-left': null, 'top-right': null, 'bottom-left': null, 'bottom-right': null };

    // members still in the bank (as ordered array for stable rendering)
    let bankOrder = shuffled(Object.keys(MEMBERS));

    let draggingId   = null; // which member is currently being dragged
    let draggingFrom = null; // 'bank' | slot key

    // =========================================================================
    // DOM refs
    // =========================================================================
    const bank     = document.getElementById('puzzle-bank');
    const grid     = document.getElementById('puzzle-grid');
    const cells    = document.querySelectorAll('.puzzle-cell');
    const reveal   = document.getElementById('puzzle-reveal');
    const statusEl = document.getElementById('puzzle-status');

    // =========================================================================
    // Helpers
    // =========================================================================
    function shuffled(arr) {
        const a = [...arr];
        for (let i = a.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [a[i], a[j]] = [a[j], a[i]];
        }
        return a;
    }

    function correctCount() {
        return Object.entries(slotState).filter(([slot, id]) => id && CORRECT[slot] === id).length;
    }

    // =========================================================================
    // Card factory
    // =========================================================================
    function makeCard(memberId) {
        const m = MEMBERS[memberId];
        const card = document.createElement('div');
        card.className = 'puzzle-card';
        card.draggable = true;
        card.dataset.member = memberId;
        card.innerHTML = `
            <img src="${m.img}" alt="${m.name}">
            <h4>${m.name}</h4>
            <p class="role">${m.role}</p>
        `;
        card.addEventListener('dragstart', onCardDragStart);
        card.addEventListener('dragend',   onCardDragEnd);
        return card;
    }

    // =========================================================================
    // Render
    // =========================================================================
    function render() {
        renderBank();
        renderCells();
        renderStatus();
    }

    function renderBank() {
        bank.innerHTML = '';
        if (bankOrder.length === 0) {
            bank.innerHTML = '<span class="puzzle-bank-empty">All members placed — check the grid!</span>';
            return;
        }
        bankOrder.forEach(id => bank.appendChild(makeCard(id)));
    }

    function renderCells() {
        cells.forEach(cell => {
            const slot     = cell.dataset.slot;
            const memberId = slotState[slot];
            const isRight  = Boolean(memberId && CORRECT[slot] === memberId);
            const isWrong  = Boolean(memberId && !isRight);

            cell.classList.toggle('correct', isRight);
            cell.classList.toggle('wrong',   isWrong);

            // Preserve the position label element across re-renders
            const posLabel = cell.querySelector('.cell-pos-label');
            cell.innerHTML = '';
            if (posLabel) cell.appendChild(posLabel);

            if (memberId) {
                cell.appendChild(makeCard(memberId));
                if (isRight) {
                    const badge = document.createElement('span');
                    badge.className = 'cell-correct-badge';
                    badge.textContent = '✓';
                    cell.appendChild(badge);
                }
            } else {
                const hint = document.createElement('div');
                hint.className = 'cell-empty-hint';
                cell.appendChild(hint);
            }
        });
    }

    function renderStatus() {
        const n = correctCount();
        statusEl.textContent = `${n} / 4 correct`;

        if (n === 4) {
            // Brief pause so the last card's correct-state renders first
            setTimeout(() => {
                grid.style.display = 'none';
                reveal.style.display = 'block';
            }, 350);
        }
    }

    // =========================================================================
    // Drag handlers — cards
    // =========================================================================
    function onCardDragStart(e) {
        draggingId   = e.currentTarget.dataset.member;
        const parentCell = e.currentTarget.closest('.puzzle-cell');
        draggingFrom = parentCell ? parentCell.dataset.slot : 'bank';
        e.currentTarget.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
    }

    function onCardDragEnd(e) {
        e.currentTarget.classList.remove('dragging');
        // Clean up all drag-over highlights
        document.querySelectorAll('.drag-over').forEach(el => el.classList.remove('drag-over'));
    }

    // =========================================================================
    // Drop zone — bank
    // =========================================================================
    let bankDepth = 0; // prevent flicker on child-element enter/leave

    bank.addEventListener('dragenter', (e) => {
        e.preventDefault();
        if (++bankDepth === 1) bank.classList.add('drag-over');
    });
    bank.addEventListener('dragleave', () => {
        if (--bankDepth === 0) bank.classList.remove('drag-over');
    });
    bank.addEventListener('dragover', (e) => { e.preventDefault(); });
    bank.addEventListener('drop', (e) => {
        e.preventDefault();
        bankDepth = 0;
        bank.classList.remove('drag-over');
        if (!draggingId || draggingFrom === 'bank') return; // nothing to do

        // Return card from a grid cell back to the bank
        slotState[draggingFrom] = null;
        bankOrder.push(draggingId);

        draggingId = draggingFrom = null;
        render();
    });

    // =========================================================================
    // Drop zone — cells
    // =========================================================================
    const cellDepth = {};

    cells.forEach(cell => {
        const slot = cell.dataset.slot;
        cellDepth[slot] = 0;

        cell.addEventListener('dragenter', (e) => {
            e.preventDefault();
            if (++cellDepth[slot] === 1) cell.classList.add('drag-over');
        });
        cell.addEventListener('dragleave', () => {
            if (--cellDepth[slot] === 0) cell.classList.remove('drag-over');
        });
        cell.addEventListener('dragover', (e) => { e.preventDefault(); });

        cell.addEventListener('drop', (e) => {
            e.preventDefault();
            cellDepth[slot] = 0;
            cell.classList.remove('drag-over');
            if (!draggingId || draggingFrom === slot) return; // dropped on itself

            const evicted = slotState[slot]; // whoever currently lives here (may be null)

            if (draggingFrom === 'bank') {
                // Remove from bank
                bankOrder = bankOrder.filter(id => id !== draggingId);
                // If target was occupied, send its occupant to the bank
                if (evicted) bankOrder.push(evicted);
            } else {
                // Cell → cell swap: send evicted (or null) to source slot
                slotState[draggingFrom] = evicted;
            }

            slotState[slot] = draggingId;
            draggingId = draggingFrom = null;
            render();
        });
    });

    // =========================================================================
    // Reset
    // =========================================================================
    function reset() {
        Object.keys(slotState).forEach(k => { slotState[k] = null; });
        bankOrder = shuffled(Object.keys(MEMBERS));
        reveal.style.display = 'none';
        grid.style.display   = '';
        render();
    }

    document.getElementById('reset-btn').addEventListener('click', reset);
    document.getElementById('reset-btn-reveal').addEventListener('click', reset);

    // =========================================================================
    // Boot
    // =========================================================================
    render();
});
