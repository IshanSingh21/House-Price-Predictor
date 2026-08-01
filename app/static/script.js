/**
 * House Price Predictor — Frontend Logic
 * Handles form interaction, AJAX predictions, and animated price display.
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- State ---
    const state = {
        sqft: 2000,
        bedrooms: 3,
        bathrooms: 2,
        age: 15,
        lot_size: 0.5,
        garage: 2,
        neighborhood: 6,
        has_pool: 0,
        distance_city: 10,
        condition: 7,
    };

    // --- Range Slider Setup ---
    const rangeConfigs = {
        sqft:          { suffix: ' sq ft',  decimals: 0, comma: true },
        age:           { suffix: ' years',  decimals: 0, comma: false },
        lot_size:      { suffix: ' acres',  decimals: 2, comma: false },
        neighborhood:  { suffix: ' / 10',   decimals: 0, comma: false },
        distance_city: { suffix: ' miles',  decimals: 1, comma: false },
        condition:     { suffix: ' / 10',   decimals: 0, comma: false },
    };

    Object.entries(rangeConfigs).forEach(([id, config]) => {
        const slider = document.getElementById(id);
        const display = document.getElementById(`${id}-value`);
        if (!slider || !display) return;

        const updateDisplay = () => {
            const val = parseFloat(slider.value);
            state[id] = val;
            let formatted = config.decimals > 0
                ? val.toFixed(config.decimals)
                : config.comma
                    ? val.toLocaleString()
                    : val.toString();
            display.textContent = formatted + config.suffix;
        };

        slider.addEventListener('input', updateDisplay);
        updateDisplay(); // Set initial
    });

    // --- Pill Selectors ---
    document.querySelectorAll('.select-pills').forEach(pillGroup => {
        const name = pillGroup.dataset.name;
        pillGroup.querySelectorAll('.pill').forEach(pill => {
            pill.addEventListener('click', () => {
                // Deactivate siblings
                pillGroup.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
                pill.classList.add('active');
                state[name] = parseFloat(pill.dataset.value);
            });
        });
    });

    // --- Form Submit ---
    const form = document.getElementById('prediction-form');
    const predictBtn = document.getElementById('predict-btn');
    const btnText = predictBtn.querySelector('.btn-text');
    const btnLoader = predictBtn.querySelector('.btn-loader');
    const resultPlaceholder = document.getElementById('result-placeholder');
    const resultContent = document.getElementById('result-content');
    const priceAmount = document.getElementById('price-amount');
    const importanceBars = document.getElementById('importance-bars');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Show loading
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline';
        predictBtn.disabled = true;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(state),
            });

            const data = await response.json();

            if (data.error) {
                alert('Prediction Error: ' + data.error);
                return;
            }

            // Show result
            resultPlaceholder.style.display = 'none';
            resultContent.style.display = 'block';
            resultContent.classList.remove('visible');
            void resultContent.offsetWidth; // trigger reflow
            resultContent.classList.add('visible');

            // Animate price counter
            animatePrice(data.predicted_price);

            // Render feature importance
            renderImportance(data.feature_importance);

        } catch (err) {
            alert('Failed to connect to the server. Make sure it is running.');
            console.error(err);
        } finally {
            btnText.style.display = 'inline';
            btnLoader.style.display = 'none';
            predictBtn.disabled = false;
        }
    });

    // --- Animated Price Counter ---
    function animatePrice(targetPrice) {
        const duration = 1200;
        const startTime = performance.now();
        const startPrice = parseFloat(priceAmount.textContent.replace(/,/g, '')) || 0;

        function step(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const ease = 1 - Math.pow(1 - progress, 3);
            const currentPrice = startPrice + (targetPrice - startPrice) * ease;

            priceAmount.textContent = Math.round(currentPrice).toLocaleString();

            if (progress < 1) {
                requestAnimationFrame(step);
            }
        }

        requestAnimationFrame(step);
    }

    // --- Feature Importance Bars ---
    const featureLabels = {
        sqft: 'Square Footage',
        bedrooms: 'Bedrooms',
        bathrooms: 'Bathrooms',
        age: 'House Age',
        lot_size: 'Lot Size',
        garage: 'Garage',
        neighborhood: 'Neighborhood',
        has_pool: 'Pool',
        distance_city: 'City Distance',
        condition: 'Condition',
    };

    function renderImportance(importanceData) {
        if (!importanceData || Object.keys(importanceData).length === 0) {
            importanceBars.innerHTML = '<p style="color: var(--text-muted); font-size: 0.82rem;">Feature importance not available for this model type.</p>';
            return;
        }

        // Sort by importance
        const sorted = Object.entries(importanceData).sort((a, b) => b[1] - a[1]);
        const maxVal = sorted[0][1];

        importanceBars.innerHTML = sorted.map(([feature, value]) => {
            const widthPercent = (value / maxVal) * 100;
            const label = featureLabels[feature] || feature;
            return `
                <div class="importance-bar-row">
                    <span class="importance-label">${label}</span>
                    <div class="importance-bar-container">
                        <div class="importance-bar-fill" data-width="${widthPercent}"></div>
                    </div>
                    <span class="importance-bar-value">${(value * 100).toFixed(1)}%</span>
                </div>
            `;
        }).join('');

        // Animate bars after a short delay
        requestAnimationFrame(() => {
            setTimeout(() => {
                document.querySelectorAll('.importance-bar-fill').forEach(bar => {
                    bar.style.width = bar.dataset.width + '%';
                });
            }, 100);
        });
    }
});
