// Global Chart Configuration
if (typeof Chart !== 'undefined') {
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';
    Chart.defaults.font.family = "'Inter', sans-serif";
}

const chartColors = {
    purple: '#7c3aed',
    cyan: '#06b6d4',
    green: '#10b981',
    orange: '#f59e0b',
    red: '#ef4444'
};

function createLineChart(canvasId, title, datasets) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top' },
                title: { display: true, text: title, color: '#e2e8f0' }
            },
            scales: {
                x: { grid: { display: false } },
                y: { grid: { color: 'rgba(255, 255, 255, 0.05)' } }
            },
            animation: { duration: 400 }
        }
    });
}

window.initTrainingCharts = function() {
    const lossChart = createLineChart('lossChart', 'Training Losses', [
        { label: 'Box Loss', borderColor: chartColors.purple, data: [], tension: 0.4 },
        { label: 'Cls Loss', borderColor: chartColors.cyan, data: [], tension: 0.4 },
        { label: 'DFL Loss', borderColor: chartColors.green, data: [], tension: 0.4 }
    ]);

    const accChart = createLineChart('accChart', 'Accuracy (mAP)', [
        { label: 'mAP50', borderColor: chartColors.orange, data: [], tension: 0.4 },
        { label: 'mAP50-95', borderColor: chartColors.purple, data: [], tension: 0.4 }
    ]);

    let epochCounter = 0;
    
    document.addEventListener('TrainingMetricsReceived', (e) => {
        const metrics = e.detail;
        epochCounter++;
        
        if (lossChart) {
            lossChart.data.labels.push(epochCounter);
            lossChart.data.datasets[0].data.push(metrics.box_loss);
            lossChart.data.datasets[1].data.push(metrics.cls_loss);
            lossChart.data.datasets[2].data.push(metrics.dfl_loss);
            if (lossChart.data.labels.length > 50) {
                lossChart.data.labels.shift();
                lossChart.data.datasets.forEach(d => d.data.shift());
            }
            lossChart.update();
        }

        if (accChart) {
            accChart.data.labels.push(epochCounter);
            accChart.data.datasets[0].data.push(metrics.map50);
            accChart.data.datasets[1].data.push(metrics.map5095);
            if (accChart.data.labels.length > 50) {
                accChart.data.labels.shift();
                accChart.data.datasets.forEach(d => d.data.shift());
            }
            accChart.update();
        }
        
        // Update metric cards if they exist
        const map50Card = document.getElementById('metric-map50');
        if (map50Card) map50Card.innerText = metrics.map50.toFixed(3);
    });
};
