// Load and display the ML results
async function loadResults() {
    try {
        // Look for results.json in the same directory
        const response = await fetch('results.json?t=' + Date.now());
        if (!response.ok) {
            throw new Error('Failed to load results. Please run the analysis first.');
        }
        const data = await response.json();
        
        // Update last updated time
        document.getElementById('last-updated').innerHTML = 
            `Last updated: ${new Date(data.generated_date).toLocaleString()}`;
        
        // Display project info
        displayProjectInfo(data.project_info);
        
        // Display performance metrics
        displayPerformance(data.performance, data.model_info);
        
        // Create charts
        createClassDistributionChart(data.project_info.class_counts);
        createFeatureImportanceChart(data.feature_importance);
        
        // Display classification report
        displayClassificationReport(data.performance);
        
        // Display sample predictions
        displaySamplePredictions(data.sample_predictions);
        
    } catch (error) {
        console.error('Error loading results:', error);
        const statsContainer = document.getElementById('stats-container');
        if (statsContainer) {
            statsContainer.innerHTML = 
                '<div class="stat-card error">❌ Error loading data. Please run the analysis first.<br><small>' + error.message + '</small></div>';
        }
    }
}

function displayProjectInfo(info) {
    const container = document.getElementById('stats-container');
    if (!container) return;
    
    container.innerHTML = '';
    
    const statsData = [
        { label: 'Total Objects', value: info.dataset_size, icon: '🌌' },
        { label: 'Features Used', value: info.features.length, icon: '��' },
        { label: 'Classes', value: info.classes.length, icon: '🎯' }
    ];
    
    // Add class counts if available
    if (info.class_counts) {
        if (info.class_counts.STAR) {
            statsData.push({ label: 'Stars', value: info.class_counts.STAR, icon: '⭐' });
        }
        if (info.class_counts.GALAXY) {
            statsData.push({ label: 'Galaxies', value: info.class_counts.GALAXY, icon: '🌠' });
        }
        if (info.class_counts.QSO) {
            statsData.push({ label: 'Quasars', value: info.class_counts.QSO, icon: '💫' });
        }
    }
    
    statsData.forEach(stat => {
        const card = document.createElement('div');
        card.className = 'stat-card glass-effect';
        card.innerHTML = `
            <h3>${stat.icon} ${stat.label}</h3>
            <div class="stat-value">${stat.value.toLocaleString()}</div>
        `;
        container.appendChild(card);
    });
}

function displayPerformance(performance, modelInfo) {
    const container = document.getElementById('performance-stats');
    if (!container) return;
    
    container.innerHTML = `
        <div class="performance-grid">
            <div class="metric-card">
                <h4>🎯 Algorithm</h4>
                <div class="metric-value">Random Forest</div>
                <div class="metric-label">n_estimators: ${modelInfo?.n_estimators || 100}</div>
            </div>
            <div class="metric-card">
                <h4>📈 Test Accuracy</h4>
                <div class="metric-value">${(performance.testing_accuracy * 100).toFixed(2)}%</div>
            </div>
            <div class="metric-card">
                <h4>🎲 Train Accuracy</h4>
                <div class="metric-value">${(performance.training_accuracy * 100).toFixed(2)}%</div>
            </div>
        </div>
    `;
}

function displayClassificationReport(performance) {
    const container = document.getElementById('classification-report');
    if (!container) return;
    
    // If we have detailed classification report, use it
    if (performance.classification_report) {
        const report = performance.classification_report;
        let html = `
            <table class="report-table">
                <thead>
                    <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Support</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        Object.entries(report).forEach(([key, value]) => {
            if (key !== 'accuracy' && key !== 'macro avg' && key !== 'weighted avg' && typeof value === 'object') {
                html += `
                    <tr>
                        <td class="highlight">${key}</td>
                        <td>${(value.precision * 100).toFixed(2)}%</td>
                        <td>${(value.recall * 100).toFixed(2)}%</td>
                        <td>${(value['f1-score'] * 100).toFixed(2)}%</td>
                        <td>${value.support}</td>
                    </tr>
                `;
            }
        });
        
        html += `
                </tbody>
            </table>
            <p style="text-align: center; margin-top: 20px; color: #8899aa;">
                Overall Accuracy: ${(report.accuracy * 100).toFixed(2)}%
            </p>
        `;
        
        container.innerHTML = html;
    } else {
        // Simple report if detailed not available
        container.innerHTML = `
            <table class="report-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="highlight">Test Accuracy</td>
                        <td>${(performance.testing_accuracy * 100).toFixed(2)}%</td>
                    </tr>
                    <tr>
                        <td class="highlight">Training Accuracy</td>
                        <td>${(performance.training_accuracy * 100).toFixed(2)}%</td>
                    </tr>
                </tbody>
            </table>
        `;
    }
}

function createClassDistributionChart(classCounts) {
    const ctx = document.getElementById('object-chart')?.getContext('2d');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(classCounts),
            datasets: [{
                data: Object.values(classCounts),
                backgroundColor: [
                    '#FFD700', // Gold for STAR
                    '#4169E1', // Royal Blue for GALAXY
                    '#9400D3'  // Dark Violet for QSO
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: '#e0e0e0',
                        font: { size: 12 }
                    }
                },
                title: {
                    display: true,
                    text: 'Class Distribution in Dataset',
                    color: '#e0e0e0',
                    font: { size: 14 }
                }
            }
        }
    });
}

function createFeatureImportanceChart(importance) {
    const ctx = document.getElementById('constellation-chart')?.getContext('2d');
    if (!ctx) return;
    
    // Sort features by importance
    const sorted = Object.entries(importance)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8); // Top 8 features
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sorted.map(item => item[0]),
            datasets: [{
                label: 'Feature Importance',
                data: sorted.map(item => item[1]),
                backgroundColor: '#4ECDC4',
                borderRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                title: {
                    display: true,
                    text: 'Top 8 Feature Importances',
                    color: '#e0e0e0',
                    font: { size: 14 }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    ticks: { color: '#e0e0e0' }
                },
                x: {
                    grid: { display: false },
                    ticks: { 
                        color: '#e0e0e0',
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });
}

function displaySamplePredictions(samples) {
    const container = document.getElementById('sample-predictions');
    if (!container) return;
    
    if (samples && samples.length > 0) {
        let html = '<div class="predictions-grid">';
        samples.forEach((sample, index) => {
            const isCorrect = sample.actual === sample.predicted;
            html += `
                <div class="prediction-card ${isCorrect ? 'correct' : 'incorrect'}">
                    <div class="prediction-header">
                        <span class="sample-id">Sample #${index + 1}</span>
                        <span class="confidence">${(sample.confidence * 100).toFixed(1)}% confident</span>
                    </div>
                    <div class="prediction-result">
                        <span class="actual">${sample.actual}</span>
                        <span class="arrow">→</span>
                        <span class="predicted ${isCorrect ? 'correct' : 'incorrect'}">${sample.predicted}</span>
                    </div>
                    ${sample.redshift ? `<div class="prediction-features">redshift: ${sample.redshift.toFixed(4)}</div>` : ''}
                </div>
            `;
        });
        html += '</div>';
        container.innerHTML = html;
    } else {
        container.innerHTML = '<p class="loading">No sample predictions available</p>';
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', loadResults);

// Auto-refresh every 60 seconds
setInterval(loadResults, 60000);
