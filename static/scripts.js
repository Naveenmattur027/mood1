const suggestions = {
    "Very Positive": [
        "Keep up the great work!",
        "You're doing amazing!",
        "Your positivity is inspiring.",
        "Stay awesome!",
        "You're on the right track.",
        "Your happiness is contagious.",
        "Keep shining!",
        "You're making a difference.",
        "Your enthusiasm is wonderful.",
        "Stay positive and keep going!",
        "You're a beacon of joy.",
        "Your optimism is admirable.",
        "Keep spreading happiness.",
        "You're a ray of sunshine.",
        "Your positive energy is infectious.",
        "Stay cheerful and bright.",
        "You're a joy to be around.",
        "Keep that smile going.",
        "Your positivity is a gift.",
        "Stay happy and blessed."
    ],
    "Positive": [
        "Good job!",
        "You're doing well.",
        "Keep it up!",
        "Your efforts are paying off.",
        "Stay positive.",
        "You're on the right path.",
        "Keep moving forward.",
        "Your attitude is great.",
        "Stay optimistic.",
        "You're making progress.",
        "Keep your spirits high.",
        "Your outlook is positive.",
        "Stay encouraged.",
        "You're doing better every day.",
        "Keep your head up.",
        "Your progress is noticeable.",
        "Stay hopeful.",
        "You're improving.",
        "Keep your positive momentum.",
        "Your future looks bright."
    ],
    "Neutral": [
        "Stay balanced.",
        "Keep a steady pace.",
        "Maintain your composure.",
        "Stay calm and collected.",
        "Keep your emotions in check.",
        "Stay level-headed.",
        "Maintain your equilibrium.",
        "Keep a neutral perspective.",
        "Stay centered.",
        "Your balance is commendable.",
        "Keep your cool.",
        "Stay steady.",
        "Maintain your neutrality.",
        "Keep your emotions stable.",
        "Stay even-keeled.",
        "Your composure is admirable.",
        "Keep a neutral outlook.",
        "Stay poised.",
        "Maintain your neutral stance.",
        "Keep your emotions neutral."
    ],
    "Negative": [
        "Stay strong.",
        "You can overcome this.",
        "Keep your head up.",
        "Better days are ahead.",
        "Stay resilient.",
        "You're stronger than you think.",
        "Keep pushing through.",
        "Your struggles are temporary.",
        "Stay hopeful.",
        "You can turn this around.",
        "Keep your spirits up.",
        "Your strength is admirable.",
        "Stay determined.",
        "You can rise above this.",
        "Keep your chin up.",
        "Your perseverance will pay off.",
        "Stay positive despite challenges.",
        "You can find a solution.",
        "Keep your faith strong.",
        "Your tough times will pass."
    ],
    "Very Negative": [
        "Seek support if needed.",
        "You're not alone in this.",
        "Stay strong through this tough time.",
        "Better days will come.",
        "Keep fighting.",
        "Your struggles are valid.",
        "Stay resilient.",
        "You can overcome this.",
        "Keep your hope alive.",
        "Your pain is temporary.",
        "Stay determined to improve.",
        "Your strength is admirable.",
        "Keep pushing through.",
        "You can rise above this.",
        "Stay hopeful despite challenges.",
        "Your perseverance will pay off.",
        "Stay positive despite hardships.",
        "You can find a way out.",
        "Keep your faith strong.",
        "Your tough times will pass."
    ]
  };
  
  // Add loading indicator when making requests
  function showLoading(targetElement = "#sentimentResult") {
    $(targetElement).html('<div class="text-center"><div class="spinner"></div><p class="text-muted">Analyzing your entry...</p></div>');
  }
  
  function addEntry() {
    const entry = $("#diaryEntry").val();
    if (entry.trim() === "") {
        showToast("Please enter some text!", "error");
        return;
    }
    
    showLoading();
    
    $.post("/add_entry", { entry: entry }, (response) => {
        $("#diaryEntry").val("");
        updateEntryHistory(response.entries);
        showToast("Entry added successfully!", "success");
        updateDashboard();
        $("#sentimentResult").html(`
            <div class="text-center py-4">
                <i class="fas fa-check-circle fa-2x text-success mb-3"></i>
                <p class="text-muted">Your entry has been saved</p>
                <p class="small text-muted">Click "Analyze" to get insights about your entry</p>
            </div>
        `);
    }).fail(() => {
        showToast("Error adding entry. Please try again.", "error");
        $("#sentimentResult").html(`
            <div class="text-center py-4">
                <i class="fas fa-exclamation-circle fa-2x text-danger mb-3"></i>
                <p class="text-muted">Error adding entry</p>
                <p class="small text-muted">Please try again</p>
            </div>
        `);
    });
  }
  
  function updateEntryHistory(entries) {
    const entryHistory = $("#entryHistory");
    
    if (!entries || entries.length === 0) {
        entryHistory.html(`
            <div class="text-center py-5">
                <i class="fas fa-book fa-2x text-muted mb-3"></i>
                <p class="text-muted">No entries yet. Start writing your first diary entry!</p>
            </div>
        `);
        return;
    }
    
    entryHistory.empty();
  
    entries.forEach((e) => {
        const entryItem = $(
            `<div class="entry-item fade-in">
                <div class="entry-date"><i class="fas fa-calendar me-1"></i>${e.date}</div>
                <p>${e.entry}</p>
            </div>`
        );
        entryHistory.prepend(entryItem);
    });
  }
  
  function getSentiment() {
    showLoading();
    
    $.get("/get_sentiment", (response) => {
        if (response.sentiment) {
            displaySentimentResult(response.sentiment);
        } else {
            $("#sentimentResult").html(`
                <div class="text-center py-4">
                    <i class="fas fa-info-circle fa-2x text-info mb-3"></i>
                    <p class="text-muted">${response.message}</p>
                </div>
            `);
            showToast(response.message, "info");
        }
    }).fail(() => {
        $("#sentimentResult").html(`
            <div class="text-center py-4">
                <i class="fas fa-exclamation-circle fa-2x text-danger mb-3"></i>
                <p class="text-muted">Error analyzing sentiment. Please try again.</p>
            </div>
        `);
        showToast("Error analyzing sentiment. Please try again.", "error");
    });
  }
  
  function getCurrentEntrySentiment() {
    const entry = $("#diaryEntry").val();
    if (entry.trim() === "") {
        showToast("Please enter some text first!", "error");
        return;
    }
    
    showLoading("#currentEntryAnalysis");
    
    $.post("/get_current_entry_sentiment", { entry: entry }, (response) => {
        if (response.sentiment) {
            displaySentimentResult(response.sentiment, "Current Entry", "#currentEntryAnalysis");
        } else {
            $("#currentEntryAnalysis").html(`
                <div class="text-center py-4">
                    <i class="fas fa-info-circle fa-2x text-info mb-3"></i>
                    <p class="text-muted">${response.message}</p>
                </div>
            `);
            showToast(response.message, "info");
        }
    }).fail(() => {
        $("#currentEntryAnalysis").html(`
            <div class="text-center py-4">
                <i class="fas fa-exclamation-circle fa-2x text-danger mb-3"></i>
                <p class="text-muted">Error analyzing sentiment. Please try again.</p>
            </div>
        `);
        showToast("Error analyzing sentiment. Please try again.", "error");
    });
  }
  
  function getComprehensiveAnalysis() {
    const entry = $("#diaryEntry").val();
    if (entry.trim() === "") {
        showToast("Please enter some text first!", "error");
        return;
    }
    
    showLoading("#currentEntryAnalysis");
    
    $.post("/get_comprehensive_analysis", { entry: entry }, (response) => {
        if (response.overall_sentiment) {
            displayComprehensiveAnalysis(response, "#currentEntryAnalysis");
        } else {
            $("#currentEntryAnalysis").html(`
                <div class="text-center py-4">
                    <i class="fas fa-info-circle fa-2x text-info mb-3"></i>
                    <p class="text-muted">${response.message}</p>
                </div>
            `);
            showToast(response.message, "info");
        }
    }).fail(() => {
        $("#currentEntryAnalysis").html(`
            <div class="text-center py-4">
                <i class="fas fa-exclamation-circle fa-2x text-danger mb-3"></i>
                <p class="text-muted">Error analyzing entry. Please try again.</p>
            </div>
        `);
        showToast("Error analyzing entry. Please try again.", "error");
    });
  }
  
  function getDailySentiment() {
    showLoading();
    
    $.get("/get_daily_sentiment", (response) => {
        if (response.sentiment) {
            displaySentimentResult(response.sentiment, "Today's Entries");
        } else {
            $("#sentimentResult").html(`
                <div class="text-center py-4">
                    <i class="fas fa-info-circle fa-2x text-info mb-3"></i>
                    <p class="text-muted">${response.message}</p>
                </div>
            `);
            showToast(response.message, "info");
        }
    }).fail(() => {
        $("#sentimentResult").html(`
            <div class="text-center py-4">
                <i class="fas fa-exclamation-circle fa-2x text-danger mb-3"></i>
                <p class="text-muted">Error analyzing daily sentiment. Please try again.</p>
            </div>
        `);
        showToast("Error analyzing daily sentiment. Please try again.", "error");
    });
  }
  
  function getWeeklySentiment() {
    showLoading();
    
    $.get("/get_weekly_sentiment", (response) => {
        if (response.sentiment) {
            displaySentimentResult(response.sentiment, "This Week's Entries");
            // Create line graph for weekly data
            if (response.daily_data) {
                createLineChart(response.daily_data, "Weekly Sentiment Trend");
            }
        } else {
            $("#sentimentResult").html(`
                <div class="text-center py-4">
                    <i class="fas fa-info-circle fa-2x text-info mb-3"></i>
                    <p class="text-muted">${response.message}</p>
                </div>
            `);
            showToast(response.message, "info");
        }
    }).fail(() => {
        $("#sentimentResult").html(`
            <div class="text-center py-4">
                <i class="fas fa-exclamation-circle fa-2x text-danger mb-3"></i>
                <p class="text-muted">Error analyzing weekly sentiment. Please try again.</p>
            </div>
        `);
        showToast("Error analyzing weekly sentiment. Please try again.", "error");
    });
}

function getMonthlySentiment() {
    showLoading();
    
    $.get("/get_monthly_sentiment", (response) => {
        if (response.sentiment) {
            displaySentimentResult(response.sentiment, "This Month's Entries");
            // Create line graph for monthly data
            if (response.daily_data) {
                createLineChart(response.daily_data, "Monthly Sentiment Trend");
            }
        } else {
            $("#sentimentResult").html(`
                <div class="text-center py-4">
                    <i class="fas fa-info-circle fa-2x text-info mb-3"></i>
                    <p class="text-muted">${response.message}</p>
                </div>
            `);
            showToast(response.message, "info");
        }
    }).fail(() => {
        $("#sentimentResult").html(`
            <div class="text-center py-4">
                <i class="fas fa-exclamation-circle fa-2x text-danger mb-3"></i>
                <p class="text-muted">Error analyzing monthly sentiment. Please try again.</p>
            </div>
        `);
        showToast("Error analyzing monthly sentiment. Please try again.", "error");
    });
}

// Function to create line chart
function createLineChart(dailyData, title) {
    // Sort dates
    const sortedDates = Object.keys(dailyData).sort();
    const sentiments = sortedDates.map(date => dailyData[date]);
    
    // Format dates for display
    const formattedDates = sortedDates.map(date => {
        const d = new Date(date);
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    });
    
    // Create chart container
    const chartContainer = $(`
        <div class="mt-4">
            <h4><i class="fas fa-chart-line me-2"></i>${title}</h4>
            <div class="chart-container" style="position: relative; height: 300px;">
                <canvas id="lineChart"></canvas>
            </div>
        </div>
    `);
    
    $("#sentimentResult").append(chartContainer);
    
    // Create line chart
    const ctx = document.getElementById('lineChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (window.lineChart) {
        window.lineChart.destroy();
    }
    
    window.lineChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: formattedDates,
            datasets: [{
                label: 'Sentiment Score',
                data: sentiments,
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                borderWidth: 3,
                pointBackgroundColor: function(context) {
                    const value = context.dataset.data[context.dataIndex];
                    if (value > 0.1) return '#2ecc71'; // Positive sentiment
                    if (value < -0.1) return '#e74c3c'; // Negative sentiment
                    return '#3498db'; // Neutral sentiment
                },
                pointRadius: 6,
                pointHoverRadius: 8,
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: -1,
                    max: 1,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    title: {
                        display: true,
                        text: 'Sentiment Score'
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Date'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Sentiment: ${context.parsed.y.toFixed(3)}`;
                        }
                    }
                }
            },
            interaction: {
                mode: 'index',
                intersect: false
            }
        }
    });
}

function displaySentimentResult(sentiment, period = "All Entries", targetElement = "#sentimentResult") {
    const sentimentWords = getSentimentWords(sentiment.compound);
    const resultHtml =
        `<div class="sentiment-result fade-in">
            <h3><i class="fas fa-heartbeat me-2"></i>Sentiment Analysis - ${period}</h3>
            <div class="sentiment-bar-container mt-3">
                <div class="sentiment-label"><span>Positive</span><span>${(sentiment.pos * 100).toFixed(1)}%</span></div>
                <div class="sentiment-bar" style="width: 0%; background: linear-gradient(90deg, #2ecc71, #27ae60);"></div>
  
                <div class="sentiment-label"><span>Neutral</span><span>${(sentiment.neu * 100).toFixed(1)}%</span></div>
                <div class="sentiment-bar" style="width: 0%; background: linear-gradient(90deg, #3498db, #2980b9);"></div>
  
                <div class="sentiment-label"><span>Negative</span><span>${(sentiment.neg * 100).toFixed(1)}%</span></div>
                <div class="sentiment-bar" style="width: 0%; background: linear-gradient(90deg, #e74c3c, #c0392b);"></div>
            </div>
            <div class="mt-3 p-3 bg-white rounded">
                <p class="mb-0"><strong>Overall sentiment:</strong> ${sentiment.compound.toFixed(2)} (${sentimentWords})</p>
            </div>
        </div>`;
    $(targetElement).html(resultHtml);
  
    setTimeout(() => {
        $(".sentiment-bar").each(function (index) {
            const width = (index === 0 ? sentiment.pos : index === 1 ? sentiment.neu : sentiment.neg) * 100 + "%";
            $(this).animate({ width: width }, 1000);
        });
    }, 100);
  
    const sentimentResultText = `${period}: Overall sentiment: ${sentiment.compound.toFixed(2)} (${sentimentWords})`;
    const sentimentUtterance = new SpeechSynthesisUtterance(sentimentResultText);
    window.speechSynthesis.speak(sentimentUtterance);
  
    sentimentUtterance.onend = () => {
        speakSuggestions(sentimentWords);
    };
  }
  
  function displayComprehensiveAnalysis(analysis, targetElement = "#sentimentResult") {
    let resultHtml = `
        <div class="comprehensive-analysis fade-in">
            <h3><i class="fas fa-brain me-2"></i>Comprehensive Analysis</h3>
            
            <div class="analysis-section">
                <h4><i class="fas fa-heart me-2"></i>1. Overall Sentiment</h4>
                <div class="p-3 bg-white rounded">
                    <p class="mb-0"><strong>${analysis.overall_sentiment.category}</strong> (${analysis.overall_sentiment.intensity})</p>
                </div>
            </div>
            
            <div class="analysis-section">
                <h4><i class="fas fa-smile me-2"></i>2. Emotion Detection</h4>
                <div class="emotion-bars">`;
    
    // Emotion bars
    for (const [emotion, percentage] of Object.entries(analysis.emotions)) {
        const color = emotion === 'Happy' ? 'linear-gradient(90deg, #2ecc71, #27ae60)' : 
                     emotion === 'Fear' ? 'linear-gradient(90deg, #9b59b6, #8e44ad)' : 
                     emotion === 'Anger' ? 'linear-gradient(90deg, #e74c3c, #c0392b)' : 
                     emotion === 'Sadness' ? 'linear-gradient(90deg, #3498db, #2980b9)' : 
                     emotion === 'Surprise' ? 'linear-gradient(90deg, #f1c40f, #f39c12)' : 
                     'linear-gradient(90deg, #e67e22, #d35400)';
        
        resultHtml += `
                    <div class="emotion-item">
                        <div class="d-flex justify-content-between">
                            <span><strong>${emotion}</strong></span>
                            <span>${percentage}%</span>
                        </div>
                        <div class="emotion-bar-container">
                            <div class="emotion-bar" style="width: 0%; background: ${color}">
                                ${percentage}%
                            </div>
                        </div>
                    </div>`;
    }
    
    resultHtml += `
                </div>
            </div>
            
            <div class="analysis-section">
                <h4><i class="fas fa-star me-2"></i>3. Highlights / Important Events</h4>`;
    
    if (analysis.highlights.length > 0) {
        resultHtml += `<ul class="list-group">`;
        analysis.highlights.forEach(highlight => {
            resultHtml += `<li class="list-group-item">${highlight}</li>`;
        });
        resultHtml += `</ul>`;
    } else {
        resultHtml += `<p class="text-muted">No specific highlights detected.</p>`;
    }
    
    resultHtml += `
            </div>
            
            <div class="analysis-section">
                <h4><i class="fas fa-brain me-2"></i>4. Mental State Patterns</h4>`;
    
    if (analysis.mental_patterns.length > 0) {
        resultHtml += `<div class="d-flex flex-wrap gap-2">`;
        analysis.mental_patterns.forEach(pattern => {
            resultHtml += `<span class="badge bg-info">${pattern}</span>`;
        });
        resultHtml += `</div>`;
    } else {
        resultHtml += `<p class="text-muted">No specific mental patterns detected.</p>`;
    }
    
    resultHtml += `
            </div>
            
            <div class="analysis-section">
                <h4><i class="fas fa-medal me-2"></i>5. Personal Strengths Shown</h4>`;
    
    if (analysis.strengths.length > 0) {
        resultHtml += `<div class="d-flex flex-wrap gap-2">`;
        analysis.strengths.forEach(strength => {
            resultHtml += `<span class="badge bg-success">${strength}</span>`;
        });
        resultHtml += `</div>`;
    } else {
        resultHtml += `<p class="text-muted">No specific strengths detected.</p>`;
    }
    
    resultHtml += `
            </div>
            
            <div class="analysis-section">
                <h4><i class="fas fa-lightbulb me-2"></i>6. Supportive Suggestions</h4>
                <ul class="list-group">`;
    
    // Suggestions
    analysis.suggestions.forEach(suggestion => {
        resultHtml += `<li class="list-group-item"><i class="fas fa-hand-point-right me-2 text-success"></i>${suggestion}</li>`;
    });
    
    resultHtml += `
                </ul>
            </div>
            
            <div class="analysis-section">
                <h4><i class="fas fa-align-left me-2"></i>7. One-Line Summary</h4>
                <div class="p-3 bg-white rounded">
                    <p class="mb-0">${analysis.summary}</p>
                </div>
            </div>
            
            <div class="analysis-section">
                <h4><i class="fas fa-chart-line me-2"></i>8. Trend Insight</h4>
                <div class="p-3 bg-white rounded">
                    <p class="mb-0">${analysis.trend_insight}</p>
                </div>
            </div>
        </div>`;
    
    $(targetElement).html(resultHtml);
    
    // Animate emotion bars
    setTimeout(() => {
        $(".emotion-bar").each(function() {
            const width = $(this).text().trim();
            $(this).animate({ width: width }, 1000);
        });
    }, 100);
    
    // Speak the summary
    const utterance = new SpeechSynthesisUtterance(analysis.summary);
    window.speechSynthesis.speak(utterance);
  }
  
  function getSentimentWords(compound) {
    if (compound >= 0.5) {
        return "Very Positive";
    } else if (compound > 0) {
        return "Positive";
    } else if (compound === 0) {
        return "Neutral";
    } else if (compound > -0.5) {
        return "Negative";
    } else {
        return "Very Negative";
    }
  }
  
  function speakSuggestions(sentimentWords) {
    const sentimentSuggestions = suggestions[sentimentWords];
    if (sentimentSuggestions) {
       
        const selectedSuggestions = sentimentSuggestions.sort(() => 0.5 - Math.random()).slice(0, 3);
        const utterance = new SpeechSynthesisUtterance(selectedSuggestions.join(". "));
        window.speechSynthesis.speak(utterance);
    }
  }
  
  function speakSentiment() {
    const sentimentResult = $("#sentimentResult").text();
    if (sentimentResult.trim() === "") {
        showToast("No analysis results to speak.", "info");
        return;
    }
    
    const utterance = new SpeechSynthesisUtterance(sentimentResult);
    window.speechSynthesis.speak(utterance);
    showToast("Speaking analysis results...", "info");
  }
  
  function clearEntries() {
    console.log("Clear entries function called");
    if (confirm("Are you sure you want to clear all entries? This cannot be undone.")) {
        console.log("User confirmed deletion");
        showLoading();
        
        $.post("/clear_entries", (response) => {
            console.log("Clear entries response:", response);
            $("#entryHistory").empty();
            $("#sentimentResult").html(`
                <div class="text-center py-4">
                    <i class="fas fa-check-circle fa-2x text-success mb-3"></i>
                    <p class="text-muted">All entries have been cleared</p>
                </div>
            `);
            showToast("All entries have been cleared.", "success");
            updateDashboard();
            updateEntryHistory([]);
        }).fail((xhr, status, error) => {
            console.log("Clear entries error:", error);
            $("#sentimentResult").html(`
                <div class="text-center py-4">
                    <i class="fas fa-exclamation-circle fa-2x text-danger mb-3"></i>
                    <p class="text-muted">Error clearing entries. Please try again.</p>
                </div>
            `);
            showToast("Error clearing entries. Please try again.", "error");
        });
    } else {
        console.log("User cancelled deletion");
    }
  }
  
  function showToast(message, type = "success") {
    // Remove existing toasts
    $(".toast").remove();
    
    const toast = $(`<div class="toast ${type}" role="alert" aria-live="assertive" aria-atomic="true">
        <div class="toast-body">
            ${message}
        </div>
    </div>`);
    
    $("body").append(toast);
    
    // Add animation classes
    toast.addClass("fade-in");
    
    // Auto remove after delay
    setTimeout(() => {
        toast.fadeOut(300, function() {
            $(this).remove();
        });
    }, 3000);
  }
  
  function startVoiceInput() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        showToast("Speech recognition is not supported in your browser.", "error");
        return;
    }
    
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    
    showToast("Listening... Speak now", "info");
    
    recognition.start();
  
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        $("#diaryEntry").val(transcript);
        showToast("Voice input received!", "success");
    };
  
    recognition.onerror = (event) => {
        console.error('Speech recognition error detected: ' + event.error);
        showToast("Error with voice input: " + event.error, "error");
    };
  
    recognition.onend = () => {
        console.log('Speech recognition service disconnected');
    };
  }
  
  function updateDashboard() {
    $.get("/get_sentiment_counts", (response) => {
        const counts = response.counts;
        $("#positiveCount").text(counts.happy);
        $("#neutralCount").text(counts.neutral);
        $("#negativeCount").text(counts.sad);
        renderBarChart(counts);
        renderPieChart(counts);
    }).fail(() => {
        showToast("Error fetching dashboard data. Please try again.", "error");
    });
  }
  
  function renderBarChart(counts) {
    const ctx = document.getElementById('sentimentBarChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (window.barChart) {
        window.barChart.destroy();
    }
    
    window.barChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Positive', 'Neutral', 'Negative'],
            datasets: [{
                label: 'Entry Counts',
                data: [counts.happy, counts.neutral, counts.sad],
                backgroundColor: [
                    'rgba(46, 204, 113, 0.7)',
                    'rgba(52, 152, 219, 0.7)',
                    'rgba(231, 76, 60, 0.7)'
                ],
                borderColor: [
                    'rgba(46, 204, 113, 1)',
                    'rgba(52, 152, 219, 1)',
                    'rgba(231, 76, 60, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
  }
  
  function renderPieChart(counts) {
    const ctx = document.getElementById('sentimentPieChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (window.pieChart) {
        window.pieChart.destroy();
    }
    
    window.pieChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Positive', 'Neutral', 'Negative'],
            datasets: [{
                label: 'Entry Counts',
                data: [counts.happy, counts.neutral, counts.sad],
                backgroundColor: [
                    'rgba(46, 204, 113, 0.7)',
                    'rgba(52, 152, 219, 0.7)',
                    'rgba(231, 76, 60, 0.7)'
                ],
                borderColor: [
                    'rgba(46, 204, 113, 1)',
                    'rgba(52, 152, 219, 1)',
                    'rgba(231, 76, 60, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
  }
  
  $(document).ready(() => {
    // Initialize dashboard
    updateDashboard();
    
    // Load existing entries
    $.get("/get_entries", (response) => {
        updateEntryHistory(response.entries);
    });
    
    // Add subtle animations to cards on load
    setTimeout(() => {
        $('.card').each(function(index) {
            $(this).delay(100 * index).queue(function() {
                $(this).addClass('fade-in').dequeue();
            });
        });
    }, 300);
  });