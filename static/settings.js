// Settings page functionality

// Load saved settings when page loads
$(document).ready(() => {
    loadSettings();
    
    // Add event listeners for toggle switches
    $('#darkModeToggle').change(toggleDarkMode);
    $('#autoThemeToggle').change(toggleAutoTheme);
    $('#dailyReminderToggle').change(toggleDailyReminder);
    $('#weeklySummaryToggle').change(toggleWeeklySummary);
});

// Load saved settings from localStorage
function loadSettings() {
    const settings = JSON.parse(localStorage.getItem('moodmateSettings')) || {
        darkMode: false,
        autoTheme: false,
        dailyReminder: false,
        weeklySummary: false
    };
    
    // Apply settings to UI
    $('#darkModeToggle').prop('checked', settings.darkMode);
    $('#autoThemeToggle').prop('checked', settings.autoTheme);
    $('#dailyReminderToggle').prop('checked', settings.dailyReminder);
    $('#weeklySummaryToggle').prop('checked', settings.weeklySummary);
    
    // Apply dark mode if enabled
    if (settings.darkMode) {
        applyDarkMode();
    }
    
    // Check for system preference if auto theme is enabled
    if (settings.autoTheme) {
        checkSystemPreference();
    }
}

// Save settings to localStorage
function saveSettings() {
    const settings = {
        darkMode: $('#darkModeToggle').is(':checked'),
        autoTheme: $('#autoThemeToggle').is(':checked'),
        dailyReminder: $('#dailyReminderToggle').is(':checked'),
        weeklySummary: $('#weeklySummaryToggle').is(':checked')
    };
    
    localStorage.setItem('moodmateSettings', JSON.stringify(settings));
    
    // Show success message
    showToast("Settings saved successfully!", "success");
    
    // Apply dark mode if needed
    if (settings.darkMode) {
        applyDarkMode();
    } else {
        removeDarkMode();
    }
}

// Reset settings to default
function resetSettings() {
    if (confirm("Are you sure you want to reset all settings to default?")) {
        localStorage.removeItem('moodmateSettings');
        loadSettings();
        showToast("Settings reset to default!", "info");
    }
}

// Toggle dark mode
function toggleDarkMode() {
    const isEnabled = $(this).is(':checked');
    
    if (isEnabled) {
        applyDarkMode();
    } else {
        removeDarkMode();
    }
}

// Apply dark mode styles
function applyDarkMode() {
    $('body').addClass('dark-mode');
    $('.card').addClass('dark-mode');
    $('.app-header').addClass('dark-mode');
    $('.app-footer').addClass('dark-mode');
    
    // Update CSS variables for dark mode
    document.documentElement.style.setProperty('--bg-color', '#1a1a2e');
    document.documentElement.style.setProperty('--text-color', '#eeeeee');
    document.documentElement.style.setProperty('--card-bg', '#16213e');
    document.documentElement.style.setProperty('--header-bg', '#0f3460');
    document.documentElement.style.setProperty('--footer-bg', '#0f3460');
}

// Remove dark mode styles
function removeDarkMode() {
    $('body').removeClass('dark-mode');
    $('.card').removeClass('dark-mode');
    $('.app-header').removeClass('dark-mode');
    $('.app-footer').removeClass('dark-mode');
    
    // Reset CSS variables to light mode
    document.documentElement.style.setProperty('--bg-color', '#cce5ff');
    document.documentElement.style.setProperty('--text-color', '#333333');
    document.documentElement.style.setProperty('--card-bg', '#ffffff');
    document.documentElement.style.setProperty('--header-bg', '#3498db');
    document.documentElement.style.setProperty('--footer-bg', '#2c3e50');
}

// Toggle auto theme
function toggleAutoTheme() {
    const isEnabled = $(this).is(':checked');
    
    if (isEnabled) {
        checkSystemPreference();
        // Listen for system theme changes
        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', checkSystemPreference);
        }
    } else {
        // Stop listening for system theme changes
        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: dark)').removeEventListener('change', checkSystemPreference);
        }
    }
}

// Check system preference and apply theme
function checkSystemPreference() {
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        applyDarkMode();
        $('#darkModeToggle').prop('checked', true);
    } else {
        removeDarkMode();
        $('#darkModeToggle').prop('checked', false);
    }
}

// Toggle daily reminder
function toggleDailyReminder() {
    // Implementation for daily reminder toggle
    const isEnabled = $(this).is(':checked');
    console.log("Daily reminder toggled:", isEnabled);
}

// Toggle weekly summary
function toggleWeeklySummary() {
    // Implementation for weekly summary toggle
    const isEnabled = $(this).is(':checked');
    console.log("Weekly summary toggled:", isEnabled);
}

// Show toast notification
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