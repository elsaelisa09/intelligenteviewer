
if (!window.dash_clientside) {
    window.dash_clientside = {};
}

window.dash_clientside.clientside = window.dash_clientside.clientside || {};

// Add this function to handle hiding the delete success animation
window.dash_clientside.clientside.hide_delete_success = function(style) {
    if (style && style.display === 'block') {
        // Use a timeout to allow the animation to be seen
        setTimeout(function() {
            document.getElementById('delete-success-animation').style.display = 'none';
        }, 3000);
    }
    return window.dash_clientside.no_update;
};

// Add this function to smoothly transition the delete modal when deletion is confirmed
window.dash_clientside.clientside.transition_delete_modal = function(n_clicks) {
    if (!n_clicks) return [window.dash_clientside.no_update, window.dash_clientside.no_update];
    
    // Hide confirmation content with fade effect
    const confirmContent = document.getElementById('delete-confirmation-content');
    const progressContent = document.getElementById('delete-progress-content');
    
    if (confirmContent && progressContent) {
        // Fade out confirmation
        confirmContent.style.opacity = '0';
        confirmContent.style.transition = 'opacity 0.3s ease';
        
        setTimeout(function() {
            // Hide confirmation and show progress
            confirmContent.style.display = 'none';
            progressContent.style.display = 'block';
            progressContent.style.opacity = '0';
            
            // Fade in progress
            setTimeout(function() {
                progressContent.style.opacity = '1';
                progressContent.style.transition = 'opacity 0.3s ease';
            }, 50);
        }, 300);
    }
    
    return [
        {"display": "none", "opacity": "0"},
        {"display": "block", "opacity": "1"}
    ];
};

// Add this function to refresh document page after upload (existing function)
window.dash_clientside.clientside.refresh_document_page = function(upload_contents) {
    if (upload_contents) {
        setTimeout(function() {
            window.location.reload();
        }, 10000);
    }
    return '';
};