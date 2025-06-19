// Create this file at assets/force_refresh.js

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        refresh_document_page: function(contents) {
            // Check if we're on the documents page
            if (window.location.pathname === '/documents') {
                if (contents) {
                    console.log("Upload detected, refreshing in 3 seconds...");
                    
                    // Set a timer to refresh the page after upload
                    setTimeout(function() {
                        console.log("Refreshing page now");
                        window.location.reload();
                    }, 14000);
                }
            }
            return '';
        },
        format_tag_input: function(value) {
            if (!value) return '';
            
            // Split by commas and trim whitespace
            const tags = value.split(',').map(tag => tag.trim());
            
            // Remove empty tags and join back with commas and spaces
            return tags.filter(tag => tag.length > 0).join(', ');
        }
    }
});