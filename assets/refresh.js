// Create this file at assets/force_refresh.js

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    refresh_document_page: {
        force_refresh_after_upload: function(contents) {
            // Only execute when contents are provided (after upload)
            if (contents) {
                // Set a timeout to allow the server to process the upload
                setTimeout(function() {
                    console.log("Refreshing page after upload...");
                    window.location.href = '/documents';
                }, 1000); // 2-second delay
            }
            return '';
        }
    }
});