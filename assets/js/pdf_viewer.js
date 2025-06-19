// pdf_viewer.js
(function() {
    const DEBUG = true;
    const MAX_INIT_RETRIES = 10;
    const RETRY_DELAY = 100;
    const DEFAULT_RENDER_SCALE = 1; // Higher default scale for better resolution
    
    function log(message, type = 'info') {
        if (DEBUG) {
            const timestamp = new Date().toISOString();
            console[type](`[PDF Viewer ${timestamp}] ${message}`);
        }
    }

    // Improved PDF.js loading with retry mechanism
    async function loadPDFJS() {
        log('Starting PDF.js load sequence');
        
        if (typeof window.pdfjsLib !== 'undefined') {
            log('PDF.js already loaded');
            return;
        }

        try {
            // Load main script with version lock
            await new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js';
                script.onload = resolve;
                script.onerror = reject;
                document.head.appendChild(script);
            });

            // Configure worker with same version
            window.pdfjsLib.GlobalWorkerOptions.workerSrc = 
                'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
                
            log('PDF.js loaded successfully');
        } catch (error) {
            log(`Failed to load PDF.js: ${error.message}`, 'error');
            throw error;
        }
    }

    class PDFViewer {
        constructor(containerId) {
            this.containerId = containerId;
            this.container = null;
            this.pagesContainer = null;
            this.pdfDoc = null;
            this.pageCanvases = new Map();
            this.observer = null;
            this.scale = DEFAULT_RENDER_SCALE;
            this.pagesLoaded = false;
            this.initRetries = 0;
            this.isFirstLoad = true;
            
            this.handleScroll = this.handleScroll.bind(this);
            this.renderVisiblePages = this.renderVisiblePages.bind(this);

            this.debouncedRender = this.debounce(this.renderVisiblePages, 100);
        }

        debounce(func, wait) {
            let timeout;
            let lastArgs;
            let lastThis;
            let lastCallTime;
            let lastInvokeTime = 0;
            let result;
        
            const shouldInvoke = (time) => {
                const timeSinceLastCall = time - lastCallTime;
                const timeSinceLastInvoke = time - lastInvokeTime;
                return !lastCallTime || timeSinceLastCall >= wait || timeSinceLastCall < 0 || timeSinceLastInvoke >= wait;
            };
        
            const invokeFunc = (time) => {
                lastInvokeTime = time;
                result = func.apply(lastThis, lastArgs);
                lastThis = lastArgs = null;
                return result;
            };
        
            const debouncedFunction = function(...args) {
                const time = Date.now();
                lastThis = this;
                lastArgs = args;
                lastCallTime = time;
        
                if (timeout) {
                    clearTimeout(timeout);
                }
        
                if (shouldInvoke(time)) {
                    return invokeFunc(time);
                }
        
                timeout = setTimeout(() => {
                    const time = Date.now();
                    if (shouldInvoke(time)) {
                        invokeFunc(time);
                    }
                }, wait);
        
                return result;
            };
        
            debouncedFunction.cancel = () => {
                if (timeout) {
                    clearTimeout(timeout);
                    timeout = null;
                }
                lastInvokeTime = 0;
                lastCallTime = 0;
                lastArgs = lastThis = null;
            };
        
            debouncedFunction.flush = () => {
                if (timeout) {
                    const time = Date.now();
                    if (shouldInvoke(time)) {
                        return invokeFunc(time);
                    }
                }
                return result;
            };
        
            return debouncedFunction;
        }
    
        async initialize() {
            return new Promise(async (resolve, reject) => {
                const initContainer = async () => {
                    this.container = document.getElementById(this.containerId);
                    if (!this.container || this.container.clientWidth === 0) {
                        if (this.initRetries < MAX_INIT_RETRIES) {
                            this.initRetries++;
                            log(`Container not ready, retrying (${this.initRetries}/${MAX_INIT_RETRIES})...`);
                            setTimeout(initContainer, RETRY_DELAY);
                            return;
                        }
                        reject(new Error('Container initialization failed'));
                        return;
                    }

                    try {
                        // Clear existing content
                        this.container.innerHTML = '';
                        
                        // Set container styles
                        this.container.style.cssText = `
                            position: relative;
                            width: 100%;
                            height: 100vh;
                            overflow-y: auto;
                            overflow-x: hidden;
                            background: #f5f5f5;
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                        `;
                        
                        // Create pages container
                        this.pagesContainer = document.createElement('div');
                        this.pagesContainer.style.cssText = `
                            position: relative;
                            width: 100%;
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                            gap: 20px;
                            padding: 20px 0;
                            min-height: 100%;
                        `;
                        
                        this.container.appendChild(this.pagesContainer);
                        this.setupIntersectionObserver();
                        this.container.addEventListener('scroll', this.handleScroll);
                        
                        resolve();
                    } catch (error) {
                        reject(error);
                    }
                };

                await initContainer();
            });
        }
        async cleanup() {
            this.container?.removeEventListener('scroll', this.handleScroll);
            
            if (this.observer) {
                this.observer.disconnect();
                this.observer = null;
            }
            
            if (this.pdfDoc) {
                await this.pdfDoc.destroy();
                this.pdfDoc = null;
            }
    
            this.pageCanvases.clear();
            if (this.container) {
                this.container.innerHTML = '';
            }
            
            this.pagesContainer = null;
            this.pagesLoaded = false;
            this.initialRenderComplete = false;
            this.scale = DEFAULT_RENDER_SCALE;
        }
        async loadDocument(arrayBuffer) {
            try {
                await this.cleanup();
                await this.initialize();
                
                // Clear existing content
                this.pagesContainer.innerHTML = '';
                this.pageCanvases.clear();
                
                // Load document with enhanced caching
                const loadingTask = pdfjsLib.getDocument({
                    data: arrayBuffer,
                    cMapUrl: 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/cmaps/',
                    cMapPacked: true,
                    enableXfa: true
                });
                
                this.pdfDoc = await loadingTask.promise;
                
                // Setup pages with loading indicators
                await this.setupPages();
                
                // Initial render of visible pages
                this.initialRenderComplete = false;
                await this.renderVisiblePages();
                this.initialRenderComplete = true;
                
                // Scroll to top after document is loaded
                this.container.scrollTop = 0;
                
                return true;
            } catch (error) {
                console.error('Error loading document:', error);
                throw error;
            }
        }
        async setHighlights(highlights) {
            console.log('setHighlights called with:', highlights);
            
            if (!this.pagesLoaded || !this.pdfDoc) {
                console.log('PDF not ready - loaded:', this.pagesLoaded, 'doc:', !!this.pdfDoc);
                return;
            }
            
            // Clear existing highlights
            const existingHighlights = this.container.querySelectorAll('.highlight-overlay');
            existingHighlights.forEach(el => el.remove());
            
            let firstHighlight = null;
            
            for (const highlight of highlights) {
                console.log('Processing highlight:', highlight);
                const pageElement = this.pagesContainer.querySelector(`[data-page-number="${highlight.page + 1}"]`);
                
                if (!pageElement) {
                    console.log('Page element not found for page:', highlight.page + 1);
                    continue;
                }
                
                const overlay = document.createElement('div');
                overlay.className = 'highlight-overlay';
                
                // Scale coordinates according to current page scale
                const scaledCoords = {
                    x1: highlight.coords.x1 * this.scale,
                    y1: highlight.coords.y1 * this.scale,
                    x2: highlight.coords.x2 * this.scale,
                    y2: highlight.coords.y2 * this.scale
                };
                
                // Apply the highlight
                overlay.style.cssText = `
                    position: absolute;
                    background-color: rgba(255, 255, 0, 0.3);
                    pointer-events: none;
                    left: ${scaledCoords.x1}px;
                    top: ${scaledCoords.y1}px;
                    width: ${scaledCoords.x2 - scaledCoords.x1}px;
                    height: ${scaledCoords.y2 - scaledCoords.y1}px;
                    z-index: 1;
                `;
                
                pageElement.appendChild(overlay);
                
                // Store the first highlight for scrolling
                if (!firstHighlight) {
                    firstHighlight = {
                        pageElement: pageElement,
                        coords: scaledCoords
                    };
                }
            }
            
            // Scroll to the first highlight if it exists
            if (firstHighlight) {
                this.scrollToHighlight(firstHighlight);
            }
            
            console.log('Highlight rendering complete');
        }
        async setupPages() {
            const numPages = this.pdfDoc.numPages;
            
            // Get first page to calculate dimensions
            const firstPage = await this.pdfDoc.getPage(1);
            const viewport = firstPage.getViewport({ scale: 1.0 });
            
            // Calculate scale to fit container width with margins
            const containerWidth = this.container.clientWidth;
            this.scale = ((containerWidth - 60) / viewport.width) * DEFAULT_RENDER_SCALE;
            
            // Create placeholders for all pages
            for (let i = 1; i <= numPages; i++) {
                const pageContainer = document.createElement('div');
                pageContainer.dataset.pageNumber = i;
                pageContainer.style.cssText = `
                    width: ${viewport.width * this.scale}px;
                    height: ${viewport.height * this.scale}px;
                    background-color: white;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin: 10px auto;
                    position: relative;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                `;
                
                // Add loading indicator
                const loadingIndicator = document.createElement('div');
                loadingIndicator.className = 'loading-indicator';
                loadingIndicator.textContent = `Loading page ${i}...`;
                pageContainer.appendChild(loadingIndicator);
                
                const canvas = document.createElement('canvas');
                canvas.style.cssText = `
                    display: none;
                    width: 100%;
                    height: 100%;
                `;
                pageContainer.appendChild(canvas);
                
                this.pagesContainer.appendChild(pageContainer);
                this.observer.observe(pageContainer);
            }
            
            this.pagesLoaded = true;
        }
        
        scrollToHighlight(highlight) {
            // Wait for any ongoing rendering to complete
            setTimeout(() => {
                const { pageElement, coords } = highlight;
                
                // Calculate the scroll position
                const containerRect = this.container.getBoundingClientRect();
                const pageRect = pageElement.getBoundingClientRect();
                const scrollTop = pageElement.offsetTop + coords.y1 - containerRect.height / 4;
                
                // Smooth scroll to the highlight
                this.container.scrollTo({
                    top: scrollTop,
                    behavior: 'smooth'
                });
                
                // Add a temporary focus effect
                const focusEffect = document.createElement('div');
                focusEffect.style.cssText = `
                    position: absolute;
                    left: ${coords.x1}px;
                    top: ${coords.y1}px;
                    width: ${coords.x2 - coords.x1}px;
                    height: ${coords.y2 - coords.y1}px;
                    border: 2px solid #FFD700;
                    border-radius: 3px;
                    animation: focusPulse 2s ease-out;
                    pointer-events: none;
                    z-index: 2;
                `;
                
                // Add the animation style if it doesn't exist
                if (!document.querySelector('#highlight-animations')) {
                    const style = document.createElement('style');
                    style.id = 'highlight-animations';
                    style.textContent = `
                        @keyframes focusPulse {
                            0% { transform: scale(1); opacity: 1; }
                            70% { transform: scale(1.05); opacity: 0.7; }
                            100% { transform: scale(1); opacity: 0; }
                        }
                    `;
                    document.head.appendChild(style);
                }
                
                pageElement.appendChild(focusEffect);
                
                // Remove the focus effect after animation
                setTimeout(() => {
                    focusEffect.remove();
                }, 2000);
                
            }, 100); // Short delay to ensure rendering is complete
        }
        setupIntersectionObserver() {
            this.observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    const pageNum = parseInt(entry.target.dataset.pageNumber);
                    if (entry.isIntersecting && !this.pageCanvases.get(pageNum)?.rendered) {
                        this.renderPage(pageNum);
                    }
                });
            }, {
                root: this.container,
                rootMargin: '100px 0px',
                threshold: 0.1
            });
        }


        async renderPage(pageNum) {
            if (!this.pagesLoaded) return;
    
            try {
                const page = await this.pdfDoc.getPage(pageNum);
                const viewport = page.getViewport({ scale: this.scale });
                
                const pageContainer = this.pagesContainer.querySelector(`[data-page-number="${pageNum}"]`);
                const canvas = pageContainer.querySelector('canvas');
                const loadingIndicator = pageContainer.querySelector('.loading-indicator');
                
                // Set canvas dimensions
                canvas.width = viewport.width;
                canvas.height = viewport.height;
                
                const ctx = canvas.getContext('2d', {
                    alpha: false,
                    desynchronized: true
                });
                
                // Clear and set white background
                ctx.fillStyle = 'white';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                // Render the page with WebGL acceleration
                await page.render({
                    canvasContext: ctx,
                    viewport: viewport,
                    enableWebGL: true,
                    renderInteractiveForms: true
                }).promise;
                
                // Show canvas and remove loading indicator
                canvas.style.display = 'block';
                loadingIndicator?.remove();
                
                this.pageCanvases.set(pageNum, { rendered: true });
                
            } catch (error) {
                console.error(`Error rendering page ${pageNum}:`, error);
            }
        }
    
        handleScroll() {
            if (!this.pagesLoaded) return;
            this.debouncedRender();
        }
    
        renderVisiblePages() {
            // The IntersectionObserver handles the rendering
        }
    
        cleanup() {
            this.container.removeEventListener('scroll', this.handleScroll);
            if (this.observer) {
                this.observer.disconnect();
            }
            this.pageCanvases.clear();
            if (this.pagesContainer) {
                this.pagesContainer.innerHTML = '';
            }
        }
    }
    
    
    // Initialize PDF viewer
    async function init() {
        try {
            log('Starting initialization sequence');
            await loadPDFJS();
            
            // Export initialization function
            window.initPDFViewer = function(containerId) {
                const viewer = new PDFViewer(containerId);
                viewer.initialize().catch(error => {
                    log(`Viewer initialization failed: ${error.message}`, 'error');
                });
                viewer.jumpToPage = jumpToPage;
                viewer.scrollToPosition = scrollToPosition;
                return viewer;
            };
            
            window.pdfViewerInitialized = true;
            window.dispatchEvent(new Event('pdfViewerReady'));
            log('Initialization complete');
        } catch (error) {
            log(`Initialization failed: ${error.message}`, 'error');
            throw error;
        }
    }

    // Start initialization
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // Function to jump to a specific page
    function jumpToPage(pageNumber) {
        // PDF pages are 1-based, but internally we use 0-based indexing
        const zeroBasedPage = pageNumber;
        
        try {
        // Check if the viewer and page are valid
        if (!this.pdfViewer || !this.pdfViewer.pdfDocument) return;
        
        // Ensure valid page number
        const pageCount = this.pdfViewer.pdfDocument.numPages;
        if (zeroBasedPage < 0 || zeroBasedPage >= pageCount) return;
        
        // Go to the specified page
        this.pdfViewer.currentPageNumber = zeroBasedPage + 1;
        console.log("Jumped to page:", zeroBasedPage + 1);
        } catch (error) {
        console.error('Error jumping to page:', error);
        }
    }

    // Function to scroll to a specific position on a page
// Function to scroll to a specific position on a page
function scrollToPosition(pageNumber, yPos) {
    try {
      // PDF pages are 1-based, but internally we use 0-based indexing
      const zeroBasedPage = pageNumber;
      
      // Get the page viewport
      const page = this.pdfViewer.getPageView(zeroBasedPage);
      if (!page) return;
      
      const viewport = page.viewport;
      
      // Calculate position relative to viewport
      // yPos is typically in PDF coordinates (bottom-up)
      // Need to convert to DOM coordinates (top-down)
      const domYPos = viewport.height - yPos;
      
      // Get the page div
      const pageDiv = document.querySelector(`[data-page-number="${zeroBasedPage + 1}"]`);
      if (pageDiv) {
        // Scroll the page div into view
        pageDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
        
        // Then scroll to the specific position
        setTimeout(() => {
          // Get the container
          const container = document.getElementById('pdf-js-viewer');
          if (container) {
            // Add offset to scroll to the highlight position
            container.scrollTop += domYPos - 200; // Offset to position highlight in view
            console.log("Scrolled to position:", domYPos);
          }
        }, 300);
      }
    } catch (error) {
      console.error('Error scrolling to position:', error);
    }
  }




})();