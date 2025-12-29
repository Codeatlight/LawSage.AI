// Tab switching functionality
document.addEventListener('DOMContentLoaded', () => {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.getAttribute('data-tab');

            // Remove active class from all tabs and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Add active class to clicked tab and corresponding content
            button.classList.add('active');
            const targetContent = document.getElementById(`${targetTab}-content`);
            if (targetContent) {
                targetContent.classList.add('active');
            }
        });
    });

    // Initialize key clauses and timeline from summary text
    initializeSummaryTabs();

    // File input change handler
    const fileInput = document.getElementById('file-input');
    const fileName = document.getElementById('file-name');
    if (fileInput && fileName) {
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                fileName.textContent = `Selected: ${file.name}`;
                fileName.style.display = 'block';
            } else {
                fileName.style.display = 'none';
            }
        });
    }

    // Drag and drop functionality
    const uploadArea = document.getElementById('upload-area');
    if (uploadArea) {
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#f4d03f';
            uploadArea.style.backgroundColor = 'rgba(44, 44, 44, 0.95)';
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.borderColor = '#d4af37';
            uploadArea.style.backgroundColor = 'rgba(44, 44, 44, 0.8)';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#d4af37';
            uploadArea.style.backgroundColor = 'rgba(44, 44, 44, 0.8)';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                if (fileName) {
                    fileName.textContent = `Selected: ${files[0].name}`;
                    fileName.style.display = 'block';
                }
            }
        });
    }

    // Download button functionality
    const downloadBtn = document.getElementById('download-btn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', () => {
            downloadSummary();
        });
    }

    // Share button functionality
    const shareBtn = document.getElementById('share-btn');
    if (shareBtn) {
        shareBtn.addEventListener('click', () => {
            shareSummary();
        });
    }
});

// Initialize summary tabs with extracted information
function initializeSummaryTabs() {
    const summaryText = document.getElementById('summary-text')?.textContent;
    if (!summaryText) return;

    // Extract key clauses (simple extraction - can be enhanced)
    extractKeyClauses(summaryText);
    
    // Extract timeline information
    extractTimeline(summaryText);
}

// Extract key clauses from summary
function extractKeyClauses(summaryText) {
    const clausesList = document.getElementById('key-clauses-list');
    if (!clausesList) return;

    // Simple extraction: look for sentences with key legal terms
    const sentences = summaryText.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const keyTerms = ['obligation', 'confidentiality', 'termination', 'agreement', 'contract', 
                      'liability', 'indemnity', 'warranty', 'breach', 'remedy', 'jurisdiction'];
    
    const keyClauses = sentences.filter(sentence => {
        const lowerSentence = sentence.toLowerCase();
        return keyTerms.some(term => lowerSentence.includes(term));
    }).slice(0, 10); // Limit to 10 clauses

    clausesList.innerHTML = '';
    if (keyClauses.length === 0) {
        clausesList.innerHTML = '<li>No key clauses identified. View Executive Summary for full details.</li>';
    } else {
        keyClauses.forEach(clause => {
            const li = document.createElement('li');
            li.textContent = clause.trim();
            clausesList.appendChild(li);
        });
    }
}

// Extract timeline information
function extractTimeline(summaryText) {
    const timelineItems = document.getElementById('timeline-items');
    if (!timelineItems) return;

    // Extract dates and events
    const datePattern = /\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4})\b/g;
    const dates = summaryText.match(datePattern) || [];
    
    // Extract sentences with dates
    const sentences = summaryText.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const timelineEvents = sentences.filter(sentence => {
        return datePattern.test(sentence);
    }).slice(0, 5); // Limit to 5 events

    timelineItems.innerHTML = '';
    if (timelineEvents.length === 0) {
        timelineItems.innerHTML = '<div class="timeline-item"><p>No timeline information available.</p></div>';
    } else {
        timelineEvents.forEach((event, index) => {
            const dateMatch = event.match(datePattern);
            const date = dateMatch ? dateMatch[0] : 'Date not specified';
            const description = event.replace(datePattern, '').trim();

            const timelineItem = document.createElement('div');
            timelineItem.className = 'timeline-item';
            timelineItem.innerHTML = `
                <h4>${date}</h4>
                <p>${description}</p>
            `;
            timelineItems.appendChild(timelineItem);
        });
    }
}

// Download summary as text file
function downloadSummary() {
    const summaryText = document.getElementById('summary-text')?.textContent;
    const documentTitle = document.getElementById('document-title')?.textContent || 'Legal Document';
    
    if (!summaryText) {
        alert('No summary available to download.');
        return;
    }

    const content = `Document: ${documentTitle}\n\nAI Summary:\n${summaryText}\n\nGenerated by Lawsage AI`;
    const blob = new Blob([content], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${documentTitle.replace(/\s+/g, '_')}_Summary.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

// Share summary functionality
function shareSummary() {
    const summaryText = document.getElementById('summary-text')?.textContent;
    
    if (!summaryText) {
        alert('No summary available to share.');
        return;
    }

    // Check if Web Share API is available
    if (navigator.share) {
        navigator.share({
            title: 'Legal Document Summary - Lawsage AI',
            text: summaryText,
        }).catch(err => {
            console.error('Error sharing:', err);
            fallbackShare(summaryText);
        });
    } else {
        fallbackShare(summaryText);
    }
}

// Fallback share method (copy to clipboard)
function fallbackShare(summaryText) {
    const textArea = document.createElement('textarea');
    textArea.value = summaryText;
    textArea.style.position = 'fixed';
    textArea.style.opacity = '0';
    document.body.appendChild(textArea);
    textArea.select();
    
    try {
        document.execCommand('copy');
        alert('Summary copied to clipboard!');
    } catch (err) {
        console.error('Failed to copy:', err);
        alert('Failed to share summary. Please copy manually.');
    }
    
    document.body.removeChild(textArea);
}
