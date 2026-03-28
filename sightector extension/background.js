chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "getSelectedText") {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
          chrome.scripting.executeScript({
              target: { tabId: tabs[0].id },
              function: getSelectedText,
          }, (results) => {
              const selectedText = results && results.length > 0 ? results[0].result : "No text selected.";
              sendResponse({ selectedText });
          });
      });
      return true; // Keep the message channel open for async response
  }
});

// Function to get selected text from the active tab
function getSelectedText() {
  const selection = window.getSelection().toString().trim();
  return selection.length > 0 ? selection : "No text selected.";
}
