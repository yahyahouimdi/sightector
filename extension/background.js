const NO_SELECTION = "";

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action !== "getSelectedText") return false;
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const tab = tabs[0];
    // chrome:// and store pages cannot be scripted.
    if (!tab || !/^https?:|^file:/.test(tab.url || "")) {
      sendResponse({ selectedText: NO_SELECTION });
      return;
    }
    chrome.scripting.executeScript({ target: { tabId: tab.id }, func: getSelectedText }, (results) => {
      if (chrome.runtime.lastError) {
        sendResponse({ selectedText: NO_SELECTION });
        return;
      }
      sendResponse({ selectedText: results?.[0]?.result || NO_SELECTION });
    });
  });
  return true; // keep the message channel open for the async response
});

// Runs inside the page.
function getSelectedText() {
  return window.getSelection().toString().trim();
}
