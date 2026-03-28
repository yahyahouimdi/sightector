document.addEventListener("DOMContentLoaded", () => {
    const selectedTextArea = document.getElementById("selectedText");
    const scanButton = document.getElementById("scanButton");
    const resultDiv = document.getElementById("result");
    const linkResultDiv = document.getElementById("linkResult");
    const checkLinkButton = document.getElementById("checkLinkButton");
    const harassmentAdviceContainer = document.getElementById("harassmentAdviceContainer");
    const linkAdviceContainer = document.getElementById("linkAdviceContainer");
    const linkDetailsContainer = document.getElementById("linkDetailsContainer");

    // Request the selected text from the background script
    chrome.runtime.sendMessage({ action: "getSelectedText" }, (response) => {
        selectedTextArea.value = response?.selectedText || "No text selected";
    });

    // Handle the scan button click for harassment detection
    scanButton.addEventListener("click", async () => {
        const text = selectedTextArea.value.trim();
        if (!text || text === "No text selected") {
            resultDiv.innerText = "Please select some text to scan.";
            resultDiv.style.backgroundColor = "#ffcc00"; // Yellow for warning
            return;
        }

        try {
            const response = await fetch("http://localhost:5000/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text }),
            });

            const data = await response.json();
            resultDiv.innerText = data.harassment ? "Harassment detected." : "No harassment detected.";
            resultDiv.style.backgroundColor = data.harassment ? "#cc163f" : "#28a745";

            // Update harassment advice
            harassmentAdviceContainer.innerHTML = "";
            if (data.harassment) {
                const harassmentAdvice = document.createElement("p");
                harassmentAdvice.innerText = "This text contains harmful content. Consider reporting or blocking.";
                harassmentAdvice.style.fontWeight = "bold";
                harassmentAdvice.style.color = "#cc163f";
                harassmentAdviceContainer.appendChild(harassmentAdvice);

                // Add report and block messages
                const reportMessage = document.createElement("p");
                reportMessage.innerText = "Report this content to keep the community safe.";
                reportMessage.style.color = "#d9534f";
                harassmentAdviceContainer.appendChild(reportMessage);

                const blockMessage = document.createElement("p");
                blockMessage.innerText = "You can block the sender to prevent further interactions.";
                blockMessage.style.color = "#d9534f";
                harassmentAdviceContainer.appendChild(blockMessage);
            }
        } catch (error) {
            resultDiv.innerText = "Error connecting to the server.";
            resultDiv.style.backgroundColor = "#cc163f";
        }
    });

    // Handle the check link button click for basic malicious link detection
    checkLinkButton.addEventListener("click", () => {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            const currentTabUrl = tabs[0]?.url;
            if (!currentTabUrl) return;

            const isMalicious = checkIfLinkIsMalicious(currentTabUrl);
            linkResultDiv.innerText = isMalicious ? "Potentially Malicious Link." : "Safe Link.";
            linkResultDiv.style.backgroundColor = isMalicious ? "#cc163f" : "#28a745";

            // Update security advice
            linkAdviceContainer.innerHTML = "";
            linkDetailsContainer.innerHTML = "";
            if (isMalicious) {
                
                const securityAdvice = document.createElement("p");
                securityAdvice.innerText = "For your safety, avoid sharing personal data on this page.";
                securityAdvice.style.fontWeight = "bold";
                linkAdviceContainer.appendChild(securityAdvice);

                const warningMessage = document.createElement("p");
                warningMessage.innerText = "If this link looks suspicious, do NOT click on it.";
                warningMessage.style.color = "#d9534f";
                linkAdviceContainer.appendChild(warningMessage);
            }

            // Display link details
            const linkDetails = document.createElement("p");
            linkDetails.innerText = `Checked URL: ${currentTabUrl}`;
            linkDetails.style.fontSize = "0.9em";
            linkDetailsContainer.appendChild(linkDetails);
        });
    });

    // Enhanced URL safety checks
    function checkIfLinkIsMalicious(url) {
        if (url.startsWith("http://")) return true; // Flag non-HTTPS sites

        const suspiciousKeywords = [
            "malware", "phishing", "untrusted", "attack", "suspicious", "free-gift",
            "login-confirm", "update-now", ".exe", ".xyz", "click-here", "verify-account",
            "reset-password", "free-vpn"
        ];
        return suspiciousKeywords.some(keyword => url.includes(keyword)) || url.length > 100;
    }
});
