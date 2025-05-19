document.addEventListener("DOMContentLoaded", () => {
    const selectedTextArea = document.getElementById("selectedText");
    const scanButton = document.getElementById("scanButton");
    const resultDiv = document.getElementById("result");
    const meter = document.getElementById("meter");
    const meterBar = document.getElementById("meterBar");
    const detailsDiv = document.getElementById("details");
    const linkResultDiv = document.getElementById("linkResult");
    const checkLinkButton = document.getElementById("checkLinkButton");
    const harassmentAdviceContainer = document.getElementById("harassmentAdviceContainer");
    const linkAdviceContainer = document.getElementById("linkAdviceContainer");
    const linkDetailsContainer = document.getElementById("linkDetailsContainer");

    const COLORS = { danger: "#cc163f", ok: "#28a745", warn: "#ffcc00" };

    // Pre-fill with the text selected on the page; the user can also type.
    chrome.runtime.sendMessage({ action: "getSelectedText" }, (response) => {
        selectedTextArea.value = response?.selectedText || "";
    });

    function showResult(div, text, color) {
        div.innerText = text;
        div.style.backgroundColor = color;
        div.style.color = color === COLORS.warn ? "#000" : "#fff";
    }

    function addLine(container, text, style = {}) {
        const p = document.createElement("p");
        p.innerText = text;
        Object.assign(p.style, style);
        container.appendChild(p);
    }

    scanButton.addEventListener("click", async () => {
        const text = selectedTextArea.value.trim();
        harassmentAdviceContainer.innerHTML = "";
        detailsDiv.innerHTML = "";
        meter.hidden = true;
        if (!text) {
            showResult(resultDiv, "Select or type some text to scan.", COLORS.warn);
            return;
        }

        scanButton.disabled = true;
        showResult(resultDiv, "Analyzing…", "#0c2547");
        try {
            const response = await fetch(`${SIGHTECTOR_API}/api/analyze`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || response.statusText);

            const pct = Math.round(data.score * 100);
            showResult(resultDiv,
                data.harassment ? `Harassment detected - ${data.severity} severity (${pct}%)`
                                : `No harassment detected (${pct}%)`,
                data.harassment ? COLORS.danger : COLORS.ok);
            meter.hidden = false;
            meterBar.style.width = `${pct}%`;
            meterBar.style.backgroundColor = data.harassment ? COLORS.danger : COLORS.ok;

            if (data.deobfuscated.length) {
                addLine(detailsDiv, "Disguised words: " +
                    data.deobfuscated.map((d) => `${d.original} → ${d.normalized}`).join(", "));
            }
            const words = data.highlights.length ? data.highlights.map((h) => h.word) : data.offensive_terms;
            if (data.harassment && words.length) {
                addLine(detailsDiv, "Key words: " + words.join(", "), { color: COLORS.danger });
            }

            if (data.harassment) {
                addLine(harassmentAdviceContainer, "This text contains harmful content. Consider reporting or blocking.",
                        { fontWeight: "bold", color: COLORS.danger });
                addLine(harassmentAdviceContainer, "Report this content to keep the community safe.", { color: "#d9534f" });
                addLine(harassmentAdviceContainer, "You can block the sender to prevent further interactions.", { color: "#d9534f" });
            }
        } catch (error) {
            showResult(resultDiv, "Cannot reach the Sightector server. Is it running? (python -m server)", COLORS.danger);
        } finally {
            scanButton.disabled = false;
        }
    });

    // Basic heuristic URL check for the active tab.
    checkLinkButton.addEventListener("click", () => {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            const currentTabUrl = tabs[0]?.url;
            if (!currentTabUrl) return;

            const isMalicious = checkIfLinkIsMalicious(currentTabUrl);
            showResult(linkResultDiv, isMalicious ? "Potentially Malicious Link." : "Safe Link.",
                       isMalicious ? COLORS.danger : COLORS.ok);

            linkAdviceContainer.innerHTML = "";
            linkDetailsContainer.innerHTML = "";
            if (isMalicious) {
                addLine(linkAdviceContainer, "For your safety, avoid sharing personal data on this page.", { fontWeight: "bold" });
                addLine(linkAdviceContainer, "If this link looks suspicious, do NOT click on it.", { color: "#d9534f" });
            }
            addLine(linkDetailsContainer, `Checked URL: ${currentTabUrl}`, { fontSize: "0.9em", wordBreak: "break-all" });
        });
    });

    function checkIfLinkIsMalicious(url) {
        if (url.startsWith("http://") && !/^http:\/\/(localhost|127\.0\.0\.1)/.test(url)) return true;
        const suspiciousKeywords = [
            "malware", "phishing", "untrusted", "attack", "suspicious", "free-gift",
            "login-confirm", "update-now", ".exe", ".xyz", "click-here", "verify-account",
            "reset-password", "free-vpn"
        ];
        return suspiciousKeywords.some((keyword) => url.includes(keyword)) || url.length > 100;
    }
});
