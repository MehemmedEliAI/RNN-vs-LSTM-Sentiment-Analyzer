async function analyze() {
    const text = document.getElementById("reviewInput").value.trim();

    if (!text) {
        alert("Please enter a review text first.");
        return;
    }

    const analyzeButton = document.getElementById("analyzeButton");
    if (analyzeButton) {
        analyzeButton.disabled = true;
        analyzeButton.innerText = "Analyzing...";
    }

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text })
        });

        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }

        const data = await response.json();

        const rnnScoreEl = document.getElementById("rnnScore");
        const rnnLabelEl = document.getElementById("rnnLabel");
        const lstmScoreEl = document.getElementById("lstmScore");
        const lstmLabelEl = document.getElementById("lstmLabel");
        const rnnBarEl = document.getElementById("rnnBar");
        const lstmBarEl = document.getElementById("lstmBar");

        const rnnScore = typeof data.rnn_result?.score === "number" ? data.rnn_result.score : null;
        const lstmScore = typeof data.lstm_result?.score === "number" ? data.lstm_result.score : null;
        const rnnLabel = data.rnn_result?.label ?? "-";
        const lstmLabel = data.lstm_result?.label ?? "-";

        rnnScoreEl.innerText = rnnScore !== null ? rnnScore.toFixed(4) : "-";
        lstmScoreEl.innerText = lstmScore !== null ? lstmScore.toFixed(4) : "-";

        if (rnnBarEl) {
            const width = rnnScore !== null ? Math.round(rnnScore * 100) : 0;
            rnnBarEl.style.width = width + "%";
        }
        if (lstmBarEl) {
            const width = lstmScore !== null ? Math.round(lstmScore * 100) : 0;
            lstmBarEl.style.width = width + "%";
        }

        rnnLabelEl.innerText = rnnLabel;
        lstmLabelEl.innerText = lstmLabel;

        [rnnLabelEl, lstmLabelEl].forEach((el, idx) => {
            el.classList.remove("pill-neutral", "pill-positive", "pill-negative");
            const labelValue = idx === 0 ? rnnLabel : lstmLabel;
            if (labelValue === "Positive") {
                el.classList.add("pill-positive");
            } else if (labelValue === "Negative") {
                el.classList.add("pill-negative");
            } else {
                el.classList.add("pill-neutral");
            }
        });
    } catch (err) {
        console.error(err);
        alert("Failed to analyze sentiment. Please make sure the backend server is running.");
    } finally {
        if (analyzeButton) {
            analyzeButton.disabled = false;
            analyzeButton.innerText = "Analyze";
        }
    }
}