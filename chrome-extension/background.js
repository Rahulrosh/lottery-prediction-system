chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log("Listener activated", message);

  if (message.action === "sendApiData") {
    console.log("Received data in background:", message.data);

    // Check if the data is valid before sending
    if (!message.data || !message.data.winningNumber || !message.data.drawTime) {
      console.error("Invalid data:", message.data);
      sendResponse({ status: "error", message: "Invalid data" });
      return;
    }

    fetch("http://localhost:8080/api/data/save", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        drawTime: message.data.drawTime,  // Send draw time as a timestamp
        result: message.data.winningNumber  // Send the winning number
      })
    })
    .then((res) => {
      console.log("Response from backend:", res);

      if (!res.ok) {
        // Handle response errors, e.g. 4xx or 5xx responses
        throw new Error(`Backend returned error: ${res.statusText}`);
      }

      return res.json();
    })
    .then((response) => {
      console.log("Data sent to backend successfully, response:", response);
      sendResponse({ status: "success" });
    })
    .catch((err) => {
      console.error("Error sending data to backend:", err);
      sendResponse({ status: "error", message: err.message });
    });

    return true;  // Indicate async response
  }
});