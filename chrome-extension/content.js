console.log("Content script loaded on:", window.location.href);

// Track the last sent data to prevent duplicates
let lastDataSent = null;

const observer = new MutationObserver(() => {
  const winningNumberElement = document.querySelector('.last-draw .right .result.color-game-result .ball');
  const drawTimeElement = document.querySelector('.last-draw .right .ft .drawtime');

  if (winningNumberElement && drawTimeElement) {
    const result = parseInt(winningNumberElement.textContent.trim());  // Convert to integer
    const drawTimeText = drawTimeElement.textContent.trim();
    console.log("Draw time text extracted:", drawTimeText);

    // Extract the actual date/time part and append current year
    const dateTimeMatch = drawTimeText.match(/(\d{2}\/\d{2} \d{2}:\d{2}:\d{2})/);
    if (!dateTimeMatch) {
      console.error("Invalid draw time format:", drawTimeText);
      return;
    }

    const dateTime = dateTimeMatch[1];
    console.log("Extracted date and time:", dateTime);

    // Append the current year to the date/time
    const currentYear = new Date().getFullYear();
    const formattedDateTime = `${dateTime}/${currentYear}`;
    console.log("Formatted date and time with year:", formattedDateTime);

    // Remove the unnecessary year after the time (after /) and reorder date to YYYY-MM-DD
    const formattedDate = formattedDateTime
      .replace("/", "-")  // Replace '/' with '-'
      .replace(/^(\d{2})-(\d{2})/, `${currentYear}-$2-$1`)  // Reorder to YYYY-MM-DD
      .replace(/\/\d{4}$/, '');  // Remove the trailing year part after time

    console.log("Reformatted date for parsing:", formattedDate);

    // Convert to timestamp
    const drawTime = new Date(formattedDate).getTime();

    // Validate draw time
    if (isNaN(drawTime)) {
      console.error("Invalid draw time after formatting:", formattedDate);
      return;
    }

    const data = {
      result,    // Use shorthand for 'result: result'
      drawTime   // Use shorthand for 'drawTime: drawTime'
    };

    console.log("Extracted data:", data);

    // Prevent sending duplicate data
    if (!lastDataSent || data.result !== lastDataSent.result || data.drawTime !== lastDataSent.drawTime) {
      lastDataSent = data;  // Update last sent data
      sendToApi(data);  // Send to backend
    } else {
      console.log("Duplicate data detected, not sending to API.");
    }
  }
});

// Start observing for DOM changes, to detect when the relevant data is updated
observer.observe(document.body, { childList: true, subtree: true });

// Function to send data directly to the API
function sendToApi(data) {
  console.log("Sending data to API:", data);

  // Make a POST request to the backend API
  fetch("http://127.0.0.1:5000/api/data/save", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data)  // Sending the data directly (no need to re-map the keys)
  })
  .then((res) => {
    console.log("Response from backend:", res);

    if (!res.ok) {
      // Handle response errors, e.g. 4xx or 5xx responses
      throw new Error(`Backend returned error: ${res.statusText}`);
    }

    return res.json();  // Parse the JSON response from the backend
  })
  .then((response) => {
    console.log("Data sent to backend successfully:", response);
  })
  .catch((err) => {
    console.error("Error sending data to backend:", err);
  });
}