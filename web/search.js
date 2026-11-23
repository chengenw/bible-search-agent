const form = document.querySelector("#search-form");
const question = document.querySelector("#question");
const translation = document.querySelector("#version");
const submitButton = document.querySelector("#submit");
const statusMessage = document.querySelector("#status");
const results = document.querySelector("#results");
const answer = document.querySelector("#answer");
const sources = document.querySelector("#sources");
const examples = document.querySelectorAll(".example");

for (const example of examples) {
  example.addEventListener("click", () => {
    question.value = example.textContent;
    question.focus();
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (submitButton.disabled) return;

  const query = question.value.trim();
  results.hidden = true;
  statusMessage.classList.toggle("error", !query);
  if (!query) {
    statusMessage.textContent = "Enter a question to search.";
    question.focus();
    return;
  }

  submitButton.disabled = true;
  question.readOnly = true;
  const version = translation.value;
  translation.disabled = true;
  for (const example of examples) example.disabled = true;
  statusMessage.textContent = `Searching ${version}. The first search in a translation may take longer.`;

  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: query, version }),
    });
    const data = await response.json();
    if (!response.ok) {
      statusMessage.textContent = data.error || "Search failed. Please try again.";
      statusMessage.classList.add("error");
      return;
    }

    answer.textContent = data.answer.replaceAll("**", "");
    sources.replaceChildren();
    const verses = Object.entries(data.sources);
    for (const [reference, text] of verses) {
      const row = document.createElement("li");
      const title = document.createElement("strong");
      const verse = document.createElement("p");
      title.textContent = reference;
      verse.textContent = text;
      row.append(title, verse);
      sources.append(row);
    }
    sources.parentElement.hidden = verses.length === 0;
    results.hidden = false;
    statusMessage.textContent = `Search complete for ${version}.`;
  } catch {
    statusMessage.textContent = "Could not complete the search. Please try again.";
    statusMessage.classList.add("error");
  } finally {
    submitButton.disabled = false;
    question.readOnly = false;
    translation.disabled = false;
    for (const example of examples) example.disabled = false;
  }
});
