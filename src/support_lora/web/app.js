const $ = (id) => document.getElementById(id);
$('compare').addEventListener('submit', async (event) => {
  event.preventDefault(); const button = event.target.querySelector('button'); button.disabled = true; $('results').replaceChildren();
  const variants = $('variant').value === 'all' ? ['baseline', 'base', 'lora'] : [$('variant').value];
  try {
    // Sequential requests avoid overlapping GPU generations on a small workstation.
    for (const variant of variants) {
      const card = document.createElement('section'); const heading = document.createElement('h2'); heading.textContent = variant;
      const output = document.createElement('pre'); output.dir = 'auto'; output.textContent = 'Analyzing…'; card.append(heading, output); $('results').append(card);
      try {
        const response = await fetch('/predict', {method: 'POST', headers: {'Content-Type': 'application/json', 'Authorization': `Bearer ${$('token').value}`}, body: JSON.stringify({text: $('text').value, variant})});
        const data = await response.json(); if (!response.ok) throw new Error(JSON.stringify(data.detail)); output.textContent = JSON.stringify(data, null, 2);
      } catch (error) { output.textContent = error.message; }
    }
  } finally { button.disabled = false; }
});
