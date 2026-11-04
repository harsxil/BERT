/**
 * API service — connects frontend to FastAPI backend.
 */

const API_BASE = 'http://localhost:8000';

async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    let errMsg = `HTTP ${res.status}`;
    if (typeof body.detail === 'string') {
      errMsg = body.detail;
    } else if (body.detail && typeof body.detail === 'object') {
      errMsg = body.detail.message || JSON.stringify(body.detail);
    }
    const err = new Error(errMsg);
    err.status = res.status;
    err.detail = body.detail;
    throw err;
  }
  return res.json();
}

// Liveness check
export function checkHealth() {
  return request('/api/health');
}

// Dual-sentence contextual & self-attention analysis
export function analyzeContext({ sentenceA, sentenceB, targetWord, layer = 0, head = 0 }) {
  return request('/api/analyze-context', {
    method: 'POST',
    body: JSON.stringify({
      sentence_a: sentenceA,
      sentence_b: sentenceB,
      target_word: targetWord,
      layer,
      head,
    }),
  });
}
