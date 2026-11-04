import React, { useState, useEffect, useCallback } from 'react';
import ContextualVectorPanel from './components/ContextualVectorPanel';
import AttentionHeatmapPanel from './components/AttentionHeatmapPanel';
import AttentionDistributionPanel from './components/AttentionDistributionPanel';
import { analyzeContext, checkHealth } from './services/api';

const DEFAULT_SENTENCE_A = "The animal didn't cross the road because it was tired.";
const DEFAULT_SENTENCE_B = "The animal didn't cross the road because it was raining.";
const DEFAULT_TARGET = "it";

// Lightweight frontend word-occurrence validation (word boundaries & case-insensitive)
function hasWordOccurrence(sentence, word) {
  const cleanWord = word.trim().replace(/^["']|["']$/g, '').toLowerCase();
  if (!cleanWord) return false;
  const sentLower = sentence.toLowerCase();
  let startPos = 0;
  while (true) {
    const idx = sentLower.indexOf(cleanWord, startPos);
    if (idx === -1) return false;
    const endIdx = idx + cleanWord.length;
    const beforeOk = idx === 0 || !/[a-z0-9]/i.test(sentLower[idx - 1]);
    const afterOk = endIdx === sentLower.length || !/[a-z0-9]/i.test(sentLower[endIdx]);
    if (beforeOk && afterOk) return true;
    startPos = idx + 1;
  }
}

export default function App() {
  const [sentenceA, setSentenceA] = useState(DEFAULT_SENTENCE_A);
  const [sentenceB, setSentenceB] = useState(DEFAULT_SENTENCE_B);
  const [targetWord, setTargetWord] = useState(DEFAULT_TARGET);

  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);

  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendOk, setBackendOk] = useState(null);

  // Health check on mount
  useEffect(() => {
    checkHealth()
      .then((res) => setBackendOk(res.status === 'ok'))
      .catch(() => setBackendOk(false));
  }, []);

  // Main Analyze action
  const handleAnalyze = useCallback(
    async (overrideLayer, overrideHead) => {
      const l = overrideLayer !== undefined ? overrideLayer : selectedLayer;
      const h = overrideHead !== undefined ? overrideHead : selectedHead;

      if (!sentenceA.trim() || !sentenceB.trim() || !targetWord.trim()) return;

      // Always clear previous target-specific results immediately on new analysis run
      setAnalysisData(null);
      setError(null);

      // Lightweight frontend validation
      const cleanTarget = targetWord.trim().replace(/^["']|["']$/g, '');
      const inA = hasWordOccurrence(sentenceA, cleanTarget);
      const inB = hasWordOccurrence(sentenceB, cleanTarget);

      if (!inA || !inB) {
        let msg;
        if (!inA && !inB) {
          msg = `Target word "${cleanTarget}" was not found in Sentence A or Sentence B. Please enter a word that appears in both sentences.`;
        } else if (!inA) {
          msg = `Target word "${cleanTarget}" was not found in Sentence A. Please enter a word that appears in both sentences.`;
        } else {
          msg = `Target word "${cleanTarget}" was not found in Sentence B. Please enter a word that appears in both sentences.`;
        }
        setError(msg);
        return;
      }

      setLoading(true);

      try {
        const res = await analyzeContext({
          sentenceA: sentenceA.trim(),
          sentenceB: sentenceB.trim(),
          targetWord: targetWord.trim(),
          layer: l,
          head: h,
        });

        if (!res.valid && res.valid !== undefined) {
          setAnalysisData(null);
          setError(res.message || `Target word "${cleanTarget}" was not found in the sentence.`);
        } else {
          setAnalysisData(res);
          setError(null);
        }
      } catch (err) {
        // Clear any previous target-specific result when a new invalid target is submitted
        setAnalysisData(null);
        if (err.detail && err.detail.type === 'TARGET_NOT_FOUND') {
          setError(err.detail.message || `Target word "${cleanTarget}" was not found in the sentence.`);
        } else {
          setError(err.message || 'Failed to analyze context with BERT.');
        }
      } finally {
        setLoading(false);
      }
    },
    [sentenceA, sentenceB, targetWord, selectedLayer, selectedHead]
  );

  // Auto-run on mount once backend is ready
  useEffect(() => {
    handleAnalyze(0, 0);
  }, []);

  // When layer or head changes and active data exists, update
  const handleLayerChange = (newLayer) => {
    setSelectedLayer(newLayer);
    if (analysisData) {
      handleAnalyze(newLayer, selectedHead);
    }
  };

  const handleHeadChange = (newHead) => {
    setSelectedHead(newHead);
    if (analysisData) {
      handleAnalyze(selectedLayer, newHead);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="app-header__badge">
          <span className="dot dot--online" />
          {backendOk === true ? 'BERT Engine Ready' : backendOk === false ? 'Backend Offline (Start Port 8000)' : 'Connecting…'}
        </div>
        <h1 className="app-header__title">
          Self-Attention: Understanding Context Inside BERT
        </h1>
        <p className="app-header__subtitle">
          How BERT uses multi-head self-attention to build different contextual representations for the same word across different sentences.
        </p>
      </header>

      {/* Two-Sentence Input Box */}
      <div className="card input-card" style={{ marginBottom: '2rem' }}>
        <div className="card__header">
          <span className="card__title card__title--blue">✏️ Input Sentences &amp; Target Word</span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Comparing the same word in two different contexts
          </span>
        </div>

        <div className="card__body">
          <div className="inputs-grid">
            {/* Sentence A */}
            <div className="input-group">
              <label className="input-label" htmlFor="sent-a">
                <span className="badge badge--blue">Sentence A</span>
              </label>
              <textarea
                id="sent-a"
                className="input-textarea"
                rows={2}
                value={sentenceA}
                onChange={(e) => {
                  setSentenceA(e.target.value);
                  if (error) setError(null);
                }}
                placeholder="Enter first sentence…"
              />
            </div>

            {/* Sentence B */}
            <div className="input-group">
              <label className="input-label" htmlFor="sent-b">
                <span className="badge badge--cyan">Sentence B</span>
              </label>
              <textarea
                id="sent-b"
                className="input-textarea"
                rows={2}
                value={sentenceB}
                onChange={(e) => {
                  setSentenceB(e.target.value);
                  if (error) setError(null);
                }}
                placeholder="Enter second sentence…"
              />
            </div>
          </div>

          {/* Target Word & Action Button */}
          <div className="action-row">
            <div className="target-word-group">
              <label htmlFor="target-input" className="input-label">Target Word to Compare:</label>
              <input
                id="target-input"
                type="text"
                className={`input-text ${error ? 'input-text--error' : ''}`}
                value={targetWord}
                onChange={(e) => {
                  setTargetWord(e.target.value);
                  if (error) setError(null);
                }}
                placeholder="e.g. it"
              />
              {/* User-facing validation message near target input */}
              {error && (
                <div
                  className="target-validation-message"
                  style={{
                    color: '#f87171',
                    fontSize: '0.82rem',
                    marginTop: '0.45rem',
                    fontWeight: 500,
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '0.35rem',
                    lineHeight: 1.4,
                  }}
                >
                  <span style={{ fontSize: '0.95rem' }}>⚠</span>
                  <span>{error}</span>
                </div>
              )}
            </div>

            <button
              className="btn btn--primary btn--lg"
              onClick={() => handleAnalyze()}
              disabled={loading || !sentenceA.trim() || !sentenceB.trim() || !targetWord.trim()}
            >
              {loading ? (
                <>
                  <span className="spinner" /> Running BERT…
                </>
              ) : (
                '🔍 ANALYZE BOTH SENTENCES'
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Main Analysis Sections — Only shown when a valid target analysis exists */}
      {analysisData && analysisData.valid !== false && (
        <main>
          {/* Part A: Contextual Representations & Cosine Similarity */}
          <ContextualVectorPanel data={analysisData} />

          {/* Part B: Attention Heatmaps */}
          <AttentionHeatmapPanel
            data={analysisData}
            selectedLayer={selectedLayer}
            selectedHead={selectedHead}
            onLayerChange={handleLayerChange}
            onHeadChange={handleHeadChange}
          />

          {/* Part C: Attention Distribution Comparison */}
          <AttentionDistributionPanel
            data={analysisData}
            layer={selectedLayer}
            head={selectedHead}
          />
        </main>
      )}

      {/* Footer */}
      <footer className="app-footer">
        <p><strong>Case Study:</strong> Self-Attention: Understanding Context Inside BERT</p>
        <p style={{ marginTop: '0.35rem', color: 'var(--text-muted)' }}>
          Academic References: Vaswani et al. (2017) <em>Attention Is All You Need</em> • Devlin et al. (2019) <em>BERT: Pre-training of Deep Bidirectional Transformers</em>
        </p>
      </footer>
    </div>
  );
}
