import React, { useState } from 'react';

export default function ContextualVectorPanel({ data }) {
  const [showFullA, setShowFullA] = useState(false);
  const [showFullB, setShowFullB] = useState(false);

  if (!data || data.valid === false || data.cosine_similarity === null || data.cosine_similarity === undefined) return null;

  const { sentence_a, sentence_b, target_word, cosine_similarity, hidden_dim, hidden_layer } = data;

  return (
    <section className="card fade-in" style={{ marginBottom: '2rem' }}>
      <div className="card__header">
        <span className="card__title card__title--purple">
          🧪 Part A: Contextual Representations &amp; Cosine Similarity
        </span>
        <span className="badge badge--purple">Layer {hidden_layer} Final Hidden State ({hidden_dim}-D)</span>
      </div>

      <div className="card__body">
        {/* Core Metric Comparison Banner */}
        <div className="similarity-banner">
          <div className="similarity-card similarity-card--contextual">
            <div className="similarity-card__label">Cosine Similarity</div>
            <div className="similarity-card__value">{cosine_similarity.toFixed(4)}</div>
            <div className="similarity-card__desc">
              Representation of "{target_word}" in Sentence A vs. Sentence B
            </div>
            <div className="similarity-progress">
              <div
                className="similarity-progress__bar similarity-progress__bar--contextual"
                style={{ width: `${Math.max(10, cosine_similarity * 100)}%` }}
              />
            </div>
            <p className="similarity-card__note">
              ✓ Context-dependent: Surrounding words produce distinct internal representations.
            </p>
          </div>
        </div>

        {/* Vector Inspection Grid */}
        <div className="vector-grid" style={{ marginTop: '1.5rem' }}>
          {/* Sentence A Vector Card */}
          <div className="vector-card">
            <div className="vector-card__header">
              <span className="badge badge--blue">Sentence A</span>
              <span className="vector-card__word">Target: "{target_word}"</span>
            </div>
            <p className="vector-card__context">"{sentence_a.sentence}"</p>
            <div className="vector-preview">
              <span className="vector-preview__label">First 10 of {sentence_a.hidden_dim} dimensions:</span>
              <div className="vector-chips">
                {sentence_a.hidden_state_10.map((val, idx) => (
                  <span key={idx} className="chip">
                    <span className="chip__idx">d{idx}:</span> {val > 0 ? `+${val.toFixed(4)}` : val.toFixed(4)}
                  </span>
                ))}
              </div>
            </div>
            <button
              className="btn btn--secondary btn--sm"
              onClick={() => setShowFullA(!showFullA)}
              style={{ marginTop: '0.8rem' }}
            >
              {showFullA ? '▲ Hide Full Vector' : `▼ View Full ${sentence_a.hidden_dim}-D Vector`}
            </button>
            {showFullA && (
              <div className="vector-full-dump">
                {sentence_a.hidden_state_full.map((val, idx) => (
                  <span key={idx} className="vector-full-cell">
                    [{idx}]: {val.toFixed(4)}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Sentence B Vector Card */}
          <div className="vector-card">
            <div className="vector-card__header">
              <span className="badge badge--cyan">Sentence B</span>
              <span className="vector-card__word">Target: "{target_word}"</span>
            </div>
            <p className="vector-card__context">"{sentence_b.sentence}"</p>
            <div className="vector-preview">
              <span className="vector-preview__label">First 10 of {sentence_b.hidden_dim} dimensions:</span>
              <div className="vector-chips">
                {sentence_b.hidden_state_10.map((val, idx) => (
                  <span key={idx} className="chip">
                    <span className="chip__idx">d{idx}:</span> {val > 0 ? `+${val.toFixed(4)}` : val.toFixed(4)}
                  </span>
                ))}
              </div>
            </div>
            <button
              className="btn btn--secondary btn--sm"
              onClick={() => setShowFullB(!showFullB)}
              style={{ marginTop: '0.8rem' }}
            >
              {showFullB ? '▲ Hide Full Vector' : `▼ View Full ${sentence_b.hidden_dim}-D Vector`}
            </button>
            {showFullB && (
              <div className="vector-full-dump">
                {sentence_b.hidden_state_full.map((val, idx) => (
                  <span key={idx} className="vector-full-cell">
                    [{idx}]: {val.toFixed(4)}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
