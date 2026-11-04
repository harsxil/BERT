import React from 'react';

export default function AttentionDistributionPanel({ data, layer, head }) {
  if (!data || data.valid === false) return null;

  const { sentence_a, sentence_b, target_word } = data;

  const renderDistributionList = (sentData, badgeClass, label) => {
    if (!sentData || sentData.valid === false || !Array.isArray(sentData.target_token_indices) || sentData.target_token_indices.length === 0 || !Array.isArray(sentData.attention_weights)) {
      return null;
    }
    const { tokens, attention_weights, target_token_indices, attention_sum, attention_std, is_uniform } = sentData;

    // Find maximum weight to scale bars proportionally
    const maxWeight = Math.max(...attention_weights, 0.001);

    return (
      <div className="dist-card">
        <div className="dist-card__header">
          <span className={`badge ${badgeClass}`}>{label}</span>
          <span className="dist-card__word">Query: "{target_word}"</span>
        </div>

        <p className="dist-card__sentence">"{sentData.sentence}"</p>

        {/* Stats banner */}
        <div className="dist-stats">
          <span className="dist-stat">Sum: <strong>{attention_sum.toFixed(4)}</strong></span>
          <span className="dist-stat">Std: <strong>{attention_std.toFixed(4)}</strong></span>
          <span className="dist-stat">
            Uniform: <strong>{is_uniform ? 'Yes ⚠' : 'No (Non-uniform ✓)'}</strong>
          </span>
        </div>

        {/* Token attention rows */}
        <div className="dist-table-wrap">
          <table className="dist-table">
            <thead>
              <tr>
                <th>Token</th>
                <th>Attention Weight</th>
                <th style={{ textAlign: 'right' }}>Score</th>
              </tr>
            </thead>
            <tbody>
              {tokens.map((tok, i) => {
                const score = attention_weights[i];
                const isTarget = target_token_indices.includes(i);
                const percent = (score * 100).toFixed(1);
                const barWidth = Math.round((score / maxWeight) * 100);

                return (
                  <tr key={i} className={isTarget ? 'dist-row--target' : ''}>
                    <td className="dist-td-token">
                      <span className={`token ${isTarget ? 'token--selected' : ''}`}>{tok}</span>
                    </td>
                    <td className="dist-td-bar">
                      <div className="dist-bar-track">
                        <div
                          className="dist-bar-fill"
                          style={{
                            width: `${barWidth}%`,
                            backgroundColor: isTarget ? 'var(--accent-purple)' : 'var(--accent-blue)',
                          }}
                        />
                      </div>
                    </td>
                    <td className="dist-td-score">
                      <span className="score-val">{score.toFixed(4)}</span>
                      <span className="score-pct">({percent}%)</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  return (
    <section className="card fade-in" style={{ marginBottom: '2rem' }}>
      <div className="card__header">
        <span className="card__title card__title--amber">
          📊 Part C: Attention Distribution for "{target_word}"
        </span>
        <span className="badge badge--amber">Layer {layer}, Head {head}</span>
      </div>

      <div className="card__body">
        <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '1.25rem' }}>
          Comparing the exact attention weights assigned by BERT when the query token is{' '}
          <strong>"{target_word}"</strong>. Notice how surrounding tokens receive differing weights across
          both contexts.
        </p>

        <div className="dual-dist-grid">
          {renderDistributionList(sentence_a, 'badge--blue', 'Sentence A Context')}
          {renderDistributionList(sentence_b, 'badge--cyan', 'Sentence B Context')}
        </div>
      </div>
    </section>
  );
}
