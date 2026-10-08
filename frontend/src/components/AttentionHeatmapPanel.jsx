import React, { useState } from 'react';

function getHeatmapColor(value) {
  // Value from 0.0 to 1.0 -> dark blue/slate to bright amber/gold
  const v = Math.min(1, Math.max(0, value));
  const r = Math.round(18 + v * 225);
  const g = Math.round(28 + v * 155);
  const b = Math.round(52 + (v > 0.4 ? -20 : 60));
  const a = 0.2 + v * 0.8;
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

export default function AttentionHeatmapPanel({
  data,
  selectedLayer,
  selectedHead,
  onLayerChange,
  onHeadChange,
}) {
  const [tooltip, setTooltip] = useState(null);

  if (!data) return null;

  const { sentence_a, sentence_b, target_word, num_layers, num_heads } = data;

  const renderHeatmap = (sentData, badgeClass, label) => {
    const { tokens, attention_matrix, target_token_indices } = sentData;

    return (
      <div className="heatmap-box">
        <div className="heatmap-box__header">
          <span className={`badge ${badgeClass}`}>{label}</span>
          <span className="heatmap-box__tokens-count">{tokens.length} WordPiece Tokens</span>
        </div>

        {/* Tokens list */}
        <div className="tokens-strip">
          {tokens.map((tok, i) => (
            <span
              key={i}
              className={`token ${target_token_indices.includes(i) ? 'token--selected' : ''} ${
                tok === '[CLS]' || tok === '[SEP]' ? 'token--special' : ''
              }`}
            >
              {tok}
            </span>
          ))}
        </div>

        {/* Matrix table */}
        <div className="heatmap-scroll">
          <table className="heatmap-table">
            <thead>
              <tr>
                <th></th>
                {tokens.map((tok, ci) => (
                  <th
                    key={ci}
                    className={`heatmap-th-col ${target_token_indices.includes(ci) ? 'heatmap-th-col--target' : ''}`}
                  >
                    <span>{tok}</span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {attention_matrix.map((row, ri) => {
                const isTargetRow = target_token_indices.includes(ri);
                return (
                  <tr key={ri} className={isTargetRow ? 'heatmap-tr--target' : ''}>
                    <th className={`heatmap-th-row ${isTargetRow ? 'heatmap-th-row--target' : ''}`}>
                      {tokens[ri]}
                    </th>
                    {row.map((val, ci) => (
                      <td
                        key={ci}
                        style={{ backgroundColor: getHeatmapColor(val) }}
                        className={isTargetRow ? 'heatmap-td--target-row' : ''}
                        onMouseEnter={(e) => {
                          setTooltip({
                            x: e.clientX + 10,
                            y: e.clientY - 35,
                            source: tokens[ri],
                            target: tokens[ci],
                            val: val.toFixed(4),
                            isTarget: isTargetRow,
                          });
                        }}
                        onMouseLeave={() => setTooltip(null)}
                      />
                    ))}
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
      <div className="card__header" style={{ flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <span className="card__title card__title--blue">
            🔍 Part B: Look Inside BERT — Real Attention Heatmaps
          </span>
          <p style={{ margin: '0.2rem 0 0', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Inspecting exact attention tensors <code>outputs.attentions[layer][head]</code> across both contexts
          </p>
        </div>

        {/* Controls */}
        <div className="layer-head-controls">
          <div className="control-group">
            <label htmlFor="layer-select">Layer (0–{num_layers - 1}):</label>
            <select
              id="layer-select"
              className="select-control"
              value={selectedLayer}
              onChange={(e) => onLayerChange(Number(e.target.value))}
            >
              {Array.from({ length: num_layers }, (_, i) => (
                <option key={i} value={i}>
                  Layer {i}
                </option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <label htmlFor="head-select">Attention Head (0–{num_heads - 1}):</label>
            <select
              id="head-select"
              className="select-control"
              value={selectedHead}
              onChange={(e) => onHeadChange(Number(e.target.value))}
            >
              {Array.from({ length: num_heads }, (_, i) => (
                <option key={i} value={i}>
                  Head {i}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="card__body">
        <div className="legend-strip">
          <span className="legend-strip__label">Highlighted query row: <strong>"{target_word}"</strong></span>
          <div className="legend-scale">
            <span>0.0 (low interaction)</span>
            <div className="legend-scale__gradient" />
            <span>1.0 (high interaction)</span>
          </div>
        </div>

        <div className="dual-heatmaps-grid" style={{ marginTop: '1rem' }}>
          {renderHeatmap(sentence_a, 'badge--blue', 'Sentence A Context')}
          {renderHeatmap(sentence_b, 'badge--cyan', 'Sentence B Context')}
        </div>

        {/* Floating tooltip */}
        {tooltip && (
          <div
            className="heatmap-tooltip"
            style={{ left: `${tooltip.x}px`, top: `${tooltip.y}px` }}
          >
            <strong>{tooltip.source}</strong> → <strong>{tooltip.target}</strong>
            <div className="heatmap-tooltip__val">Attention: {tooltip.val}</div>
            {tooltip.isTarget && <span className="heatmap-tooltip__tag">Selected Word Query</span>}
          </div>
        )}
      </div>
    </section>
  );
}
