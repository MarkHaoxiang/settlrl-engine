// Board-layer style helpers still used by the SVG renderers (Anchored,
// BoardView, Robber, …) where styling is computed/positioned per element. The
// rest of the app styles through styles/ui.module.css over the design tokens in
// index.css.

const FONT = "Georgia, serif";

// Bright board click-target highlight (ghost strokes, robber tiles): a literal,
// readable against the tile colours in either theme.
export const HIGHLIGHT = "#FCE38A";
// Its on-panel counterpart (the accent token).
export const ACCENT = "var(--accent)";

// The shared panel surface, for board-anchored popovers (Anchored).
export const panelStyle: React.CSSProperties = {
  borderRadius: 12,
  background: "var(--panel-bg)",
  border: "1px solid var(--panel-border)",
  color: "var(--text)",
  fontFamily: FONT,
  backdropFilter: "blur(2px)",
  userSelect: "none",
};
