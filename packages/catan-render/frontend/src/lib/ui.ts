// Shared look-and-feel: the standard panel, button, and overlay message,
// reused by every view. Spread these into a style object and override what
// differs. Colours resolve through the theme variables in index.css
// (body[data-theme]), so everything here follows the light/dark toggle.

export const FONT = "Georgia, serif";

export const panelStyle: React.CSSProperties = {
  borderRadius: 12,
  background: "var(--panel-bg)",
  border: "1px solid var(--panel-border)",
  color: "var(--text)",
  fontFamily: FONT,
  backdropFilter: "blur(2px)",
  userSelect: "none",
};

export const buttonStyle: React.CSSProperties = {
  background: "var(--button-bg)",
  border: "1px solid var(--button-border)",
  color: "var(--text)",
  borderRadius: 8,
  padding: "9px 16px",
  fontSize: 14,
  fontFamily: FONT,
  cursor: "pointer",
};

// Highlight for board click targets (ghost strokes, robber tiles): bright
// against the tile colours in either theme, so it stays a literal.
export const HIGHLIGHT = "#FCE38A";
// Its readable counterpart on panels: selected buttons, glowing chips, the
// winner banner.
export const ACCENT = "var(--accent)";
export const ACCENT_GLOW = "0 0 8px 2px var(--accent-glow)";
export const selectedStyle: React.CSSProperties = {
  background: "var(--selected-bg)",
  borderColor: "var(--accent)",
};

// Links on panels.
export const LINK = "var(--link)";

export const overlayMsgStyle: React.CSSProperties = {
  color: "var(--text)",
  padding: 24,
  fontFamily: FONT,
};

// Hairline separators inside panels.
export const DIVIDER = "var(--divider)";
