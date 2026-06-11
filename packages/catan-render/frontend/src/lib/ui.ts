// Shared look-and-feel: a warm "paper" panel on a sunlit table, the standard
// button, and the plain overlay message, reused by every view. Spread these
// into a style object and override what differs.

export const FONT = "Georgia, serif";

export const panelStyle: React.CSSProperties = {
  borderRadius: 12,
  background: "rgba(253, 248, 235, 0.92)",
  border: "1px solid rgba(90, 70, 40, 0.3)",
  color: "#2B2418",
  fontFamily: FONT,
  backdropFilter: "blur(2px)",
  userSelect: "none",
};

export const buttonStyle: React.CSSProperties = {
  background: "rgba(90, 70, 40, 0.08)",
  border: "1px solid rgba(90, 70, 40, 0.4)",
  color: "#2B2418",
  borderRadius: 8,
  padding: "9px 16px",
  fontSize: 14,
  fontFamily: FONT,
  cursor: "pointer",
};

// Highlight for board click targets (ghost strokes, robber tiles): bright
// against the tile colours.
export const HIGHLIGHT = "#FCE38A";
// Its readable counterpart on the light panels: selected buttons, glowing
// chips, the winner banner.
export const ACCENT = "#A87B16";
export const selectedStyle: React.CSSProperties = {
  background: "rgba(201, 154, 46, 0.25)",
  borderColor: ACCENT,
};

// Links on the light panels.
export const LINK = "#2C5F9E";

export const overlayMsgStyle: React.CSSProperties = {
  color: "#2B2418",
  padding: 24,
  fontFamily: FONT,
};

// Hairline separators inside panels.
export const DIVIDER = "rgba(90, 70, 40, 0.25)";
