// Shared look-and-feel: the translucent dark "glass" panel, the standard
// button, and the plain overlay message, reused by every view. Spread these
// into a style object and override what differs.

export const FONT = "Georgia, serif";

export const panelStyle: React.CSSProperties = {
  borderRadius: 12,
  background: "rgba(12, 28, 46, 0.82)",
  border: "1px solid rgba(255,255,255,0.15)",
  color: "#F2EFE6",
  fontFamily: FONT,
  backdropFilter: "blur(2px)",
  userSelect: "none",
};

export const buttonStyle: React.CSSProperties = {
  background: "rgba(255,255,255,0.08)",
  border: "1px solid rgba(255,255,255,0.2)",
  color: "#F2EFE6",
  borderRadius: 8,
  padding: "9px 16px",
  fontSize: 14,
  fontFamily: FONT,
  cursor: "pointer",
};

// Highlight for the armed / selected button and board click targets.
export const HIGHLIGHT = "#FCE38A";
export const selectedStyle: React.CSSProperties = {
  background: "rgba(252, 227, 138, 0.25)",
  borderColor: HIGHLIGHT,
};

export const overlayMsgStyle: React.CSSProperties = {
  color: "#fff",
  padding: 24,
  fontFamily: FONT,
};
