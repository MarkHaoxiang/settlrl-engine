import { panelStyle } from "../lib/ui";

interface Props {
  // Anchor point in the board container's coordinate space (BoardTargetPoint).
  x: number;
  y: number;
  onClose: () => void;
  children: React.ReactNode;
}

// A small chooser anchored to a clicked board element. The full-size backdrop
// closes it on any outside press, so a stray click can't fire a second action;
// it also blocks pan/zoom while open (a wheel just closes it).
export default function BoardPopover({ x, y, onClose, children }: Props) {
  // Near the top edge, open downward instead of clipping off-screen.
  const below = y < 150;
  return (
    <div
      style={{ position: "absolute", inset: 0, zIndex: 20 }}
      onPointerDown={onClose}
      onWheel={onClose}
    >
      <div
        onPointerDown={(e) => e.stopPropagation()}
        style={{
          ...panelStyle,
          position: "absolute",
          left: x,
          top: y,
          transform: below ? "translate(-50%, 14px)" : "translate(-50%, calc(-100% - 14px))",
          display: "flex",
          flexDirection: "column",
          gap: 6,
          padding: 10,
          boxShadow: "0 6px 24px rgba(0,0,0,0.45)",
        }}
      >
        {children}
      </div>
    </div>
  );
}
