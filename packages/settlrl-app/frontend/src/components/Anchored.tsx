import { useEffect } from "react";
import {
  FloatingOverlay,
  FloatingPortal,
  autoUpdate,
  flip,
  offset,
  shift,
  useFloating,
} from "@floating-ui/react";
import { panelStyle } from "../lib/ui";

// A small panel anchored to a point (viewport coordinates) by the board.
// Floating UI keeps it on-screen — it opens above the point, flipping below and
// shifting sideways near an edge rather than clipping. A transparent overlay
// closes it on any outside press or a wheel (a pan/zoom gesture), so a stray
// click can't fire a second action.
export default function Anchored({
  x,
  y,
  onClose,
  children,
}: {
  x: number;
  y: number;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const { refs, floatingStyles } = useFloating({
    placement: "top",
    strategy: "fixed",
    middleware: [offset(14), flip({ padding: 8 }), shift({ padding: 8 })],
    whileElementsMounted: autoUpdate,
  });
  // A zero-size virtual reference at the anchor point.
  useEffect(() => {
    refs.setPositionReference({
      getBoundingClientRect: () =>
        ({ x, y, top: y, left: x, right: x, bottom: y, width: 0, height: 0 }) as DOMRect,
    });
  }, [x, y, refs]);

  return (
    <FloatingPortal>
      <FloatingOverlay lockScroll={false} style={{ zIndex: 20 }} onPointerDown={onClose} onWheel={onClose}>
        <div
          ref={refs.setFloating}
          onPointerDown={(e) => e.stopPropagation()}
          style={{
            ...panelStyle,
            ...floatingStyles,
            display: "flex",
            flexDirection: "column",
            gap: 6,
            padding: 10,
            boxShadow: "0 6px 24px rgba(0,0,0,0.45)",
          }}
        >
          {children}
        </div>
      </FloatingOverlay>
    </FloatingPortal>
  );
}
