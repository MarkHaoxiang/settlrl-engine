import { useEffect, useRef, useState } from "react";
import { useGesture } from "@use-gesture/react";

const MIN_ZOOM = 0.3;
const MAX_ZOOM = 3;
const PAN_STEP = 60;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

// Pan / zoom / rotate for the table, driven by mouse drag, wheel, pinch, and
// keyboard (arrows pan, +/- zoom, [ ] spin a quarter turn, 0 re-fits). The
// view opens fitted to the scene. Pointer/wheel/pinch recognition is delegated
// to @use-gesture (cross-device deltas, the click-vs-drag threshold, pointer
// capture, and the non-passive wheel listener); the state, rotation, keyboard,
// and fit-to-scene are ours.
//
// Wiring: put `containerRef` on the viewport element (gestures bind to it),
// `sceneTransform` on a layer inside it, and `rotationTransform` on a second
// layer inside that — rotation gets its own (transitioned) layer so spinning
// animates without dragging on pan/zoom updates.
//
// `faceAngle` is the rotation that orients the table toward the viewer's seat
// (0 = the default bottom-facing view). The view opens at it, manual spins are
// relative to it, and reset / re-facing return to it.
export function useTableViewport(sceneW: number, sceneH: number, faceAngle = 0) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  // Cumulative degrees, so each quarter turn animates the short way round
  // instead of unwinding.
  const [rotation, setRotation] = useState(faceAngle);

  // Snap rotation to faceAngle (keeping the accumulated full turns, so it spins
  // the short way). Used by reset and when the viewer's seat resolves.
  const reface = (deg: number) => (r: number) => deg + Math.round((r - deg) / 360) * 360;

  // Re-face when the viewer's seat resolves (e.g. after an auto-join): spin to
  // put their seat at the bottom. Manual spins survive — they don't change
  // faceAngle, so this effect doesn't re-run.
  useEffect(() => {
    setRotation(reface(faceAngle));
  }, [faceAngle]);
  // Latest zoom, so pinch can seed its scale from the current value.
  const zoomRef = useRef(zoom);
  zoomRef.current = zoom;

  useGesture(
    {
      // memo holds the pan at gesture start, so movement (screen px) is added
      // to a fixed base rather than a value that shifts as we re-render.
      onDrag: ({ movement: [mx, my], first, memo }) => {
        const base = first || !memo ? [pan.x, pan.y] : memo;
        setPan({ x: base[0] + mx, y: base[1] + my });
        return base;
      },
      // Trackpads report small deltas, mice large ones; exponential scaling
      // keeps the zoom feel consistent across both.
      onWheel: ({ delta: [, dy], event }) => {
        event.preventDefault();
        setZoom((z) => clamp(z * Math.exp(-dy * 0.0015), MIN_ZOOM, MAX_ZOOM));
      },
      onPinch: ({ offset: [scale] }) => {
        setZoom(clamp(scale, MIN_ZOOM, MAX_ZOOM));
      },
    },
    {
      target: containerRef,
      eventOptions: { passive: false },
      // filterTaps + a small threshold keep a click from registering as a pan,
      // so taps still reach the board elements underneath.
      drag: { filterTaps: true, threshold: 4, pointer: { buttons: 1 } },
      pinch: {
        scaleBounds: { min: MIN_ZOOM, max: MAX_ZOOM },
        from: () => [zoomRef.current, 0],
      },
    }
  );

  const fitZoom = () => {
    const el = containerRef.current;
    if (!el) return 1;
    return clamp(Math.min(el.clientWidth / sceneW, el.clientHeight / sceneH, 1) * 0.98, MIN_ZOOM, 1);
  };

  // Open with the whole scene in view; wheel/pinch zoom freely afterwards.
  useEffect(() => {
    setZoom(fitZoom());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sceneW, sceneH]);

  // Keyboard navigation; ignored while typing (chat box, dialog fields).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      switch (e.key) {
        case "ArrowLeft":
          setPan((p) => ({ ...p, x: p.x + PAN_STEP }));
          break;
        case "ArrowRight":
          setPan((p) => ({ ...p, x: p.x - PAN_STEP }));
          break;
        case "ArrowUp":
          setPan((p) => ({ ...p, y: p.y + PAN_STEP }));
          break;
        case "ArrowDown":
          setPan((p) => ({ ...p, y: p.y - PAN_STEP }));
          break;
        case "+":
        case "=":
          setZoom((z) => clamp(z * 1.15, MIN_ZOOM, MAX_ZOOM));
          break;
        case "-":
        case "_":
          setZoom((z) => clamp(z / 1.15, MIN_ZOOM, MAX_ZOOM));
          break;
        case "[":
          setRotation((r) => r - 90);
          break;
        case "]":
          setRotation((r) => r + 90);
          break;
        case "0":
          setZoom(fitZoom());
          setPan({ x: 0, y: 0 });
          // Back to the viewer's facing, the short way round.
          setRotation(reface(faceAngle));
          break;
        default:
          return;
      }
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sceneW, sceneH, faceAngle]);

  return {
    containerRef,
    sceneTransform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
    rotationTransform: `rotate(${rotation}deg)`,
    rotate: (deg: number) => setRotation((r) => r + deg),
  };
}
