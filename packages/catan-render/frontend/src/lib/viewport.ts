import { useEffect, useRef, useState } from "react";

const MIN_ZOOM = 0.3;
const MAX_ZOOM = 3;
const PAN_STEP = 60;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

// Pan / zoom / rotate for the table, driven by mouse drag, wheel, pinch, and
// keyboard (arrows pan, +/- zoom, [ ] spin a quarter turn, 0 re-fits). The
// view opens fitted to the scene.
//
// Wiring: put `containerRef` + `containerHandlers` on the viewport element,
// `sceneTransform` on a layer inside it, and `rotationTransform` on a second
// layer inside that — rotation gets its own (transitioned) layer so spinning
// animates without dragging on pan/zoom updates.
export function useTableViewport(sceneW: number, sceneH: number) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  // Cumulative degrees, so each quarter turn animates the short way round
  // instead of unwinding.
  const [rotation, setRotation] = useState(0);

  // Active drag-to-pan gesture; `moved` flips once past the click-vs-drag
  // threshold, after which the pointer is captured so releasing over a board
  // element doesn't also click it.
  const drag = useRef<{
    id: number;
    x: number;
    y: number;
    panX: number;
    panY: number;
    moved: boolean;
  } | null>(null);

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    if (drag.current) {
      // A second touch landed: this is a pinch, not a pan.
      drag.current = null;
      return;
    }
    drag.current = { id: e.pointerId, x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y, moved: false };
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d || e.pointerId !== d.id) return;
    const dx = e.clientX - d.x;
    const dy = e.clientY - d.y;
    if (!d.moved && Math.hypot(dx, dy) > 4) {
      d.moved = true;
      e.currentTarget.setPointerCapture(d.id);
    }
    if (d.moved) setPan({ x: d.panX + dx, y: d.panY + dy });
  };

  const onPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (drag.current?.id === e.pointerId) drag.current = null;
  };

  // Distance between the two active touch points during a pinch gesture.
  const pinchStart = useRef<{ dist: number; zoom: number } | null>(null);
  // Latest zoom, so the native handlers below read it without re-subscribing.
  const zoomRef = useRef(zoom);
  zoomRef.current = zoom;

  // Wheel + pinch zoom. Attached natively so we can preventDefault (React's
  // onWheel is passive and would still scroll/zoom the page).
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      // Trackpads report small deltas; mice report large ones. Exponential
      // scaling keeps the zoom feel consistent across both.
      const factor = Math.exp(-e.deltaY * 0.0015);
      setZoom((z) => clamp(z * factor, MIN_ZOOM, MAX_ZOOM));
    };

    const touchDist = (t: TouchList) =>
      Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY);

    const onTouchStart = (e: TouchEvent) => {
      if (e.touches.length === 2) {
        pinchStart.current = { dist: touchDist(e.touches), zoom: zoomRef.current };
      }
    };
    const onTouchMove = (e: TouchEvent) => {
      if (e.touches.length === 2 && pinchStart.current) {
        e.preventDefault();
        const ratio = touchDist(e.touches) / pinchStart.current.dist;
        setZoom(clamp(pinchStart.current.zoom * ratio, MIN_ZOOM, MAX_ZOOM));
      }
    };
    const onTouchEnd = (e: TouchEvent) => {
      if (e.touches.length < 2) pinchStart.current = null;
    };

    el.addEventListener("wheel", onWheel, { passive: false });
    el.addEventListener("touchstart", onTouchStart, { passive: false });
    el.addEventListener("touchmove", onTouchMove, { passive: false });
    el.addEventListener("touchend", onTouchEnd);
    return () => {
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("touchstart", onTouchStart);
      el.removeEventListener("touchmove", onTouchMove);
      el.removeEventListener("touchend", onTouchEnd);
    };
  }, []);

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
          // Unwind to the nearest full turn so the reset spins the short way.
          setRotation((r) => Math.round(r / 360) * 360);
          break;
        default:
          return;
      }
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sceneW, sceneH]);

  return {
    containerRef,
    containerHandlers: {
      onPointerDown,
      onPointerMove,
      onPointerUp,
      onPointerCancel: onPointerUp,
    },
    sceneTransform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
    rotationTransform: `rotate(${rotation}deg)`,
    rotate: (deg: number) => setRotation((r) => r + deg),
  };
}
