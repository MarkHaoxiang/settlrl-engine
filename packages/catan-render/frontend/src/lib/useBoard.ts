// Hook that loads the board once for a view (Replay / shared frame).

import { useEffect, useState } from "react";
import { fetchBoard, type Board } from "./boardData";

export function useBoard(): { board: Board | null; error: string | null } {
  const [board, setBoard] = useState<Board | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchBoard().then(
      (b) => !cancelled && setBoard(b),
      (e: unknown) => !cancelled && setError(`Failed to load board: ${String(e)}`)
    );
    return () => {
      cancelled = true;
    };
  }, []);

  return { board, error };
}
