import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createGame, type NewGameConfig } from "./game";
import { rememberGame, saveTokens } from "./seats";

const QUEUE_POLL_MS = 3000;

export interface CreateQueue {
  position: number;
  total: number;
}

// Drives the create-a-game flow: start(config) creates the game and navigates
// into its lobby room, re-polling with the returned ticket while the server is
// at its concurrency cap. queue/error expose the in-flight state for the caller
// to render; cancel abandons a queued request.
export function useCreateGame() {
  const navigate = useNavigate();
  const [queue, setQueue] = useState<CreateQueue | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  const poll = async (config: NewGameConfig, ticket?: string) => {
    try {
      const res = await createGame(config, ticket);
      if ("queued" in res) {
        setQueue({ position: res.position, total: res.total });
        timer.current = window.setTimeout(() => void poll(config, res.ticket), QUEUE_POLL_MS);
      } else {
        setQueue(null);
        saveTokens(res.id, res.tokens);
        rememberGame(res.id);
        navigate(`/lobby/${res.id}`);
      }
    } catch (e) {
      setQueue(null);
      setError(`Could not create the game: ${String(e)}`);
    }
  };

  const start = (config: NewGameConfig) => {
    setError(null);
    void poll(config);
  };

  const cancel = () => {
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = null;
    setQueue(null);
  };

  // Stop polling if the caller unmounts mid-queue.
  useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    []
  );

  return { start, queue, error, cancel };
}
