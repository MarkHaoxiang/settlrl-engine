// Full-page sign-in / registration. Accounts are optional — gameplay works
// signed out (seat tokens) — so this is reached from the menu's "Sign in" link,
// never forced. `?next=` is where a successful sign-in returns to (default the
// menu). Already signed in: bounce straight there.

import { useEffect, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import ThemeToggle from "../components/ThemeToggle";
import { currentUser, login, register } from "../lib/auth";
import ui from "../styles/ui.module.css";
import s from "./AuthView.module.css";

type Mode = "login" | "register";

// fastapi-users surfaces failures as terse codes; show something human.
function friendly(message: string): string {
  switch (message) {
    case "REGISTER_USER_ALREADY_EXISTS":
      return "An account with that email already exists — try logging in.";
    case "LOGIN_BAD_CREDENTIALS":
    case "LOGIN_USER_NOT_VERIFIED":
      return "Wrong email or password.";
    default:
      return message;
  }
}

export default function AuthView({ initialMode = "login" }: { initialMode?: Mode }) {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const next = params.get("next") || "/";

  const [mode, setMode] = useState<Mode>(initialMode);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Don't show a sign-in form to someone already signed in.
  useEffect(() => {
    void currentUser().then((u) => u && navigate(next, { replace: true }));
  }, [navigate, next]);

  const registering = mode === "register";
  const mismatch = registering && confirm !== "" && confirm !== password;
  const canSubmit =
    email.trim() !== "" &&
    password.length >= 8 &&
    (!registering || password === confirm) &&
    !busy;

  const submit = async () => {
    if (!canSubmit) return;
    setBusy(true);
    setError(null);
    try {
      if (registering) await register(email.trim(), password);
      await login(email.trim(), password);
      navigate(next, { replace: true });
    } catch (e) {
      setError(friendly(e instanceof Error ? e.message : String(e)));
    } finally {
      setBusy(false);
    }
  };

  const swap = (to: Mode) => {
    setMode(to);
    setError(null);
    setConfirm("");
  };

  return (
    <div className={s.page}>
      <div className={ui.toolbarTopRight}>
        <ThemeToggle />
      </div>
      <Link to="/" className={s.backLink}>
        ← Menu
      </Link>

      <h1 className={s.title}>Settlrl</h1>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
        className={s.form}
      >
        <span className={s.heading}>
          {registering ? "Create your account" : "Welcome back"}
        </span>

        <label className={s.label}>
          Email
          <input
            className={s.input}
            type="email"
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </label>

        <label className={s.label}>
          Password
          <input
            className={s.input}
            type="password"
            autoComplete={registering ? "new-password" : "current-password"}
            placeholder={registering ? "8+ characters" : undefined}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </label>

        {registering && (
          <label className={s.label}>
            Confirm password
            <input
              className={mismatch ? s.inputError : s.input}
              type="password"
              autoComplete="new-password"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
            />
          </label>
        )}

        {(error || mismatch) && (
          <span className={s.error}>
            {mismatch ? "Passwords don't match." : error}
          </span>
        )}

        <button type="submit" disabled={!canSubmit} className={ui.buttonPrimary}>
          {registering ? "Create account" : "Log in"}
        </button>

        <div className={s.swapRow}>
          {registering ? "Already have an account? " : "New here? "}
          <button
            type="button"
            onClick={() => swap(registering ? "login" : "register")}
            className={s.swap}
          >
            {registering ? "Log in" : "Create an account"}
          </button>
        </div>
      </form>
    </div>
  );
}
