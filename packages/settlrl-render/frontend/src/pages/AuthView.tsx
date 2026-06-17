// Full-page sign-in / registration. Accounts are optional — gameplay works
// signed out (seat tokens) — so this is reached from the menu's "Sign in" link,
// never forced. `?next=` is where a successful sign-in returns to (default the
// menu). Already signed in: bounce straight there.

import { useEffect, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import ThemeToggle from "../components/ThemeToggle";
import { currentUser, login, register } from "../lib/auth";
import { ACCENT, FONT, LINK, buttonStyle, panelStyle } from "../lib/ui";

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

  const inputStyle: React.CSSProperties = {
    ...panelStyle,
    padding: "10px 12px",
    fontSize: 14,
    width: "100%",
    boxSizing: "border-box",
    userSelect: "text",
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 24,
        padding: 24,
        color: "var(--text)",
        fontFamily: FONT,
      }}
    >
      <div style={{ position: "fixed", top: 16, right: 16 }}>
        <ThemeToggle />
      </div>
      <Link to="/" style={{ position: "fixed", top: 16, left: 16, color: LINK, textDecoration: "none", fontSize: 14 }}>
        ← Menu
      </Link>

      <h1 style={{ fontSize: 40, margin: 0, letterSpacing: 1 }}>Settlrl</h1>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
        style={{
          ...panelStyle,
          display: "flex",
          flexDirection: "column",
          gap: 12,
          padding: "28px 28px",
          width: 320,
          borderRadius: 16,
          boxShadow: "0 6px 24px rgba(0,0,0,0.4)",
        }}
      >
        <span style={{ fontSize: 22, fontWeight: 700 }}>
          {registering ? "Create your account" : "Welcome back"}
        </span>

        <label style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 12, opacity: 0.7 }}>
          Email
          <input
            style={inputStyle}
            type="email"
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </label>

        <label style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 12, opacity: 0.7 }}>
          Password
          <input
            style={inputStyle}
            type="password"
            autoComplete={registering ? "new-password" : "current-password"}
            placeholder={registering ? "8+ characters" : undefined}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </label>

        {registering && (
          <label style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 12, opacity: 0.7 }}>
            Confirm password
            <input
              style={{ ...inputStyle, ...(mismatch ? { borderColor: "var(--error)" } : {}) }}
              type="password"
              autoComplete="new-password"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
            />
          </label>
        )}

        {(error || mismatch) && (
          <span style={{ color: "var(--error)", fontSize: 12 }}>
            {mismatch ? "Passwords don't match." : error}
          </span>
        )}

        <button
          type="submit"
          disabled={!canSubmit}
          style={{
            ...buttonStyle,
            background: ACCENT,
            borderColor: ACCENT,
            color: "#1a1206",
            fontWeight: 700,
            opacity: canSubmit ? 1 : 0.5,
            cursor: canSubmit ? "pointer" : "default",
          }}
        >
          {registering ? "Create account" : "Log in"}
        </button>

        <div style={{ fontSize: 13, opacity: 0.8, textAlign: "center" }}>
          {registering ? "Already have an account? " : "New here? "}
          <button
            type="button"
            onClick={() => swap(registering ? "login" : "register")}
            style={{ background: "none", border: "none", color: LINK, cursor: "pointer", fontFamily: FONT, fontSize: 13, padding: 0 }}
          >
            {registering ? "Log in" : "Create an account"}
          </button>
        </div>
      </form>
    </div>
  );
}
