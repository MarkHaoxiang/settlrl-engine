import { useState } from "react";
import { login, logout, register, type AuthUser } from "../lib/auth";
import { buttonStyle, panelStyle, smallButtonStyle } from "../lib/ui";

// A compact sign-in / account control for the menu. Accounts are optional —
// signed out, everything still works — so this stays out of the way until
// opened. Controlled: the menu owns the user so it can also list their games.
export default function AccountMenu({
  user,
  onUser,
}: {
  user: AuthUser | null;
  onUser: (user: AuthUser | null) => void;
}) {
  const [open, setOpen] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (mode: "login" | "register") => {
    setBusy(true);
    setError(null);
    try {
      if (mode === "register") await register(email, password);
      onUser(await login(email, password));
      setOpen(false);
      setPassword("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  if (user) {
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 13 }}>
        <span style={{ opacity: 0.8 }}>
          {user.email}
          {user.is_superuser ? " · admin" : ""}
        </span>
        <button
          style={smallButtonStyle}
          onClick={() => void logout().then(() => onUser(null))}
        >
          Log out
        </button>
      </div>
    );
  }

  if (!open) {
    return (
      <button style={smallButtonStyle} onClick={() => setOpen(true)}>
        Sign in
      </button>
    );
  }

  const inputStyle: React.CSSProperties = {
    ...panelStyle,
    padding: "8px 10px",
    fontSize: 13,
    width: 220,
  };
  return (
    <div
      style={{
        ...panelStyle,
        display: "flex",
        flexDirection: "column",
        gap: 8,
        padding: 16,
        borderRadius: 12,
        boxShadow: "0 6px 24px rgba(0,0,0,0.4)",
      }}
    >
      <input
        style={inputStyle}
        type="email"
        placeholder="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
      />
      <input
        style={inputStyle}
        type="password"
        placeholder="password (8+ characters)"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && void submit("login")}
      />
      {error && <span style={{ color: "var(--error)", fontSize: 12, maxWidth: 220 }}>{error}</span>}
      <div style={{ display: "flex", gap: 8 }}>
        <button style={buttonStyle} disabled={busy} onClick={() => void submit("login")}>
          Log in
        </button>
        <button style={buttonStyle} disabled={busy} onClick={() => void submit("register")}>
          Register
        </button>
        <button style={smallButtonStyle} onClick={() => setOpen(false)}>
          Cancel
        </button>
      </div>
    </div>
  );
}
