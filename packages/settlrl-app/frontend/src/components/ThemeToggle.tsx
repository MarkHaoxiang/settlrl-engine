import { useState } from "react";
import { currentTheme, toggleTheme, type Theme } from "../lib/theme";
import { buttonStyle } from "../lib/ui";

// Light / dark switch. The themes live in CSS variables, so flipping the body
// attribute restyles everything; local state just keeps the icon current.
export default function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(currentTheme());
  return (
    <button
      title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
      style={{ ...buttonStyle, padding: "3px 8px", fontSize: 13, lineHeight: 1.2 }}
      onClick={() => setTheme(toggleTheme())}
    >
      {theme === "dark" ? "☀️" : "🌙"}
    </button>
  );
}
