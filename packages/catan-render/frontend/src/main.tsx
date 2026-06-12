import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import "./index.css";
import App from "./App";
import { initTheme } from "./lib/theme";

initTheme();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {/* basename follows the build-time base, so routes work behind a
        stripped proxy prefix (vite --base=/catan/) as well as at /. */}
    <BrowserRouter basename={import.meta.env.BASE_URL.replace(/\/$/, "")}>
      <App />
    </BrowserRouter>
  </StrictMode>
);
