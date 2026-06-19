// Visual-check harness: serve the built dist (npm run preview), inject canned
// API data, and screenshot the chrome screens in both themes so styling changes
// can be eyeballed without a live backend.
//   npm run build && npm run preview -- --port 4178 &
//   BASE=http://localhost:4178 CHROME=/usr/bin/chromium node e2e/screenshots.mjs
import { mkdirSync } from "node:fs";
import { chromium } from "playwright-core";

const CHROME = process.env.CHROME ?? "/usr/bin/chromium";
const BASE = process.env.BASE ?? "http://localhost:4178";
const OUT = process.env.OUT ?? "/tmp/settlrl-shots";
mkdirSync(OUT, { recursive: true });

const FIXTURES = {
  "/api/users/me": { id: "u1", email: "alice@example.com", is_superuser: true },
  "/api/me/games": [{ id: "abcdef12", seats: [0] }],
  "/api/me/history": [
    { id: "g1", seats: [0], n_players: 4, winner: 0, finished_at: 1750000000 },
    { id: "g2", seats: [1], n_players: 2, winner: 0, finished_at: 1750100000 },
  ],
  "/api/leaderboard": [
    { n_players: 2, kind: "bot", name: "greedy", rating: 1623, games: 40, wins: 28 },
    { n_players: 2, kind: "account", name: "alice", rating: 1588, games: 30, wins: 16 },
    { n_players: 2, kind: "bot", name: "lookahead", rating: 1541, games: 22, wins: 9 },
    { n_players: 4, kind: "account", name: "bob", rating: 1502, games: 12, wins: 3 },
  ],
  "/api/bots": { greedy: { title: "Greedy", description: "a bot", counts: [2, 3, 4] } },
  "/api/admin/bot-providers": [{ name: "greedy", base_url: "http://localhost:8100" }],
};

const browser = await chromium.launch({ executablePath: CHROME });
const ctx = await browser.newContext({
  viewport: { width: 1100, height: 850 },
  deviceScaleFactor: 2,
});
await ctx.route("**/api/**", (route) => {
  const path = new URL(route.request().url()).pathname;
  route.fulfill({ json: FIXTURES[path] ?? [] });
});

const SCREENS = [
  ["menu", "/"],
  ["leaderboard", "/leaderboard"],
  ["profile", "/profile"],
];

for (const [name, path] of SCREENS) {
  for (const theme of ["light", "dark"]) {
    const page = await ctx.newPage();
    await page.addInitScript(
      ([t]) => {
        localStorage.setItem("settlrl-theme", t);
        localStorage.setItem("settlrl-auth-token", "faketoken");
      },
      [theme]
    );
    await page.goto(BASE + path, { waitUntil: "networkidle" });
    await page.waitForTimeout(400);
    await page.screenshot({ path: `${OUT}/${name}-${theme}.png`, fullPage: true });
    await page.close();
    console.log(`shot ${name}-${theme}`);
  }
}
await browser.close();
