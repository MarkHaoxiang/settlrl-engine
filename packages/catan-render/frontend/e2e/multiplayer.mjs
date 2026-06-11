// End-to-end multiplayer check, driving the real app in headless Chromium:
// create a game in one browser, join it from two more (seat claims via the
// invite link), and verify per-seat views — markers and hands only for your
// own seats, spectating when none is left.
//
// Run a server with a built frontend first (defaults below), then:
//   npm run e2e            # BASE=http://localhost:8000 CHROME=/usr/bin/chromium
import { chromium } from "playwright-core";

const BASE = process.env.BASE ?? "http://localhost:8000";
const CHROME = process.env.CHROME ?? "/usr/bin/chromium";

let failures = 0;
const check = (name, ok) => {
  console.log(`${ok ? "ok " : "FAIL"} ${name}`);
  if (!ok) failures += 1;
};

const browser = await chromium.launch({ executablePath: CHROME });
const page = async () => {
  const ctx = await browser.newContext({ viewport: { width: 1500, height: 950 } });
  return ctx.newPage();
};

// --- solo hotseat create through the UI -------------------------------------
const A = await page();
await A.goto(`${BASE}/play`);
await A.waitForTimeout(600);
await A.getByPlaceholder("random").fill("7");
await A.getByRole("button", { name: "Start", exact: true }).click();
await A.waitForTimeout(1500);
check("create navigates to /play/{id}", A.url().includes("/play/"));
check("creator sees setup markers", (await A.locator(".board-ghost").count()) > 0);

// --- a 2-human game across two more browsers --------------------------------
const created = await A.evaluate(() =>
  fetch("/api/games", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      seed: 3,
      seats: ["human", "human", "random", "random"],
      claim: "none",
    }),
  }).then((r) => r.json())
);
const id = created.id;

const B = await page();
await B.goto(`${BASE}/play/${id}`);
await B.waitForTimeout(1200);
const C = await page();
await C.goto(`${BASE}/play/${id}`);
await C.waitForTimeout(1200);

const seatsOf = (p, gid = id) =>
  p.evaluate((g) => Object.keys(JSON.parse(localStorage.getItem("catan-seats") ?? "{}")[g] ?? {}), gid);
check("first joiner claims seat 0", String(await seatsOf(B)) === "0");
check("second joiner claims seat 1", String(await seatsOf(C)) === "1");
check("acting seat sees markers", (await B.locator(".board-ghost").count()) > 0);
check("waiting seat sees none", (await C.locator(".board-ghost").count()) === 0);

const cView = await C.evaluate(async (gid) => {
  const tokens = JSON.parse(localStorage.getItem("catan-seats") ?? "{}")[gid] ?? {};
  const r = await fetch(`/api/games/${gid}`, {
    headers: { "X-Seat-Tokens": Object.values(tokens).join(",") },
  });
  return r.json();
}, id);
check(
  "wire view redacts the other hand",
  cView.board.players[1].resources !== null && cView.board.players[0].resources === null
);
check("belief observes the owned seat", cView.belief?.observer === 1);

// B plays setup; C's turn arrives via the waiting poll.
await B.locator(".board-ghost").first().click({ force: true });
await B.getByRole("button", { name: /Place settlement/ }).click();
await B.waitForTimeout(400);
await B.locator(".board-ghost").first().click({ force: true });
await B.getByRole("button", { name: /Place road/ }).click();
for (let w = 0; w < 30; w++) {
  if ((await C.locator(".board-ghost").count()) > 0) break;
  await C.waitForTimeout(400);
}
check("turn hands over to the second player", (await C.locator(".board-ghost").count()) > 0);

// --- a third client spectates -----------------------------------------------
const D = await page();
await D.goto(`${BASE}/play/${id}`);
await D.waitForTimeout(1200);
check(
  "full game spectates",
  (await D.evaluate(() => document.body.innerText)).includes("Spectating")
);

// --- online seating through the dialog ---------------------------------------
// Two human seats + the "online" choice: the creator gets only seat 0, the
// invite link hands out seat 1.
const E = await page();
await E.goto(`${BASE}/play`);
await E.waitForTimeout(800);
await E.getByRole("button", { name: "human", exact: true }).nth(1).click();
await E.getByRole("button", { name: "online", exact: true }).click();
await E.getByRole("button", { name: "Start", exact: true }).click();
await E.waitForTimeout(1500);
const onlineId = E.url().split("/play/")[1];
check("online create claims only the first seat", String(await seatsOf(E, onlineId)) === "0");
const F = await page();
await F.goto(`${BASE}/play/${onlineId}`);
await F.waitForTimeout(1200);
check("invitee claims the second seat", String(await seatsOf(F, onlineId)) === "1");

await browser.close();
if (failures > 0) {
  console.error(`${failures} check(s) failed`);
  process.exit(1);
}
console.log("all checks passed");
