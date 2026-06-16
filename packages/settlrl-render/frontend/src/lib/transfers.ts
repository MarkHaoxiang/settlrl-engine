// What moved between two consecutive snapshots, as tokens to fly on the table.
//
// Snapshots are one-per-move, so a single state change separates them. We read
// the headline card motions straight from public counts (each player's hand
// total and the bank's per-resource piles) — no action decoding needed:
//
//   production  — the bank shrinks while only hands grow (a dice payout, or a
//                 year-of-plenty draw): bank piles → each gaining seat.
//   steal       — one hand loses a card and one gains it, bank unchanged
//                 (robber / knight): victim seat → thief seat.
//
// Everything else (builds, buys, monopoly) leaves a different signature and is
// deliberately skipped. A card's resource is only coloured when it is public —
// the bank pile it came from, or a hand this client owns; an opponent's hidden
// gain flies face-down (resource null).
//
// A domestic trade is the exception that count diffs can't see (one card each
// way nets zero), so tradeTransfer derives it from the resolved offer instead
// (both resources are public, being the offer's terms).

import { RESOURCE_ORDER, type Board, type Player, type ResourceKind } from "./boardData";

export type Anchor =
  | { kind: "bank"; resource: ResourceKind | null }
  | { kind: "seat"; seat: number };

export interface FlyToken {
  id: string;
  from: Anchor;
  to: Anchor;
  resource: ResourceKind | null; // null => a face-down card back
}

// Most a single seat shows flying at once; a big payout still reads clearly
// without burying the board under a swarm of chips.
const MAX_PER_SEAT = 5;

const handTotal = (players: Player[]) => players.map((p) => p.resourceCards);
const bankTotal = (board: Board) =>
  board.bank ? RESOURCE_ORDER.reduce((s, r) => s + board.bank!.resources[r], 0) : 0;

// The one resource that changed by exactly `delta` in a hand, if this client
// can see that hand's breakdown (else null — the card stays face-down).
function changedResource(before: Player, after: Player, delta: number): ResourceKind | null {
  if (!before.resources || !after.resources) return null;
  for (const r of RESOURCE_ORDER) {
    if (after.resources[r] - before.resources[r] === delta) return r;
  }
  return null;
}

export function deriveTransfers(prev: Board, next: Board, key: string): FlyToken[] {
  if (!prev.bank || !next.bank) return [];
  const before = handTotal(prev.players);
  const after = handTotal(next.players);
  const seatDelta = after.map((n, i) => n - before[i]);
  const gainers = seatDelta.map((d, i) => ({ i, d })).filter((s) => s.d > 0);
  const losers = seatDelta.map((d, i) => ({ i, d })).filter((s) => s.d < 0);
  const bankDelta = bankTotal(next) - bankTotal(prev);

  // Steal: a single card crosses from one hand to another, bank untouched.
  if (bankDelta === 0 && gainers.length === 1 && losers.length === 1 && gainers[0].d === 1) {
    const victim = losers[0].i;
    const thief = gainers[0].i;
    const resource =
      changedResource(prev.players[thief], next.players[thief], 1) ??
      changedResource(prev.players[victim], next.players[victim], -1);
    return [{ id: `${key}-steal`, from: { kind: "seat", seat: victim }, to: { kind: "seat", seat: thief }, resource }];
  }

  // Production: the bank pays out — it shrinks and only hands grow.
  if (bankDelta < 0 && losers.length === 0 && gainers.length > 0) {
    const tokens: FlyToken[] = [];
    for (const { i, d } of gainers) {
      const before = prev.players[i].resources;
      const after = next.players[i].resources;
      const seat: FlyToken[] = [];
      if (before && after) {
        // A hand this client owns: colour each chip by its bank pile of origin.
        for (const r of RESOURCE_ORDER)
          for (let k = after[r] - before[r]; k > 0; k--)
            seat.push({ id: `${key}-p${i}-${seat.length}`, from: { kind: "bank", resource: r }, to: { kind: "seat", seat: i }, resource: r });
      } else {
        for (let k = 0; k < d; k++)
          seat.push({ id: `${key}-p${i}-${seat.length}`, from: { kind: "bank", resource: null }, to: { kind: "seat", seat: i }, resource: null });
      }
      tokens.push(...seat.slice(0, MAX_PER_SEAT));
    }
    return tokens;
  }

  return [];
}

// The two cards crossing on an accepted 1:1 domestic trade: the proposer's
// `give` to the partner, the partner's `receive` back. Both resources are
// public (the offer's terms), so each chip flies coloured. `offer` is the
// trade pending on the prior snapshot; `accepted` whether the move that
// resolved it took it (a rejection moves no cards).
export function tradeTransfer(
  offer: { proposer: number; partner: number; give: string; receive: string } | null | undefined,
  accepted: boolean,
  key: string
): FlyToken[] {
  if (!offer || !accepted) return [];
  return [
    {
      id: `${key}-trade-give`,
      from: { kind: "seat", seat: offer.proposer },
      to: { kind: "seat", seat: offer.partner },
      resource: offer.give as ResourceKind,
    },
    {
      id: `${key}-trade-recv`,
      from: { kind: "seat", seat: offer.partner },
      to: { kind: "seat", seat: offer.proposer },
      resource: offer.receive as ResourceKind,
    },
  ];
}
