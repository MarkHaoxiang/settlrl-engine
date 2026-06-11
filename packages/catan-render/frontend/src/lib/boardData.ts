// Board types + palette, and the adapter from the server's wire format
// (BoardModel, snake_case) to the frontend's camelCase Board.

import type { Cube, CubeEdge, Hex } from "./hex";

export type Terrain = "wheat" | "sheep" | "wood" | "ore" | "brick" | "desert";
export type ResourceKind = Exclude<Terrain, "desert">;
export type DevCardKind =
  | "knight"
  | "road_building"
  | "year_of_plenty"
  | "monopoly"
  | "victory_point";

export interface Tile {
  hex: Hex;
  terrain: Terrain;
  number?: number;
}

export interface Building {
  cube: Cube;
  player: number;
  kind: "settlement" | "city";
}

export interface RoadSeg extends CubeEdge {
  player: number;
}

export interface PortData extends CubeEdge {
  // 2:1 resource port, or null for a 3:1 general port.
  resource: ResourceKind | null;
}

export interface Player {
  player: number;
  resourceCards: number;
  devCards: number;
  victoryPoints: number;
  // Per-type breakdowns are private: null for seats this client doesn't own.
  resources: Record<ResourceKind, number> | null;
  devCardTypes: Record<DevCardKind, number> | null;
}

// Cards left in the supply: resource stacks plus the development deck.
export interface Bank {
  resources: Record<ResourceKind, number>;
  devCards: number;
}

export interface Board {
  tiles: Tile[];
  buildings: Building[];
  roads: RoadSeg[];
  ports: PortData[];
  players: Player[];
  robber?: Hex;
  bank?: Bank;
}

// -- palette ------------------------------------------------------------------

export const PLAYER_NAMES = ["Red", "Blue", "White", "Orange"];
export const PLAYER_COLORS = ["#C8341F", "#2C5F9E", "#E8E4D8", "#D97B29"];
export const PLAYER_STROKES = ["#7A1F11", "#16365C", "#9A9684", "#8A4D17"];

export const playerName = (player: number): string =>
  PLAYER_NAMES[player] ?? `Player ${player + 1}`;

export const TERRAIN_FILL: Record<Terrain, string> = {
  wheat: "#EEC900",
  sheep: "#7DC95E",
  wood: "#2D6A2D",
  ore: "#8B949E",
  brick: "#C0392B",
  desert: "#E8D5A3",
};

export const TERRAIN_STROKE: Record<Terrain, string> = {
  wheat: "#B89A00",
  sheep: "#4E9A35",
  wood: "#1A4A1A",
  ore: "#5A666E",
  brick: "#8E2319",
  desert: "#C4B080",
};

// Display order and labels for the five resources, shared by every list of
// resource chips / piles in the UI.
export const RESOURCE_ORDER: ResourceKind[] = ["wood", "brick", "sheep", "wheat", "ore"];
export const RESOURCE_LABELS: Record<ResourceKind, string> = {
  wood: "Wood",
  brick: "Brick",
  sheep: "Sheep",
  wheat: "Wheat",
  ore: "Ore",
};

// Card colours: the development cards' purple and the face-down hand back.
export const DEV_CARD_BACK = { fill: "#5B4B8A", stroke: "#3C3160" };
export const HAND_CARD_BACK = { fill: "#C9A66B", stroke: "#7A5C33" };

// -- wire format (models.py BoardModel) ----------------------------------------

interface PlayerWire {
  player: number;
  resource_cards: number;
  dev_cards: number;
  victory_points: number;
  resources: Record<ResourceKind, number> | null;
  dev_card_types: Record<DevCardKind, number> | null;
}

export interface BoardWire {
  tiles: { q: number; r: number; terrain: Terrain; number: number | null }[];
  buildings: Building[];
  roads: RoadSeg[];
  ports: PortData[];
  players: PlayerWire[];
  robber: Hex | null;
  bank: { resources: Record<ResourceKind, number>; dev_cards: number } | null;
}

export function adaptBoard(wire: BoardWire): Board {
  return {
    tiles: wire.tiles.map(({ q, r, terrain, number }) => ({
      hex: { q, r },
      terrain,
      number: number ?? undefined,
    })),
    buildings: wire.buildings,
    roads: wire.roads,
    ports: wire.ports,
    players: wire.players.map((p) => ({
      player: p.player,
      resourceCards: p.resource_cards,
      devCards: p.dev_cards,
      victoryPoints: p.victory_points,
      resources: p.resources,
      devCardTypes: p.dev_card_types,
    })),
    robber: wire.robber ?? undefined,
    bank: wire.bank
      ? { resources: wire.bank.resources, devCards: wire.bank.dev_cards }
      : undefined,
  };
}

