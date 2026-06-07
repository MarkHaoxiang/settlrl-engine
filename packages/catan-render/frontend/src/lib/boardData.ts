// Board types + palette, and the adapter from the server's wire format
// (GET /api/board, snake_case) to the frontend's camelCase Board.

import { api } from "./api";
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
  resources: Record<ResourceKind, number>;
  devCardTypes: Record<DevCardKind, number>;
}

export interface Board {
  tiles: Tile[];
  buildings: Building[];
  roads: RoadSeg[];
  ports: PortData[];
  players: Player[];
  robber?: Hex;
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

// -- wire format (models.py BoardModel) ----------------------------------------

interface PlayerWire {
  player: number;
  resource_cards: number;
  dev_cards: number;
  victory_points: number;
  resources: Record<ResourceKind, number>;
  dev_card_types: Record<DevCardKind, number>;
}

export interface BoardWire {
  tiles: { q: number; r: number; terrain: Terrain; number: number | null }[];
  buildings: Building[];
  roads: RoadSeg[];
  ports: PortData[];
  players: PlayerWire[];
  robber: Hex | null;
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
  };
}

export async function fetchBoard(): Promise<Board> {
  return adaptBoard(await api<BoardWire>("/api/board"));
}
