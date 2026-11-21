// Room-by-room push-optimal Sokoban solver, written for this competition.
//
// The generator's levels are a one-wide hallway with 1-4 rooms off it, each
// room 3-4 boxes with as many goals, and a level's push-optimal count P* is
// the sum of its rooms' optima (a box pushed out of a door can never come
// back, so rooms are independent problems joined only by the player's walk).
// This solver therefore:
//   1. splits a level into rooms (the hallway's dead squares cut the floor),
//   2. solves each room push-optimally with A* over pushes, and
//   3. stitches the rooms' move strings together (the walk between rooms
//      falls out of the per-room search, which starts where the player is).
//
// The search state is (sorted boxes, lowest square of the player's reachable
// region): the player's exact square inside its region never matters for a
// push. The heuristic is the minimum-cost assignment of boxes to goals over
// wall-only BFS distances (admissible and consistent: one push moves one box
// one square, so the assignment cost moves by at most one). Deadlock pruning:
// a push may only land a box on a live square (a square from which a box can
// still be pushed onto a goal, computed by reverse pulls from the goals) and
// never in a frozen 2x2 block of walls and boxes. Budgets are by work (states
// created), never by the clock, so two runs answer byte-identically.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** One search state per room: at most this many states before giving up. */
const MAX_ROOM_STATES = 1_000_000;
/** All rooms of all instances in one run together: a deterministic work cap. */
const MAX_TOTAL_STATES = 4_000_000;
let totalStates = 0;

const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** Parse XSB: '#' wall, ' ' floor, '.' goal, '$' box, '*' box on goal, '@' player. */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outer frame is wall whatever it shows: nothing leaves the board.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * Live squares: a square is live when a box alone on it can be pushed onto
 * some goal. Computed by pulling boxes back from the goals: a box on y can be
 * pushed to x when the player can stand on y - d (open floor) and x is live.
 */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
  done: boolean;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway:
 * a box there reaches no goal) out of the floor; each remaining part holding
 * a box is a room. Null when a box starts in the hallway, a goal is cut off,
 * or a part holds a different count of boxes and goals.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) {
    if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  }
  if (goals.some((g) => part[g]! < 0)) return null;
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** The squares the player reaches from `from` around walls and `occupied`. */
function reachInto(
  level: Level,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
function frozenBlock(level: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) {
      continue;
    }
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) {
      return true;
    }
  }
  return false;
}

/** Shortest walking distance from every goal over walls alone (boxes ignored). */
function goalDistances(level: Level): Int32Array[] {
  const { w, wall, goals } = level;
  return goals.map((g) => {
    const dist = new Int32Array(wall.length).fill(-1);
    const queue = [g];
    dist[g] = 0;
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi]!;
      for (const [dx, dy] of DIRS) {
        const q = p + dy * w + dx;
        if (!wall[q] && dist[q] < 0) {
          dist[q] = dist[p]! + 1;
          queue.push(q);
        }
      }
    }
    return dist;
  });
}

/**
 * Minimum-cost assignment of boxes to goals over the goal distances; Infinity
 * when some box reaches no goal. Exact (all goal permutations) for up to 5
 * boxes, otherwise the sum of each box's nearest goal (still admissible).
 */
function assignmentHeuristic(boxes: number[], dists: Int32Array[], nGoals: number): number {
  const n = boxes.length;
  if (n === 0) return 0;
  if (n > 5) {
    let sum = 0;
    for (const b of boxes) {
      let best = Infinity;
      for (let g = 0; g < nGoals; g++) {
        const d = dists[g]![b]!;
        if (d >= 0 && d < best) best = d;
      }
      if (best === Infinity) return Infinity;
      sum += best;
    }
    return sum;
  }
  // Exact assignment by trying every order of the goals.
  const order = Array.from({ length: nGoals }, (_, i) => i);
  let best = Infinity;
  const permute = (k: number): void => {
    if (k === n) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const d = dists[order[i]!]![boxes[i]!]!;
        if (d < 0) return;
        sum += d;
      }
      if (sum < best) best = sum;
      return;
    }
    for (let i = k; i < nGoals; i++) {
      [order[k], order[i]] = [order[i]!, order[k]!];
      permute(k + 1);
      [order[k], order[i]] = [order[i]!, order[k]!];
    }
  };
  permute(0);
  return best;
}

/** The shortest walk from `from` to `to` around walls and boxes, lower LURD letters (first direction first on ties). */
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how: string[] = new Array(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

interface Solved {
  pushes: number;
  moves: string;
  endPlayer: number;
}

/**
 * Push-optimal A* for one level (a room, or a whole level as a fallback).
 * States are (boxes, player-region lowest square); each push costs 1.
 * Returns null when the state budget runs out or the level has no solution.
 */
function solvePushAStar(level: Level): Solved | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  const dists = goalDistances(level);
  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);
  if (level.boxes.some((b) => !live[b])) return null;

  // States, in creation order: boxes, player square, region low, parent, push from->to, g.
  const sBoxes: number[][] = [level.boxes.slice()];
  const sPlayer: number[] = [level.player];
  const sLow: number[] = [-1];
  const sParent: number[] = [-1];
  const sFrom: number[] = [-1];
  const sTo: number[] = [-1];
  const sG: number[] = [0];
  const keyOf = (boxes: number[], low: number): string => `${low}:${boxes.join(',')}`;
  const best = new Map<string, number>();

  const h0 = assignmentHeuristic(level.boxes, dists, goals.length);
  if (h0 === Infinity) return null;
  for (const b of level.boxes) occupied[b] = 1;
  const low0 = reachInto(level, occupied, level.player, seen, ++stamp, stack);
  for (const b of level.boxes) occupied[b] = 0;
  sLow[0] = low0;
  best.set(keyOf(level.boxes, low0), 0);

  // Binary heap on (f, h, insertion order): [f, h, seq, state].
  const heap: [number, number, number, number][] = [[h0, h0, 0, 0]];
  let seq = 1;
  const heapLess = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1]!, heap[c]!)) c++;
        if (!heapLess(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  let created = 1;
  totalStates++;
  if (solved(level.boxes)) return { pushes: 0, moves: '', endPlayer: level.player };

  while (heap.length > 0) {
    const [, , , s] = heapPop();
    const boxes = sBoxes[s]!;
    const g = sG[s]!;
    // Skip a stale heap entry (a better path to this state was found).
    if (best.get(keyOf(boxes, sLow[s]!)) !== g) continue;
    if (solved(boxes)) {
      // Rebuild the push path, then the moves: walk to each push, then push.
      const path: number[] = [];
      for (let t = s; t > 0; t = sParent[t]!) path.push(t);
      path.reverse();
      let moves = '';
      let player = level.player;
      let cur = level.boxes.slice();
      for (const t of path) {
        const from = sFrom[t]!;
        const to = sTo[t]!;
        const d = to - from;
        for (const b of cur) occupied[b] = 1;
        moves += walkTo(level, occupied, player, from - d);
        for (const b of cur) occupied[b] = 0;
        const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
        moves += dir[3];
        player = from;
        cur = sBoxes[t]!.slice();
      }
      return { pushes: path.length, moves, endPlayer: player };
    }
    // Expand: the player's region, then every push out of it.
    for (const b of boxes) occupied[b] = 1;
    const here = ++stamp;
    reachInto(level, occupied, sPlayer[s]!, seen, here, stack);
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(level, occupied, goal, t);
        let key = '';
        let next: number[] | null = null;
        let low = -1;
        let h = Infinity;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          low = reachInto(level, occupied, b, mark, ++stamp, stack);
          key = keyOf(next, low);
          const prevBest = best.get(key);
          if (prevBest === undefined || prevBest > g + 1) {
            h = assignmentHeuristic(next, dists, goals.length);
          }
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || next === null || h === Infinity) continue;
        if (created >= MAX_ROOM_STATES || totalStates >= MAX_TOTAL_STATES) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        const prevBest = best.get(key);
        if (prevBest !== undefined && prevBest <= g + 1) continue;
        best.set(key, g + 1);
        sBoxes.push(next);
        sPlayer.push(b);
        sLow.push(low);
        sParent.push(s);
        sFrom.push(b);
        sTo.push(t);
        sG.push(g + 1);
        created++;
        totalStates++;
        heapPush([g + 1 + h, h, seq++, sBoxes.length - 1]);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  return null;
}

/** One level solved room by room; the whole level as a fallback when it has no rooms. */
function solveLevel(level: Level): string {
  const rooms = roomsOf(level);
  if (rooms === null) {
    const whole = solvePushAStar(level);
    return whole === null ? '' : whole.moves;
  }
  let player = level.player;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solvePushAStar(sub);
    if (solved === null) return moves;
    moves += solved.moves;
    player = solved.endPlayer;
    room.done = true;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(parseLevel(readFileSync(join(I, id, 'level.txt'), 'utf8')));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
