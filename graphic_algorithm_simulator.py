#!/usr/bin/env python3
"""
Graph Algorithm Simulator (tkinter)
Features:
- Left-click canvas to create node.
- Click a node to select it; click another node to create an edge between them.
- When creating an edge you'll be prompted for a weight (press Enter or leave empty for weight=1).
- Choose algorithm: BFS, DFS, Dijkstra.
- Choose start and target nodes (for algorithms that use them).
- Play / Pause / Step through animation. Reset / Clear.
"""

import tkinter as tk
from tkinter import simpledialog, ttk, messagebox
import math
import heapq
import time
import threading

NODE_RADIUS = 18
NODE_FILL = "#ffffff"
NODE_OUTLINE = "#333333"
EDGE_COLOR = "#666666"
VISITED_COLOR = "#ffcc66"
FRONTIER_COLOR = "#66ccff"
PATH_COLOR = "#66ff66"
START_COLOR = "#88ff88"
TARGET_COLOR = "#ff8888"

class GraphSimulator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Graph Algorithm Simulator")
        self.geometry("1000x650")

        self.create_widgets()
        self.reset_graph_state()

        # animation control
        self._running = False
        self._step_once = False
        self._speed = 0.5

    def create_widgets(self):
        # left frame: canvas
        left = tk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        toolbar = tk.Frame(left)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.canvas = tk.Canvas(left, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # right frame: controls
        right = tk.Frame(self, width=280, padx=8)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        # algorithm selection
        tk.Label(right, text="Algorithm").pack(anchor="w", pady=(6,0))
        self.algo_var = tk.StringVar(value="BFS")
        algo_menu = ttk.Combobox(right, textvariable=self.algo_var,
                                 values=["BFS", "DFS", "Dijkstra"], state="readonly")
        algo_menu.pack(fill=tk.X, pady=4)

        # start/target node selectors
        tk.Label(right, text="Start Node").pack(anchor="w", pady=(8,0))
        self.start_var = tk.StringVar(value="")
        self.start_menu = ttk.Combobox(right, textvariable=self.start_var, values=[], state="readonly")
        self.start_menu.pack(fill=tk.X, pady=4)

        tk.Label(right, text="Target Node (optional)").pack(anchor="w", pady=(8,0))
        self.target_var = tk.StringVar(value="")
        self.target_menu = ttk.Combobox(right, textvariable=self.target_var, values=[], state="readonly")
        self.target_menu.pack(fill=tk.X, pady=4)

        # buttons
        btn_frame = tk.Frame(right)
        btn_frame.pack(fill=tk.X, pady=8)
        tk.Button(btn_frame, text="Run", command=self.run_algorithm).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(btn_frame, text="Step", command=self.step).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(btn_frame, text="Pause", command=self.pause).pack(side=tk.LEFT, expand=True, fill=tk.X)

        tk.Button(right, text="Reset Colors", command=self.reset_colors).pack(fill=tk.X, pady=6)
        tk.Button(right, text="Clear Graph", command=self.clear_graph).pack(fill=tk.X, pady=6)

        # speed control
        tk.Label(right, text="Animation Speed (s)").pack(anchor="w", pady=(8,0))
        self.speed_scale = tk.Scale(right, from_=0.05, to=1.5, resolution=0.05,
                                    orient=tk.HORIZONTAL, command=self.on_speed_change)
        self.speed_scale.set(0.5)
        self.speed_scale.pack(fill=tk.X, pady=4)

        # instructions
        inst = (
            "Instructions:\n"
            "- Left-click empty space: add node\n"
            "- Left-click a node to select it (first click), then click another node to add edge\n"
            "- When creating an edge you can set weight (default=1)\n"
            "- Select Start/Target, choose algorithm and click Run\n"
        )
        tk.Label(right, text=inst, justify="left", wraplength=260).pack(anchor="w", pady=6)

        # status bar
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

    def reset_graph_state(self):
        self.nodes = {}        # node_id -> (x,y)
        self.node_items = {}   # node_id -> canvas_oval_id
        self.node_labels = {}  # node_id -> text_id
        self.adj = {}          # node_id -> list of (neighbor_id, weight, edge_line_id)
        self.edge_items = {}   # (u,v) sorted tuple -> line_id and weight_text_id
        self.next_id = 1
        self.selected_node = None
        self.rebuild_node_menus()
        self.redraw()

    def rebuild_node_menus(self):
        keys = sorted(map(str, self.nodes.keys()), key=lambda s: int(s) if s else 0)
        self.start_menu['values'] = keys
        self.target_menu['values'] = keys
        if keys:
            self.start_var.set(keys[0])
        else:
            self.start_var.set("")
            self.target_var.set("")

    def clear_graph(self):
        self.canvas.delete("all")
        self.reset_graph_state()
        self.status("Graph cleared")

    def on_speed_change(self, val):
        try:
            self._speed = float(val)
        except:
            pass

    def on_canvas_click(self, event):
        x, y = event.x, event.y
        # clicked on node?
        clicked = self.find_node_at(x, y)
        if clicked:
            self.on_node_click(clicked)
        else:
            self.add_node(x, y)

    def find_node_at(self, x, y):
        for nid, (nx, ny) in self.nodes.items():
            if (x - nx)**2 + (y - ny)**2 <= NODE_RADIUS**2:
                return nid
        return None

    def add_node(self, x, y):
        nid = self.next_id
        self.next_id += 1
        self.nodes[nid] = (x, y)
        item = self.canvas.create_oval(x-NODE_RADIUS, y-NODE_RADIUS, x+NODE_RADIUS, y+NODE_RADIUS,
                                       fill=NODE_FILL, outline=NODE_OUTLINE, width=2)
        label = self.canvas.create_text(x, y, text=str(nid))
        self.node_items[nid] = item
        self.node_labels[nid] = label
        self.adj[nid] = []
        self.rebuild_node_menus()
        self.status(f"Added node {nid}")

    def on_node_click(self, nid):
        if self.selected_node is None:
            self.selected_node = nid
            self.highlight_node(nid, FRONTIER_COLOR)
            self.status(f"Selected node {nid} as first endpoint. Click another node to create edge.")
        else:
            if nid == self.selected_node:
                # deselect
                self.highlight_node(nid, NODE_FILL)
                self.selected_node = None
                self.status("Deselected node.")
                return
            u, v = self.selected_node, nid
            self.create_edge(u, v)
            self.highlight_node(self.selected_node, NODE_FILL)
            self.selected_node = None

    def create_edge(self, u, v):
        # ask weight
        w = simpledialog.askstring("Edge weight", f"Enter weight for edge {u} - {v} (default 1):", parent=self)
        try:
            w = float(w) if w is not None and w.strip() != "" else 1.0
        except:
            messagebox.showwarning("Invalid", "Weight must be a number. Using weight=1.")
            w = 1.0

        x1, y1 = self.nodes[u]
        x2, y2 = self.nodes[v]
        line = self.canvas.create_line(x1, y1, x2, y2, fill=EDGE_COLOR, width=2)
        # weight label midpoint
        mx, my = (x1+x2)/2, (y1+y2)/2
        txt = self.canvas.create_text(mx, my-10, text=str(w))
        self.adj[u].append((v, w, line))
        self.adj[v].append((u, w, line))  # undirected
        key = tuple(sorted((u, v)))
        self.edge_items[key] = (line, txt, w)
        self.status(f"Edge {u} <-> {v} weight={w}")

    def highlight_node(self, nid, color):
        item = self.node_items.get(nid)
        if item:
            self.canvas.itemconfigure(item, fill=color)

    def color_edge(self, u, v, color, width=3):
        key = tuple(sorted((u, v)))
        entry = self.edge_items.get(key)
        if entry:
            line_id = entry[0]
            self.canvas.itemconfigure(line_id, fill=color, width=width)

    def reset_colors(self):
        for nid, item in self.node_items.items():
            self.canvas.itemconfigure(item, fill=NODE_FILL)
        for key, (line, txt, w) in self.edge_items.items():
            self.canvas.itemconfigure(line, fill=EDGE_COLOR, width=2)
        self.status("Colors reset")

    def redraw(self):
        self.canvas.delete("all")
        # draw edges
        for (u, v), (line_id, txt_id, w) in list(self.edge_items.items()):
            x1, y1 = self.nodes[u]
            x2, y2 = self.nodes[v]
            line = self.canvas.create_line(x1, y1, x2, y2, fill=EDGE_COLOR, width=2)
            mx, my = (x1+x2)/2, (y1+y2)/2
            text = self.canvas.create_text(mx, my-10, text=str(w))
            self.edge_items[(u, v)] = (line, text, w)
        # draw nodes
        self.node_items = {}
        self.node_labels = {}
        for nid, (x, y) in self.nodes.items():
            item = self.canvas.create_oval(x-NODE_RADIUS, y-NODE_RADIUS, x+NODE_RADIUS, y+NODE_RADIUS,
                                           fill=NODE_FILL, outline=NODE_OUTLINE, width=2)
            label = self.canvas.create_text(x, y, text=str(nid))
            self.node_items[nid] = item
            self.node_labels[nid] = label

    def status(self, msg):
        self.status_var.set(msg)
        # also print for console debugging
        print(msg)

    # ---------- Algorithm drivers ----------
    def run_algorithm(self):
        if not self.nodes:
            messagebox.showinfo("No nodes", "Create some nodes first.")
            return

        algo = self.algo_var.get()
        start = self.start_var.get()
        if not start:
            messagebox.showinfo("Start missing", "Choose a start node.")
            return
        start = int(start)
        target = self.target_var.get()
        target = int(target) if target else None

        # reset colors
        self.reset_colors()
        self.highlight_node(start, START_COLOR)
        if target:
            self.highlight_node(target, TARGET_COLOR)

        # choose function
        if algo == "BFS":
            gen = self.bfs(start, target)
        elif algo == "DFS":
            gen = self.dfs(start, target)
        elif algo == "Dijkstra":
            gen = self.dijkstra(start, target)
        else:
            messagebox.showerror("Unknown", f"Unknown algorithm {algo}")
            return

        # run generator in separate thread to keep UI responsive
        self._running = True
        self._step_once = False
        thread = threading.Thread(target=self._run_generator_thread, args=(gen,), daemon=True)
        thread.start()

    def _run_generator_thread(self, gen):
        try:
            for action in gen:
                # actions are dicts describing visualization operations
                self.apply_action(action)
                # wait while paused
                while not self._running and not self._step_once:
                    time.sleep(0.05)
                # if step was requested, execute single action then pause again
                if self._step_once:
                    self._step_once = False
                    self._running = False
                time.sleep(self._speed)
        except Exception as e:
            print("Algorithm thread error:", e)
        finally:
            self.status("Algorithm finished.")
            self._running = False

    def step(self):
        # single-step: if not running, allow one action then pause
        self._step_once = True
        self._running = True

    def pause(self):
        self._running = False
        self.status("Paused.")

    def apply_action(self, action):
        typ = action.get("type")
        if typ == "visit":
            nid = action["node"]
            self.highlight_node(nid, VISITED_COLOR)
            self.status(f"Visited {nid}")
        elif typ == "frontier":
            nid = action["node"]
            self.highlight_node(nid, FRONTIER_COLOR)
            self.status(f"Added to frontier: {nid}")
        elif typ == "edge_frontier":
            u, v = action["edge"]
            self.color_edge(u, v, FRONTIER_COLOR, width=3)
            self.status(f"Edge frontier {u}-{v}")
        elif typ == "edge_visit":
            u, v = action["edge"]
            self.color_edge(u, v, VISITED_COLOR, width=3)
            self.status(f"Edge visited {u}-{v}")
        elif typ == "path":
            path = action["path"]
            for (u, v) in zip(path, path[1:]):
                self.color_edge(u, v, PATH_COLOR, width=4)
                self.highlight_node(u, PATH_COLOR)
                self.highlight_node(v, PATH_COLOR)
            self.status(f"Path: {' -> '.join(map(str, path))}")
        elif typ == "info":
            self.status(action.get("text", ""))
        # force canvas refresh
        self.canvas.update_idletasks()

    # ---------- Algorithms implemented as generators yielding visualization actions ----------
    def bfs(self, start, target=None):
        from collections import deque
        visited = set()
        parent = {}
        q = deque()
        q.append(start)
        visited.add(start)
        yield {"type": "frontier", "node": start}
        while q:
            u = q.popleft()
            yield {"type": "visit", "node": u}
            if target is not None and u == target:
                # build path
                path = self._reconstruct_path(parent, start, target)
                yield {"type": "path", "path": path}
                return
            for v, w, _ in sorted(self.adj.get(u, []), key=lambda x: x[0]):
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    q.append(v)
                    yield {"type": "frontier", "node": v}
                    yield {"type": "edge_frontier", "edge": (u, v)}
            yield {"type": "info", "text": f"Queue: {list(q)}"}
        if target is not None:
            yield {"type": "info", "text": "Target not reachable"}

    def dfs(self, start, target=None):
        visited = set()
        parent = {}
        stack = [start]
        yield {"type": "frontier", "node": start}
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            yield {"type": "visit", "node": u}
            if target is not None and u == target:
                path = self._reconstruct_path(parent, start, target)
                yield {"type": "path", "path": path}
                return
            # push neighbors (use reverse order to get natural ordering)
            neighbors = sorted(self.adj.get(u, []), key=lambda x: -x[0])
            for v, w, _ in neighbors:
                if v not in visited:
                    parent[v] = u
                    stack.append(v)
                    yield {"type": "frontier", "node": v}
                    yield {"type": "edge_frontier", "edge": (u, v)}
            yield {"type": "info", "text": f"Stack top: {stack[-1] if stack else None}"}
        if target is not None:
            yield {"type": "info", "text": "Target not reachable"}

    def dijkstra(self, start, target=None):
        dist = {n: math.inf for n in self.nodes}
        parent = {}
        dist[start] = 0
        pq = [(0, start)]
        visited = set()
        yield {"type": "frontier", "node": start}
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            yield {"type": "visit", "node": u}
            if target is not None and u == target:
                path = self._reconstruct_path(parent, start, target)
                yield {"type": "path", "path": path}
                return
            for v, w, _ in self.adj.get(u, []):
                nd = d + w
                yield {"type": "edge_frontier", "edge": (u, v)}
                if nd < dist[v]:
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(pq, (nd, v))
                    yield {"type": "frontier", "node": v}
            yield {"type": "info", "text": f"Visited={sorted(list(visited))}"}
        if target is not None:
            yield {"type": "info", "text": "Target not reachable"}

    def _reconstruct_path(self, parent, start, target):
        path = []
        cur = target
        while True:
            path.append(cur)
            if cur == start or cur not in parent:
                break
            cur = parent[cur]
        path.reverse()
        return path

if __name__ == "__main__":
    app = GraphSimulator()
    app.mainloop()

