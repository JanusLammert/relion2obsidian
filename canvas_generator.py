"""
    This program auto-creates mark-down files for obsidian to visualize Relion Projects
    Copyright (C) 2026  Janus Lammert

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
canvas_generator.py
====================
Ergänzungsmodul für rel2obsi_beta.py
Erzeugt eine Obsidian Canvas-Datei (.canvas) mit einer Baumansicht
aller RELION-Jobs – ähnlich dem CryoSPARC Job-Graph.

Verwendung (standalone):
    python canvas_generator.py -i /pfad/zum/relion/projekt -o /pfad/zum/obsidian/vault

Oder als Import in rel2obsi_beta.py:
    from canvas_generator import build_canvas_from_jobs
    build_canvas_from_jobs(jobs, output_dir)
"""

import os
import re
import json
import argparse
import logging
import traceback
import sys
from pathlib import Path
from collections import defaultdict

import starfile
from tqdm import tqdm

# Logging übernehmen wenn als Modul genutzt
logger = logging.getLogger("rel2obsi.canvas")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

# ---------------------------------------------------------------------------
# Farben je Job-Typ (Obsidian Canvas unterstützt 6 Preset-Farben: 1-6)
# 1=rot, 2=orange, 3=gelb, 4=grün, 5=cyan, 6=lila
# ---------------------------------------------------------------------------
JOB_TYPE_COLORS = {
    "Import":           "4",   # grün  – Dateneingang
    "MotionCorr":       "3",   # gelb
    "CtfFind":          "3",   # gelb
    "ManualPick":       "5",   # cyan
    "AutoPick":         "5",   # cyan
    "Extract":          "2",   # orange
    "Select":           "2",   # orange
    "Subset":           "2",   # orange
    "Class2D":          "6",   # lila
    "Class3D":          "6",   # lila
    "InitialModel":     "6",   # lila
    "Refine3D":         "1",   # rot   – wichtige Ergebnisse
    "PostProcess":      "1",   # rot
    "Polish":           "1",   # rot
    "CtfRefine":        "1",   # rot
    "BayesianPolishing":"1",   # rot
    "MaskCreate":       "3",   # gelb
    "LocalRes":         "3",   # gelb
    "MultiBody":        "6",   # lila
}

# Knotenabmessungen
NODE_W = 400
NODE_H = 400
H_GAP  = 200  # horizontaler Abstand zwischen Spalten
V_GAP  = 60   # vertikaler Abstand zwischen Knoten in einer Spalte


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def get_note_stem(job_name: str, job_type: str) -> str:
    """Gibt den Dateinamen der Markdown-Notiz zurück (ohne .md)"""
    safe = sanitize_filename(job_name)
    return f"{safe}_{job_type}"


def parse_pipeline_edges(project_dir: str):
    """
    Liest default_pipeline.star (oder pipeline.star) und gibt
    zwei Dicts zurück:
        inputs:  { 'CtfFind/job003' : ['MotionCorr/job002', ...] }
        outputs: { 'MotionCorr/job002': ['CtfFind/job003', ...] }

    Direktes Parsen aus der Pipeline – NICHT aus den einzelnen job.star/json-Dateien.
    Dadurch werden alle Edges zuverlässig erfasst.
    """
    possible = [
        os.path.join(project_dir, "default_pipeline.star"),
        os.path.join(project_dir, "pipeline.star"),
    ]
    pipeline_path = next((p for p in possible if os.path.exists(p)), None)

    inputs  = defaultdict(list)   # job -> [upstream_jobs]
    outputs = defaultdict(list)   # job -> [downstream_jobs]

    if pipeline_path is None:
        logger.warning("Keine pipeline.star / default_pipeline.star gefunden – keine Edges.")
        return inputs, outputs

    try:
        data = starfile.read(pipeline_path)
    except Exception as e:
        logger.error(f"Fehler beim Lesen der Pipeline-Datei {pipeline_path}: {e}")
        return inputs, outputs

    processes = (
        data.get("data_pipeline_processes")
        or data.get("pipeline_processes")
    )
    edges = (
        data.get("data_pipeline_input_edges")
        or data.get("pipeline_input_edges")
    )

    if processes is None or edges is None:
        logger.warning("Pipeline-Datei enthält keine Prozesse oder Edges.")
        return inputs, outputs

    # Hilfsfunktion: Node-Pfad -> Job-Name (z.B. 'CtfFind/job003/micrographs_ctf.star' -> 'CtfFind/job003')
    def node_to_job(node_path: str) -> str | None:
        parts = node_path.strip().rstrip("/").split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return None

    # Alle Prozessnamen in ein Set laden (für schnelle Lookups)
    proc_names = set(
        p.strip().rstrip("/") for p in processes["rlnPipeLineProcessName"]
    )

    for _, edge in edges.iterrows():
        from_node  = edge.get("rlnPipeLineEdgeFromNode", "").strip()
        to_process = edge.get("rlnPipeLineEdgeProcess",  "").strip().rstrip("/")

        upstream_job   = node_to_job(from_node)
        downstream_job = node_to_job(to_process)

        if upstream_job is None or downstream_job is None:
            continue
        if upstream_job == downstream_job:
            continue   # Selbst-Referenz überspringen

        # Sicherstellen, dass beide Jobs tatsächlich existieren
        # (leicht unterschiedliche Schreibweisen abfangen)
        upstream_norm   = upstream_job.rstrip("/")
        downstream_norm = downstream_job.rstrip("/")

        if upstream_norm not in inputs[downstream_norm]:
            inputs[downstream_norm].append(upstream_norm)
        if downstream_norm not in outputs[upstream_norm]:
            outputs[upstream_norm].append(downstream_norm)

    logger.info(
        f"Pipeline gelesen: {len(inputs)} Jobs mit Inputs, "
        f"{len(outputs)} Jobs mit Outputs"
    )
    return inputs, outputs


def topo_sort(all_job_names, inputs, outputs=None):
    """
    Kahn's Algorithmus für topologische Sortierung.
    Gibt geordnete Liste zurück und eine depth-Map { job_name: column }.

    outputs: dict { job_name: [downstream_jobs] } – wird direkt genutzt
             statt O(n²)-Rückwärtssuche über inputs.
    """
    if outputs is None:
        # Fallback: outputs aus inputs rekonstruieren (langsam, nur für Rückwärtskompatibilität)
        outputs = defaultdict(list)
        for child, parents in inputs.items():
            for parent in parents:
                if child not in outputs[parent]:
                    outputs[parent].append(child)

    all_job_set = set(all_job_names)
    in_degree = {j: len([p for p in inputs.get(j, []) if p in all_job_set])
                 for j in all_job_names}
    queue = sorted([j for j in all_job_names if in_degree[j] == 0])
    order = []
    depth = {j: 0 for j in all_job_names}

    while queue:
        node = queue.pop(0)
        order.append(node)
        # Direkte Nachfolger aus outputs-Dict – O(1) statt O(n)
        for child in outputs.get(node, []):
            if child not in all_job_set:
                continue
            depth[child] = max(depth.get(child, 0), depth[node] + 1)
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
                queue.sort()

    # Knoten die nicht im topo-sort gelandet sind (Zyklen) anhängen
    remaining = [j for j in all_job_names if j not in order]
    order.extend(remaining)

    return order, depth


def assign_positions(jobs_by_name: dict, inputs: dict, outputs: dict = None):
    """
    Berechnet x,y-Koordinaten für jeden Job-Knoten als zentrierter Baum.

    Algorithmus (Reingold-Tilford-ähnlich, Bottom-Up):
      1. Topologische Tiefe → x-Spalte (links = früh, rechts = spät)
      2. Leaf-Knoten werden von oben nach unten gestapelt (nach Job-Nummer sortiert)
      3. Jeder innere Knoten wird vertikal zentriert über seine direkten Kinder gesetzt
      4. Ein globaler Offset-Pass verhindert Überlappungen zwischen unabhängigen Teilbäumen

    Gibt dict { job_name: (x, y) } zurück.
    """
    all_names = list(jobs_by_name.keys())
    _, depth = topo_sort(all_names, inputs, outputs)

    if outputs is None:
        outputs = defaultdict(list)
        for child, parents in inputs.items():
            for parent in parents:
                if child not in outputs[parent]:
                    outputs[parent].append(child)

    # Hilfsfunktion: Job-Nummer extrahieren für Sortierung
    def job_num(name):
        parts = name.split("/")
        seg = parts[1] if len(parts) >= 2 else parts[0]
        m = re.search(r'\d+', seg)
        return int(m.group()) if m else 0

    # --- Schritt 1: Blattknoten je Spalte von oben stapeln -----------------
    # Blattknoten = keine Nachfolger ODER alle Nachfolger sind in früherer Spalte
    # (verhindert Sackgassen-Jobs, die als innere Knoten fehlklassifiziert würden)
    cols = defaultdict(list)
    for name in all_names:
        cols[depth[name]].append(name)
    for col in cols:
        cols[col].sort(key=job_num)

    # Initiale y-Positionen: Blätter erhalten feste Slots, Innere folgen später
    positions = {}   # name -> (x, y)  –  y ist zunächst ein float-Platzhalter

    # Verarbeite Spalten von rechts (tiefste Tiefe) nach links (Wurzeln)
    max_depth = max(depth.values()) if depth else 0

    # y-Cursor je Spalte – wächst von oben nach unten
    col_cursor = defaultdict(float)   # col_idx -> nächste freie y-Position

    def subtree_y_span(name):
        """Gibt (min_y, max_y) des gesamten Teilbaums zurück (nach aktuellem Stand)."""
        children = [c for c in outputs.get(name, []) if c in positions]
        if not children:
            return positions[name][1], positions[name][1]
        spans = [subtree_y_span(c) for c in children]
        return min(s[0] for s in spans), max(s[1] for s in spans)

    # Verarbeite alle Tiefen von der tiefsten zur flachsten
    for col_idx in range(max_depth, -1, -1):
        nodes_in_col = cols.get(col_idx, [])
        x = col_idx * (NODE_W + H_GAP)

        for name in nodes_in_col:
            children = [c for c in outputs.get(name, []) if c in positions]

            if not children:
                # Blattknoten: nächsten freien Slot in dieser Spalte belegen
                y = col_cursor[col_idx]
                col_cursor[col_idx] += NODE_H + V_GAP
                positions[name] = (x, y)
            else:
                # Innerer Knoten: vertikal über Kinder zentrieren
                child_ys = [positions[c][1] for c in children]
                y = (min(child_ys) + max(child_ys)) / 2.0

                # Sicherstellen, dass wir nicht über bereits platzierten Knoten
                # in dieser Spalte landen
                y = max(y, col_cursor[col_idx])
                col_cursor[col_idx] = y + NODE_H + V_GAP
                positions[name] = (x, y)

    # --- Schritt 2: Überlappungen in jeder Spalte durch Verschieben beheben --
    # Sortiere pro Spalte nach y und schiebe Knoten auseinander falls nötig.
    # Danach zentriere Eltern erneut über ihre Kinder (ein Pass genügt für DAGs).
    for col_idx in range(max_depth + 1):
        nodes_in_col = sorted(cols.get(col_idx, []),
                              key=lambda n: positions[n][1])
        x = col_idx * (NODE_W + H_GAP)
        cursor = 0.0
        for name in nodes_in_col:
            y = max(positions[name][1], cursor)
            positions[name] = (x, y)
            cursor = y + NODE_H + V_GAP

    # --- Schritt 3: Eltern-Zentrierung nach Überlappungskorrektur (von rechts) --
    for col_idx in range(max_depth - 1, -1, -1):
        nodes_in_col = sorted(cols.get(col_idx, []),
                              key=lambda n: positions[n][1])
        x = col_idx * (NODE_W + H_GAP)
        for name in nodes_in_col:
            children = [c for c in outputs.get(name, []) if c in positions]
            if children:
                child_ys = [positions[c][1] for c in children]
                positions[name] = (x, (min(child_ys) + max(child_ys)) / 2.0)

    # Ganzzahlige Koordinaten für sauberes JSON
    return {name: (int(x), int(y)) for name, (x, y) in positions.items()}


# ---------------------------------------------------------------------------
# Canvas-Generierung
# ---------------------------------------------------------------------------

def build_canvas_from_jobs(
    jobs: list,
    output_dir: str,
    project_dir: str = None,
    canvas_name: str = None,
    canvas_depth: int = 2,
):
    """
    Erzeugt eine Obsidian Canvas-Datei.

    Parameter
    ---------
    jobs         : Liste der Job-Dicts aus parse_relion_jobs()
    output_dir   : Verzeichnis, in dem die Markdown-Notizen liegen
    project_dir  : RELION-Projektverzeichnis (für Pipeline-Parsing)
                   Wenn None, wird versucht es aus den Job-Details abzuleiten.
    canvas_name  : Dateiname des Canvas (ohne .canvas). Fallback: Projektverzeichnisname.
    canvas_depth : Wie viele Ebenen liegt das Canvas-Verzeichnis ÜBER output_dir?
                   Standard 2: vault/canvas.canvas + vault/Projekt/Subprojekt/notes
                   Das Canvas wird in output_dir/../../ gespeichert (canvas_depth=2).
                   Die file-Pfade in den Nodes werden entsprechend angepasst.
    """
    try:
        # --- Projekt-Verzeichnis ermitteln ---
        if project_dir is None:
            for job in jobs:
                jp = job["details"].get("job_path", "")
                if jp:
                    # job_path ist z.B. /data/project/CtfFind/job003/job.star
                    # Wir wollen /data/project
                    p = Path(jp)
                    # Gehe 2 Ebenen hoch (job_dir / job_type_dir / project_dir)
                    if p.is_file():
                        p = p.parent
                    project_dir = str(p.parent.parent)
                    break

        if project_dir is None:
            logger.error("project_dir konnte nicht ermittelt werden.")
            return

        # --- Canvas-Verzeichnis & Dateiname bestimmen ---
        # Canvas wird `canvas_depth` Ebenen über output_dir gespeichert.
        canvas_dir = Path(output_dir).resolve()
        for _ in range(canvas_depth):
            canvas_dir = canvas_dir.parent

        # Relativer Präfix für Notiz-Pfade im Canvas (von canvas_dir aus gesehen)
        notes_rel = Path(output_dir).resolve().relative_to(canvas_dir)
        notes_prefix = str(notes_rel).replace("\\", "/")  # Windows-sicher

        # Canvas-Dateiname: CLI-Argument > Projektverzeichnisname
        if not canvas_name:
            canvas_name = os.path.basename(os.path.abspath(project_dir))
        safe_canvas_name = sanitize_filename(canvas_name)

        # --- Jobs in Dict umwandeln ---
        # job["name"] ist jetzt wieder der kurze Name (z.B. "job003").
        # Für Pipeline-Lookups brauchen wir aber "JobType/jobXXX" als Key.
        # Wir bauen eine Mapping-Tabelle short_name -> full_name.
        jobs_by_name = {}   # key: "JobType/jobXXX"  (für Pipeline-Matching)
        short_to_full = {}  # key: "job003" -> "CtfFind/job003"
        for job in jobs:
            raw_name  = job["name"]           # z.B. "job003"
            job_type  = job.get("type", "Unknown")
            full_name = f"{job_type}/{raw_name}"   # z.B. "CtfFind/job003"
            jobs_by_name[full_name] = job
            short_to_full[raw_name] = full_name

        # --- Edges aus Pipeline lesen (zuverlässiger als job.star) ---
        inputs, outputs = parse_pipeline_edges(project_dir)

        # Fehlende Jobs aus Edges ergänzen (können im Scan fehlen)
        all_referenced = set(inputs.keys()) | set(outputs.keys())
        for ref in all_referenced:
            if ref not in jobs_by_name:
                parts = ref.split("/")
                job_type = parts[0] if parts else "Unknown"
                jobs_by_name[ref] = {
                    "name": parts[1] if len(parts) >= 2 else ref,
                    "type": job_type,
                    "details": {"tags": []},
                }

        # --- Positionen berechnen ---
        logger.info("Berechne Layout...")
        positions = assign_positions(jobs_by_name, inputs, outputs)

        # --- Canvas-Datenstruktur aufbauen ---
        canvas_nodes = []
        canvas_edges = []
        node_id_map  = {}   # full_name -> canvas node id

        for full_name, job in jobs_by_name.items():
            node_id = f"node_{sanitize_filename(full_name)}"
            node_id_map[full_name] = node_id

            x, y = positions.get(full_name, (0, 0))
            job_type  = job.get("type", "Unknown")
            raw_name  = job["name"]   # z.B. "job003"

            # Markdown-Notiz: get_note_stem erwartet den vollen Namen für den Dateinamen.
            # rel2obsi_beta.py erzeugt: {safe_name}_{job_type}.md  mit safe_name = "job003"
            note_stem = f"{sanitize_filename(raw_name)}_{job_type}"
            # Relativer Pfad vom Canvas-Verzeichnis zur Notiz
            note_file = f"{notes_prefix}/{note_stem}.md"

            # Anzeige-Name im Canvas: "job003\nCtfFind"
            display = f"{raw_name}\n{job_type}"

            # Prüfe ob Job abgeschlossen ist
            job_dir = Path(job["details"].get("job_path", ""))
            if job_dir.is_file():
                job_dir = job_dir.parent
            success_file = job_dir / "RELION_JOB_EXIT_SUCCESS"
            is_complete = success_file.exists() if job_dir != Path("") else False

            node = {
                "id":     node_id,
                "type":   "file",
                "file":   note_file,
                "x":      x,
                "y":      y,
                "width":  NODE_W,
                "height": NODE_H,
                "label":  display,
            }

            color = JOB_TYPE_COLORS.get(job_type, "")
            if color:
                node["color"] = color

            if not is_complete:
                node["color"] = ""   # kein Farboverride → grau

            canvas_nodes.append(node)

        # --- Edges (aus inputs-Dict) ---
        edge_id_counter = 0
        seen_edges = set()
        for downstream, upstream_list in inputs.items():
            for upstream in upstream_list:
                edge_key = (upstream, downstream)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                src_id = node_id_map.get(upstream)
                tgt_id = node_id_map.get(downstream)
                if src_id is None or tgt_id is None:
                    continue

                canvas_edges.append({
                    "id":       f"edge_{edge_id_counter}",
                    "fromNode": src_id,
                    "fromSide": "right",
                    "toNode":   tgt_id,
                    "toSide":   "left",
                })
                edge_id_counter += 1

        canvas_data = {
            "nodes": canvas_nodes,
            "edges": canvas_edges,
        }

        # --- Datei schreiben ---
        os.makedirs(str(canvas_dir), exist_ok=True)
        canvas_path = canvas_dir / f"{safe_canvas_name}.canvas"

        with open(canvas_path, "w", encoding="utf-8") as f:
            json.dump(canvas_data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Canvas erstellt: {canvas_path} "
            f"({len(canvas_nodes)} Knoten, {len(canvas_edges)} Kanten)"
        )
        return str(canvas_path)

    except Exception as e:
        logger.error(f"Error while creating the Canvas: {e}")
        logger.debug(traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Standalone-Call
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Creates an Obsidian Canvas for a Relion Project"
    )
    parser.add_argument("-i", "--project_dir",  required=True,
                        help="Path to the RELION Directory")
    parser.add_argument("-o", "--output_dir",   required=True,
                        help="Path to the Obsidian notes directory")
    parser.add_argument("-v", "--verbose",       action="store_true",
                        help="Extensive logging")
    parser.add_argument(
        "--canvas-name",
        default=None,
        help=(
            "Name of the Canvas file (without .canvas). "
            "Defaults to the project directory name."
        ),
    )
    parser.add_argument(
        "--canvas-depth",
        type=int,
        default=2,
        help=(
            "How many levels above --output_dir the Canvas file should be saved. "
            "Default: 2  (vault/Canvas.canvas + vault/Project/Subproject/notes)"
        ),
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Importiere parse_relion_jobs aus dem Hauptskript
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from rel2obsi_beta import parse_relion_jobs
    except ImportError:
        logger.error(
            "rel2obsi_beta.py not found. Please make sure that both "
            "files are stored at the same path."
        )
        sys.exit(1)

    logger.info("Read RELION-Jobs...")
    jobs = list(parse_relion_jobs(args.project_dir))
    logger.info(f"{len(jobs)} jobs found.")

    build_canvas_from_jobs(
        jobs,
        args.output_dir,
        args.project_dir,
        canvas_name=args.canvas_name,
        canvas_depth=args.canvas_depth,
    )


if __name__ == "__main__":
    main()
