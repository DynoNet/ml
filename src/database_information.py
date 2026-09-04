import sqlite3
import os
import re

for path in ["/home/tudor/Code/DynoNet/data/raw/kilter.db",]:
    if not os.path.exists(path):
        continue
    print(f"--- Inspecting: {path} ---")
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table in tables:
        t_name = table[0]
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {t_name};")
            count = cursor.fetchone()[0]
            print(f"  Table '{t_name}': {count} rows")
        except sqlite3.OperationalError:
            print(f"  Table '{t_name}': Could not count rows")

    print()
    print("Random climb for general structure :")
    cursor.execute("SELECT * FROM climbs WHERE layout_id = 1 ORDER BY RANDOM() LIMIT 1;")
    column_names0 = [col[0] for col in cursor.description]
    random_climb = cursor.fetchone()
    print(column_names0)
    print(random_climb)
    frames_str = random_climb[14]
    frames = re.findall(r"p(\d+)r(\d+)", frames_str)


    print("Table Description")
    cursor.execute("SELECT * FROM placements ORDER BY RANDOM() LIMIT 1;")
    column_names1 = [col[0] for col in cursor.description]
    print("Placements: ")
    print(column_names1)
    cursor.execute("SELECT * FROM holes ORDER BY RANDOM() LIMIT 1;")
    column_names2 = [col[0] for col in cursor.description]
    print("Holes: ")
    print(column_names2)
    
    for placement_id, role_id in frames:
        # remove starting p from frame
        cursor.execute("SELECT hole_id FROM placements WHERE id = ?;", (placement_id,))
        hole_id = cursor.fetchone()[0]
        cursor.execute("SELECT x, y FROM holes WHERE id = ?;", (hole_id,))
        x, y = cursor.fetchone()
        cursor.execute("SELECT name FROM placement_roles WHERE id = ?;", (role_id,))
        role = cursor.fetchone()[0]
        print(f"Placement {placement_id} ({role}) -> X: {x}, Y: {y}")


    conn.close()
