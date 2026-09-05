from database_information import get_all_climbs
import sqlite3
import re
import random

path = "/home/tudor/Code/DynoNet/data/raw/kilter.db"

def get_tensor_from_climbs(max_holds):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    climbs = get_all_climbs(50)
    tensors = ""
    for climb in climbs:
        hold_count = 0
        climb = random.choice(climbs)
        frames = re.findall(r"p(\d+)r(\d+)", climb[4])
        if len(frames) > max_holds:
            continue
        print(frames)
        print(climb[5], climb[6])
        holds = ""
        for placement_id, role_id in frames:
            cursor.execute("SELECT hole_id FROM placements WHERE id = ?;", (placement_id,))
            hole_id = cursor.fetchone()[0]
            cursor.execute("SELECT name FROM holes WHERE id = ?;", (hole_id,))
            x, y = cursor.fetchone()[0].split(",")
            # cursor.execute("SELECT x, y FROM holes WHERE id = ?;", (hole_id,))
            # x1, y1 = cursor.fetchone()
            cursor.execute("SELECT name FROM placement_roles WHERE id = ?;", (role_id,))
            role = cursor.fetchone()[0]
            print(f"Placement {placement_id} ({role}) -> X: {x}, Y: {y}")
            # print(f"Placement {placement_id} ({role}) -> X: {x1}, Y: {y1}")
            if y == "KB1":
                y = -2
            if y == "KB2":
                y = -1
            holds += f"[{x}, {int(y)+3}, {role_id}], "
            hold_count = hold_count + 1
        while hold_count < max_holds:
            holds += "[0, 0, 0], "
            hold_count = hold_count + 1
        holds = holds[:-2]
        print("Tensor: ")
        print(f"[[{climb[1]}, {climb[2]}, 0], {holds}]")
        break

    conn.close()

if __name__ == "__main__":
    get_tensor_from_climbs(40)

