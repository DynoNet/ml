from database_information import get_all_climbs
import sqlite3
import re
import random
import torch

path = "/home/tudor/Code/DynoNet/data/raw/kilter.db"


def _build_placement_lookup(conn):
    """Pre-fetches physical X, Y coordinates for all placements to eliminate slow SQL queries inside loops."""
    cursor = conn.cursor()
    query = """
        SELECT p.id, h.name 
        FROM placements p 
        JOIN holes h ON p.hole_id = h.id
        WHERE h.product_id = 1;
    """
    cursor.execute(query)

    lookup = {}
    for placement_id, hole_name in cursor.fetchall():
        x_str, y_str = hole_name.split(",")
        x = int(x_str)

        if y_str == "KB1":
            y = 1
        elif y_str == "KB2":
            y = 2
        else:
            y = int(y_str) + 3

        lookup[placement_id] = (x, y)

    return lookup


def get_tensors_from_climbs(min_ascents=50, max_holds=40):
    conn = sqlite3.connect(path)

    placement_lookup = _build_placement_lookup(conn)

    # 0 = Padding, 1 = Metadata, 2 = Start, 3 = Middle, 4 = Finish, 5 = Foot
    ROLE_MAP = {12: 2, 13: 3, 14: 4, 15: 5}

    climbs = get_all_climbs(min_ascents)
    dataset = []

    for climb in climbs:
        climb_uuid, angle, difficulty, ascents, frames_str = climb

        frames = re.findall(r"p(\d+)r(\d+)", frames_str)
        if len(frames) > max_holds:
            continue

        holds = []
        out_of_bounds = False

        for placement_id, role_id in frames:
            p_id = int(placement_id)
            r_id = int(role_id)

            if p_id not in placement_lookup or r_id not in ROLE_MAP:
                out_of_bounds = True
                break

            x, y = placement_lookup[p_id]
            mapped_role = ROLE_MAP[r_id]

            #TODO: Check if this approach actually filters the climbs as you want (check manually a lot of random climbs and see their layout
            #Because in theory you could have smaller or different board layout that still fit this and fuck up the data
            if x < 1 or x > 35 or y < 1 or y > 38:
                out_of_bounds = True
                break

            holds.append([x, y, mapped_role])

        if out_of_bounds:
            continue

        holds.sort(key=lambda h: (h[1], h[0]))

        metadata_token = [angle, float(difficulty), 1]

        sequence = [metadata_token] + holds

        total_max_length = max_holds + 1
        while len(sequence) < total_max_length:
            sequence.append([0, 0, 0])

        dataset.append(sequence)

    conn.close()
    return dataset


def save_dataset(dataset):
    torch.save(dataset, "/home/tudor/Code/DynoNet/data/processed/dataset.pt")


if __name__ == "__main__":
    dataset = get_tensors_from_climbs(25, 35)
    save_dataset(dataset)

    for i, sequence in enumerate(dataset[:10]):
        print(f"=== Climb {i + 1} ===")
        active_tokens = [token for token in sequence if token != [0, 0, 0]]
        for token in active_tokens:
            print(token)
        print()
    print(len(dataset))


# def get_tensor_from_climbs(max_holds):
#     conn = sqlite3.connect(path)
#     cursor = conn.cursor()
#     climbs = get_all_climbs(50)
#     tensors = ""
#     for climb in climbs:
#         hold_count = 0
#         climb = random.choice(climbs)
#         frames = re.findall(r"p(\d+)r(\d+)", climb[4])
#         if len(frames) > max_holds:
#             continue
#         print(frames)
#         print(climb[5], climb[6])
#         holds = ""
#         for placement_id, role_id in frames:
#             cursor.execute("SELECT hole_id FROM placements WHERE id = ?;", (placement_id,))
#             hole_id = cursor.fetchone()[0]
#             cursor.execute("SELECT name FROM holes WHERE id = ?;", (hole_id,))
#             x, y = cursor.fetchone()[0].split(",")
#             # cursor.execute("SELECT x, y FROM holes WHERE id = ?;", (hole_id,))
#             # x1, y1 = cursor.fetchone()
#             cursor.execute("SELECT name FROM placement_roles WHERE id = ?;", (role_id,))
#             role = cursor.fetchone()[0]
#             print(f"Placement {placement_id} ({role}) -> X: {x}, Y: {y}")
#             # print(f"Placement {placement_id} ({role}) -> X: {x1}, Y: {y1}")
#             if y == "KB1":
#                 y = -2
#             if y == "KB2":
#                 y = -1
#             holds += f"[{x}, {int(y)+3}, {role_id}], "
#             hold_count = hold_count + 1
#         while hold_count < max_holds:
#             holds += "[0, 0, 0], "
#             hold_count = hold_count + 1
#         holds = holds[:-2]
#         print("Tensor: ")
#         print(f"[[{climb[1]}, {climb[2]}, 0], {holds}]")
#         break
#
#     conn.close()

