"""usda/scene_place.usda: beebot at the origin with pucks already in basket slots 0~11 and one location box (shelf_L1)
in front, docked at approach_gap - for place / unload tests without driving or loading (isaac_sim.sh place).

    ~/pa-course/.venv-isaacsim/bin/python isaac/scripts/make_scene_place.py

Built from scene_pick.usda (lights, ground, render settings, physics scene) + /World/beebot from scene_sim.usda.
Pucks sit where load leaves them: fingertip xy of basket.yaml q_low (FK, stand::fk) and, for box pucks, faces along the
finger closing axis - not the URDF slot centres (slots 0 and 3 drop 4.9 mm towards the divider). Rerun after
basket.yaml or the URDF basket changes.
"""
import os

from pxr import Gf, Sdf

HERE = os.path.dirname(os.path.abspath(__file__))
USDA = os.path.join(HERE, "..", "usda")
OUT = os.path.join(USDA, "scene_place.usda")

FLOOR_Z = 0.010 + 0.0927 + 0.0005  # base_joint 1 cm + basket floor (URDF storage_collision top) + 0.5 mm
BOX_X = 0.1025 + 0.099  # location box centre ahead of base_footprint: box_half + approach_gap (plan.yaml)
# slot: (shape, colour, x, y, yaw deg) - x y yaw from basket.yaml q_low FK (box: closing axis; cyl: any yaw)
SLOTS = [
    ("box", "red", -0.1649, -0.0953, -0.1), ("box", "green", -0.1697, -0.0325, 90.0),
    ("box", "red", -0.1697, 0.0325, 90.0), ("box", "green", -0.1649, 0.0953, -179.9),
    ("cyl", "red", -0.1048, -0.0975, 0.0), ("cyl", "green", -0.1048, -0.0325, 0.0),
    ("cyl", "red", -0.1048, 0.0325, 0.0), ("cyl", "green", -0.1048, 0.0975, 0.0),
    ("cyl", "red", -0.0398, -0.0975, 0.0), ("cyl", "green", -0.0398, -0.0325, 0.0),
    ("box", "red", -0.0398, 0.0325, -90.0), ("box", "green", -0.0398, 0.0975, 90.0),
]
ASSET = {("box", "green"): "puck/box/puck_box_green_0.usda", ("box", "red"): "puck/box/puck_box_red_1.usda",
         ("cyl", "green"): "puck/cyl/puck_cyl_green_2.usda", ("cyl", "red"): "puck/cyl/puck_cyl_red_3.usda"}


def placed(layer, path, asset, xyz, yaw):
    spec = Sdf.CreatePrimInLayer(layer, path)
    spec.specifier = Sdf.SpecifierDef
    spec.payloadList.Prepend(Sdf.Payload(asset))
    for name, tn, value in (("xformOp:translate", Sdf.ValueTypeNames.Double3, xyz),
                            ("xformOp:rotateXYZ", Sdf.ValueTypeNames.Float3, (0.0, 0.0, yaw)),
                            ("xformOp:scale", Sdf.ValueTypeNames.Float3, (1.0, 1.0, 1.0))):
        Sdf.AttributeSpec(spec, name, tn).default = value
    Sdf.AttributeSpec(spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray, variability=Sdf.VariabilityUniform).default = \
        ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]


def main():
    pick = Sdf.Layer.FindOrOpen(os.path.join(USDA, "scene_pick.usda"))
    sim = Sdf.Layer.FindOrOpen(os.path.join(USDA, "scene_sim.usda"))
    out = Sdf.Layer.CreateAnonymous(".usda")
    out.TransferContent(pick)
    world = out.GetPrimAtPath("/World")
    for child in list(world.nameChildren):  # keep lights and the view camera, drop the pick boxes and pucks
        if child.name not in ("Environment", "cam"):
            del world.nameChildren[child.name]
    Sdf.CopySpec(sim, "/World/beebot", out, "/World/beebot")
    bot = out.GetPrimAtPath("/World/beebot")
    bot.attributes["xformOp:translate"].default = (0.0, 0.0, 0.0)
    bot.attributes["xformOp:orient"].default = Gf.Quatf(1.0)
    placed(out, "/World/shelf_L1", "./shelf/shelf_L1.usda", (BOX_X, 0.0, 0.0), 0.0)
    for slot, (shape, colour, x, y, yaw) in enumerate(SLOTS):
        placed(out, f"/World/puck_{shape}_{colour}_{slot}", "./" + ASSET[(shape, colour)], (x, y, FLOOR_Z), yaw)
    data = dict(out.customLayerData)
    data.setdefault("omni_layer", {})["authoring_layer"] = "./scene_place.usda"
    out.customLayerData = data
    out.Export(OUT)
    print(f"wrote {os.path.normpath(OUT)}: beebot at origin, shelf_L1 at x {BOX_X:.4f}, {len(SLOTS)} pucks in slots 0~11")


if __name__ == "__main__":
    main()
