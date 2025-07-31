# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from bpy.utils import register_submodule_factory
import bpy
from .properties import update_active_image
from bpy.app.handlers import persistent
from .paint_system import PaintSystem, get_paint_system_images
from .common import import_legacy_updater, is_newer_than
from .custom_icons import load_custom_icons, unload_custom_icons
from . import auto_load
from . import paintsystem_reference

bl_info = {
    "name": "Paint System",
    "author": "Tawan Sunflower, @blastframe, @Skygazer123321",
    "description": "",
    "blender": (4, 1, 0),
    "version": (1, 3, 4),
    "location": "View3D > Sidebar > Paint System",
    "warning": "",
    "category": "Node",
    'support': 'COMMUNITY',
    "tracker_url": "https://github.com/natapol2547/paintsystem"
}

bl_info_copy = bl_info.copy()

print("Paint System: Registering...", __package__)

auto_load.init()

use_legacy_updater = not is_newer_than(4, 5, 0)
if use_legacy_updater:  # Check if Blender version is older than 4.5.0
    addon_updater_ops = import_legacy_updater()


@persistent
def texture_paint_handler(scene):
    # Get the active object and its mode
    obj = getattr(bpy.context, "object", None)
    if obj and hasattr(obj, "mode") and obj.mode == 'TEXTURE_PAINT':
        update_active_image()


@persistent
def save_handler(scene: bpy.types.Scene):
    images = get_paint_system_images()
    for image in images:
        if not image.is_dirty:
            continue
        if image.packed_file or image.filepath == '':
            image.pack()
        else:
            image.save()


@persistent
def refresh_image(scene: bpy.types.Scene):
    ps = PaintSystem(bpy.context)
    active_layer = ps.get_active_layer()
    if active_layer and active_layer.image:
        active_layer.image.reload()


submodules = [
    "properties",
    "operators_layers",
    "operators_utils",
    "operators_bake",
    "panels",
    "paintsystem_reference",
    # "tests",
    # "node_organizer",
    # "operation/test",
]

#This  line automatically registers and unregisters all the submodules you listed
_register, _unregister = register_submodule_factory(__name__, submodules)


def register():
    _register()
    load_custom_icons()
    if addon_updater_ops and use_legacy_updater:
        addon_updater_ops.register(bl_info_copy)
    bpy.app.handlers.depsgraph_update_post.append(texture_paint_handler)
    bpy.app.handlers.save_pre.append(save_handler)
    bpy.app.handlers.load_post.append(refresh_image)


def unregister():
    bpy.app.handlers.load_post.remove(refresh_image)
    bpy.app.handlers.save_pre.remove(save_handler)
    bpy.app.handlers.depsgraph_update_post.remove(texture_paint_handler)
    if addon_updater_ops and use_legacy_updater:
        addon_updater_ops.unregister()
    unload_custom_icons()
    _unregister()
