import bpy
from bpy.types import Panel, Operator, PropertyGroup
from bpy.props import PointerProperty
import bpy
import os

import gpu
from gpu_extras.batch import batch_for_shader

# Property group to hold reference image
class PaintSystemReferenceSettings(PropertyGroup):
    reference_image: PointerProperty(
        name="Reference Image",
        type=bpy.types.Image,
        description="Image used for reference"
    )

# Modal operator to draw rectangle
class PAINTSYSTEM_OT_DrawRectangle(Operator):
    bl_idname = "paintsystem.draw_rectangle"
    bl_label = "Draw Selection Rectangle"
    bl_options = {'REGISTER', 'UNDO'}

    _handle = None
    _start_mouse = None
    _end_mouse = None
    _drawing = False

    def modal(self, context, event):
        print("Mouse event:", event.type, event.mouse_region_x, event.mouse_region_y)
        if context.area:
            context.area.tag_redraw()

        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self._start_mouse = (event.mouse_region_x, event.mouse_region_y)
                self._end_mouse = self._start_mouse
                self._drawing = True
            elif event.value == 'RELEASE':
                #self.finish()
                #return {'FINISHED'}
                self._drawing = False
                self._start_mouse = None
                self._end_mouse = None
                # Keep running modal to allow drawing more rectangles

        elif event.type == 'MOUSEMOVE' and self._drawing:
            self._end_mouse = (event.mouse_region_x, event.mouse_region_y)

        elif event.type == 'RIGHTMOUSE' and event.value == 'RELEASE':
            if self._start_mouse and self._end_mouse:
                self.crop_and_export(context)
            self.finish()
            return {'FINISHED'}

        return {'RUNNING_MODAL'}
    
    def crop_and_export(self, context):
        import numpy as np
        import os

        image = context.space_data.image
        if not image or not image.has_data:
            self.report({'ERROR'}, "No valid image found")
            return

        # Convert screen to image coordinates
        def region_to_image_coords(region, rv3d, x, y):
            return context.space_data.region_to_view(x, y)

        x1, y1 = self._start_mouse
        x2, y2 = self._end_mouse

        # Ensure coordinates are properly ordered
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)

        sx = context.region.width
        sy = context.region.height

        ix = image.size[0]
        iy = image.size[1]

        # Normalize screen to 0-1 then to image resolution
        px1 = int((xmin / sx) * ix)
        py1 = int((ymin / sy) * iy)
        px2 = int((xmax / sx) * ix)
        py2 = int((ymax / sy) * iy)

        width = px2 - px1
        height = py2 - py1

        if width <= 0 or height <= 0:
            self.report({'ERROR'}, "Invalid selection size")
            return

        # Get pixel buffer
        pixels = np.array(image.pixels[:], dtype=np.float32)
        pixels = pixels.reshape((iy, ix, 4))  # RGBA

        # Crop selection (image Y-axis is flipped)
        cropped = pixels[iy - py2:iy - py1, px1:px2, :]

        # Flatten for image.pixels
        new_pixels = cropped[::-1].flatten()

        # Create new image
        new_image = bpy.data.images.new("Cropped_Image", width=width, height=height, alpha=True)
        new_image.pixels = new_pixels.tolist()

        # Save to disk
        output_path = r"C:\Users\mdngu\Downloads\cropped_image.png"
        new_image.filepath_raw = output_path
        new_image.file_format = 'PNG'
        new_image.save()

        self.report({'INFO'}, f"Exported to: {output_path}")

    def invoke(self, context, event):
        print("Rectangle tool started")
        if context.area.type != 'IMAGE_EDITOR':
            self.report({'WARNING'}, "This operator only works in the Image Editor")
            return {'CANCELLED'}

        self._start_mouse = None
        self._end_mouse = None
        self._drawing = False

        self._handle = bpy.types.SpaceImageEditor.draw_handler_add(
            self.draw_callback, (context,), 'WINDOW', 'POST_PIXEL'
        )

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def finish(self):
        if self._handle:
            bpy.types.SpaceImageEditor.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None
        self._drawing = False

    def draw_callback(self, context):
        print("Drawing rectangle callback")
        if not self._drawing:
            return

        x1, y1 = self._start_mouse
        x2, y2 = self._end_mouse

        vertices = [
            (x1, y1), (x2, y1),
            (x2, y2), (x1, y2)
        ]
        indices = [(0, 1), (1, 2), (2, 3), (3, 0)]

        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINES', {"pos": vertices}, indices=indices)

        gpu.state.line_width_set(2.0)
        shader.bind()
        shader.uniform_float("color", (1.0, 0.5, 0.0, 1.0))  # orange
        batch.draw(shader)

# Sidebar panel in the Image Editor or Viewport
class PAINTSYSTEM_PT_ReferenceImagePanel(Panel):
    bl_label = "Reference Image"
    bl_space_type = 'IMAGE_EDITOR'  # Change to 'VIEW_3D' if needed
    bl_region_type = 'UI'
    bl_category = 'PaintSystem'

    def draw(self, context):
        layout = self.layout
        ref = context.scene.paintsystem_reference

        if ref.reference_image:
            layout.template_preview(ref.reference_image, show_buttons=False)
        else:
            layout.label(text="No image loaded")

        layout.operator("paintsystem.open_reference_window", text="Open Image")

        layout.operator("paintsystem.draw_rectangle", text="Draw Selection Box")


# Optional floating popup viewer (not used in this method anymore)
class PAINTSYSTEM_OT_ReferencePopup(Operator):
    bl_idname = "paintsystem.reference_popup"
    bl_label = "Reference Image Viewer"
    bl_options = {'REGISTER'}

    bl_region_type = 'WINDOW'  # Crucial for popup to work from 3D View

    def execute(self, context):
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=400)

    def draw(self, context):
        layout = self.layout
        settings = context.scene.paintsystem_reference
        layout.template_ID_preview(settings, "reference_image", new="image.open")

class PAINTSYSTEM_OT_OpenReferenceFloatingWindow(bpy.types.Operator):
    bl_idname = "paintsystem.open_reference_window"
    bl_label = "Open Reference Image in New Window"
    bl_description = "Open the selected reference image in a floating Image Editor window"
    bl_options = {'REGISTER'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        scene = context.scene
        ref = scene.paintsystem_reference

        # Load new image if selected through file browser
        if self.filepath:
            image_name = os.path.basename(self.filepath)
            try:
                image = bpy.data.images.load(self.filepath)
            except RuntimeError as e:
                self.report({'ERROR'}, f"Failed to load image: {e}")
                return {'CANCELLED'}
            ref.reference_image = image
        else:
            image = ref.reference_image

        #image = ref.reference_image
        if not image:
            self.report({'ERROR'}, "No reference image selected or loaded")
            return {'CANCELLED'}

        # Create a new floating Image Editor window
        bpy.ops.screen.userpref_show('INVOKE_DEFAULT')
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'PREFERENCES':
                    area.ui_type = 'IMAGE_EDITOR'
                    for space in area.spaces:
                        if space.type == 'IMAGE_EDITOR':
                            space.image = image
                    break

         # Automatically start rectangle tool
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    for region in area.regions:
                        if region.type == 'WINDOW':
                            override = {
                                "window": window,
                                "screen": window.screen,
                                "area": area,
                                "region": region
                            }
                            with context.temp_override(**override):
                                bpy.ops.paintsystem.draw_rectangle('INVOKE_DEFAULT')
                            return {'FINISHED'}

        return {'FINISHED'}

    def invoke(self, context, event):
        ref = context.scene.paintsystem_reference

        # If an image is already selected, open directly
        if ref.reference_image:
            return self.execute(context)

        # Otherwise prompt for file
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

# Registration
classes = (
    PaintSystemReferenceSettings,
    PAINTSYSTEM_PT_ReferenceImagePanel,
    #PAINTSYSTEM_OT_ReferencePopup,
    PAINTSYSTEM_OT_OpenReferenceFloatingWindow,
    PAINTSYSTEM_OT_DrawRectangle,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.paintsystem_reference = PointerProperty(type=PaintSystemReferenceSettings)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.paintsystem_reference