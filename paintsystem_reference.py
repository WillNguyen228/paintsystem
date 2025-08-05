import sys
for mod in list(sys.modules):
    if mod.startswith("PIL"):
        sys.modules.pop(mod)

import PIL.Image as Image
sys.path.append(r"C:\Users\mdngu\AppData\Roaming\Python\Python311\site-packages")
import torch
import string
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
except ImportError:
    processor = None
    model = None

def get_caption_and_keywords(image_path):
    if processor is None or model is None:
        print("BLIP model not available. Install transformers and dependencies.")
        return None, []
    image = Image.open(image_path).convert("RGB")
    #image = image.copy()  # <-- forces memory safety
    #image.load()  # force load

    print("Image type:", type(image))
    print("Image mode:", image.mode)
    print("Image size:", image.size)
    print("Image path:", image_path)

    if not isinstance(image, Image.Image):
        print("Warning: image is not a PIL.Image.Image instance, converting...")
        import numpy as np
        image = Image.fromarray(np.array(image))  # Convert to standard PIL Image if needed

    import numpy as np
    image_np = np.array(image.convert("RGB"))
    inputs = processor(images=image_np, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    # Keyword extraction
    stopwords = {
        "a", "an", "the", "with", "and", "or", "but", "if", "to", "of", "in", "on", "for", "at", "by", "from", "as", "is", "are", "was", "were"
    }
    tokens = caption.lower().translate(str.maketrans('', '', string.punctuation)).split()
    keywords = [word for word in tokens if word not in stopwords]
    seen = set()
    unique_keywords = []
    for word in keywords:
        if word not in seen:
            seen.add(word)
            unique_keywords.append(word)
    return caption, unique_keywords

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

        # Allow zoom and pan events to pass through to Blender
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'MIDDLEMOUSE', 'PAN', 'TRACKPADPAN', 'TRACKPADZOOM'}:
            return {'PASS_THROUGH'}

        if event.type == 'LEFTMOUSE':
            # Only handle clicks inside the image region; pass through UI clicks
            region = context.region
            # The image region is usually type 'WINDOW', and mouse_region_x/y are relative to it
            # If mouse is outside the region, pass through
            if not (0 <= event.mouse_region_x < region.width and 0 <= event.mouse_region_y < region.height):
                return {'PASS_THROUGH'}
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


        elif event.type == 'ESC':
            self.finish()
            self.report({'INFO'}, "Rectangle tool cancelled")
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}
    

    def crop_and_export(self, context):
        import numpy as np
        image = context.space_data.image
        if not image or not image.has_data:
            self.report({'ERROR'}, "No valid image found")
            return

        region = context.region
        v2d = region.view2d
        ix = image.size[0]
        iy = image.size[1]

        # Convert region coords to image coords using Blender's API
        def region_to_image(x, y):
            # Use Blender's region_to_view to get normalized image-space coordinates (0-1)
            uv_x, uv_y = v2d.region_to_view(x, y)
            print(f"DEBUG: region ({x}, {y}) -> uv ({uv_x}, {uv_y})")
            px = int(round(uv_x * ix))
            py = int(round(uv_y * iy))
            return px, py

        # Get rectangle corners in region coords
        x1, y1 = self._start_mouse
        x2, y2 = self._end_mouse

        # Map both corners to image pixel coords
        px1, py1 = region_to_image(x1, y1)
        px2, py2 = region_to_image(x2, y2)

        # Clamp to image bounds BEFORE calculating width/height
        px1 = max(0, min(ix, px1))
        px2 = max(0, min(ix, px2))
        py1 = max(0, min(iy, py1))
        py2 = max(0, min(iy, py2))

        # Ensure proper order
        min_x = min(px1, px2)
        max_x = max(px1, px2)
        min_y = min(py1, py2)
        max_y = max(py1, py2)

        width = max_x - min_x
        height = max_y - min_y

        print(f"DEBUG: px1, py1 = {px1}, {py1}; px2, py2 = {px2}, {py2}")
        print(f"DEBUG: min_x, max_x = {min_x}, {max_x}; min_y, max_y = {min_y}, {max_y}")
        print(f"DEBUG: width, height = {width}, {height}")

        if width <= 0 or height <= 0:
            self.report({'ERROR'}, f"Invalid selection size (width={width}, height={height})")
            return

        # Get pixel buffer
        pixels = np.array(image.pixels[:], dtype=np.float32)
        pixels = pixels.reshape((iy, ix, 4))  # RGBA

        # Crop selection (Y axis: 0 at bottom)
        cropped = pixels[min_y:max_y, min_x:max_x, :]
        new_pixels = cropped.flatten()

        # Create new image
        new_image = bpy.data.images.new("Cropped_Image", width=width, height=height, alpha=True)
        new_image.pixels = new_pixels.tolist()

        # Save to disk
        output_path = r"C:\Users\mdngu\Downloads\cropped_image.png"
        new_image.filepath_raw = output_path
        new_image.file_format = 'PNG'
        new_image.save()

        self.report({'INFO'}, f"Exported to: {output_path}")

        from bl_ext.user_default.blenderkit import global_vars, search  # Fixed import

        def create_new_blenderkit_tab(name="search"):
            tabs = global_vars.TABS["tabs"]
            new_tab = {
                "name": name,
                "history": [],
                "history_index": 0
            }

            tabs.append(new_tab)
            global_vars.TABS["active_tab"] = len(tabs) - 1
            search.create_history_step(new_tab)

        def search_each_keyword_as_tab(keywords):
            if "blenderkit" not in bpy.context.preferences.addons:
                print("BlenderKit addon not enabled")
                return

            for keyword in keywords:
                create_new_blenderkit_tab(name=keyword)
                bpy.ops.view3d.blenderkit_search('INVOKE_DEFAULT', keywords=keyword)
                print(f"Searched: {keyword}")

        # Run BLIP captioning and keyword extraction
        caption, keywords = get_caption_and_keywords(output_path)
        if caption is not None:
            print("Caption:", caption)
            print("Keywords:", keywords)
        
        if keywords:
            search_each_keyword_as_tab(keywords)
            
        """
        if keywords:           
            if any(key.endswith("blenderkit") for key in bpy.context.preferences.addons.keys()):
                print("BlenderKit addon enabled")
                for kw in keywords:
                    print("Searching BlenderKit for:", kw)
                    bpy.ops.view3d.blenderkit_search(keywords=kw)
            else:
                print("BlenderKit addon NOT enabled")
        """

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
        # Use a timer to restore the reference image after Blender's internal reassignment
        def restore_image_later():
            ref = bpy.context.scene.paintsystem_reference
            if ref.reference_image:
                for window in bpy.context.window_manager.windows:
                    for area in window.screen.areas:
                        if area.type == 'IMAGE_EDITOR':
                            for space in area.spaces:
                                if space.type == 'IMAGE_EDITOR':
                                    space.image = ref.reference_image
            return None  # Only run once
        bpy.app.timers.register(restore_image_later, first_interval=0.1)

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