import bpy
from bpy.types import Panel, Operator, PropertyGroup
from bpy.props import PointerProperty
import bpy
import os

# Property group to hold reference image
class PaintSystemReferenceSettings(PropertyGroup):
    reference_image: PointerProperty(
        name="Reference Image",
        type=bpy.types.Image,
        description="Image used for reference"
    )


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


# Optional floating popup viewer (not used in this method anymore)
class PAINTSYSTEM_OT_ReferencePopup(Operator):
    bl_idname = "paintsystem.reference_popup"
    bl_label = "Reference Image Viewer"
    bl_options = {'REGISTER'}

    bl_region_type = 'WINDOW'  # 👈 Crucial for popup to work from 3D View

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
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.paintsystem_reference = PointerProperty(type=PaintSystemReferenceSettings)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.paintsystem_reference