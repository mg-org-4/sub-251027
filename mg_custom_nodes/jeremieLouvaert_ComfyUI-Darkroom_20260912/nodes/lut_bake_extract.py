"""
LUT Bake Extract node for ComfyUI-Darkroom.
Splits a 2-image batch (photo + padded lattice) back into separate outputs
after a grading chain has processed them together. Pair with LUT Bake Inject.
"""


class LUTBakeExtract:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batched_image": ("IMAGE", {
                    "tooltip": "Output of the grading chain that started with LUT Bake Inject. "
                               "Must still be a 2-image batch (photo + padded lattice)."
                }),
                "bake_meta": ("LUT_BAKE_META", {
                    "tooltip": "Connect from LUT Bake Inject's bake_meta output. Carries "
                               "the crop coordinates so we can split the batch back out."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("graded_photo", "graded_lattice", "lut_size")
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Pipeline"

    def execute(self, batched_image, bake_meta):
        if batched_image.shape[0] < 2:
            raise ValueError(
                f"[Darkroom] LUT Bake Extract: expected batch of 2 (photo + lattice), "
                f"got batch of {batched_image.shape[0]}. A node in the chain may have "
                f"collapsed the batch — check for any node that takes a single image."
            )

        ph = bake_meta["photo_h"]
        pw = bake_meta["photo_w"]
        lh = bake_meta["lattice_h"]
        lw = bake_meta["lattice_w"]
        lut_size = bake_meta["lut_size"]

        photo_padded = batched_image[0:1]
        lattice_padded = batched_image[1:2]

        graded_photo = photo_padded[:, :ph, :pw, :].contiguous()
        graded_lattice = lattice_padded[:, :lh, :lw, :].contiguous()

        print(f"[Darkroom] LUT Bake Extract: photo {pw}x{ph}, lattice {lw}x{lh}, "
              f"lut_size={lut_size}")

        return (graded_photo, graded_lattice, lut_size)


NODE_CLASS_MAPPINGS = {"DarkroomLUTBakeExtract": LUTBakeExtract}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomLUTBakeExtract": "LUT Bake Extract"}
