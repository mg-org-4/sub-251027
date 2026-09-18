# Layered parallax

`FL Parallax Layer` accepts RGBA video or RGB plus a foreground mask (white = opaque). It adds depth, scale, screen-relative offsets and opacity, and produces a single checkerboard matte preview. An explicit mask overrides embedded alpha.

`FL Layered Parallax` takes an opaque background and expandable layer inputs. It sorts cutouts far-to-near and projects them using camera translation and forward travel. Smaller depth values move faster; the background must be farther than every cutout. Rendering uses premultiplied-alpha bilinear sampling in four-frame chunks on ComfyUI's selected device, with a CPU option. Outputs are the moving composite, the same animation with the camera locked, and a middle-frame depth preview (near = bright).

Camera controls:

- `motion`: burst/settle, eased glide, sinusoidal loop, or locked.
- `travel_x/y`: camera excursion in output widths/heights at depth 1; projected displacement divides by depth.
- `push_in`: forward travel in depth units. Keep every plane in front of the camera.
- `overscan`: enlarges only the background to cover movement. Beyond the source boundary, the background edge is extended; hidden scenery is not generated.

All animated plates must have equal frame counts. Single images are held over the longest animation. Set `frames` to animate still-image plates for a specified duration; zero inherits the source frame count. An explicit length must match any animated inputs. There is no silent trimming or frame-rate conversion. Set FPS in the exporter.

This is multiplane 2.5D composition, not a reconstructed scene or novel-view synthesis. It cannot expose unseen sides of objects. Alpha quality and independent generated motion affect the result. Loop camera movement does not make independently generated plates loop seamlessly.

Changing compositor controls does not change upstream generation or matting inputs. ComfyUI may reuse those cached results while resident; cache survival across restarts or memory pressure is not guaranteed. Save raw plates and RGBA assets for later reuse.
