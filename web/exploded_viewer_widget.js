/**
 * ComfyUI Hunyuan3D-Part - Exploded Mesh Viewer Widget
 * Interactive 3D viewer with explosion slider for segmented meshes
 */

import { app } from "../../../scripts/app.js";

console.log("[Hunyuan3D] Loading exploded viewer extension...");

app.registerExtension({
    name: "hunyuan3d.explodedviewer",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ExplodedMeshViewer") {
            console.log("[Hunyuan3D] Registering Exploded Mesh Viewer node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                console.log("[Hunyuan3D] Exploded viewer node created, adding widget");

                // Create iframe for 3D viewer
                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.height = "100%";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#1a1a1a";
                iframe.style.aspectRatio = "1";

                // Point to our HTML viewer (with cache buster)
                iframe.src = "/extensions/ComfyUI-Hunyuan3D-Part/exploded_viewer.html?v=" + Date.now();

                // Add DOM widget to node
                const widget = this.addDOMWidget("preview", "EXPLODED_VIEWER", iframe, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                // Set widget size - make it square
                widget.computeSize = function(width) {
                    const size = [width || 600, width || 600];
                    return size;
                };

                widget.element = iframe;

                // Store iframe reference
                this.explodedViewerIframe = iframe;

                // Set initial node size
                this.setSize([600, 600]);

                // Handle execution - load mesh into viewer
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    console.log("[Hunyuan3D] onExecuted called with message:", message);
                    onExecuted?.apply(this, arguments);

                    // Extract scene file and metadata from message
                    if (message?.scene_file && message.scene_file[0]) {
                        const filename = message.scene_file[0];
                        const numParts = message.num_parts ? message.num_parts[0] : 0;
                        const globalCenter = message.global_center ? message.global_center[0] : [0, 0, 0];
                        const maxExtent = message.max_extent ? message.max_extent[0] : 1.0;

                        console.log(`[Hunyuan3D] Loading scene: ${filename} (${numParts} parts)`);

                        // ComfyUI serves output files via /view API endpoint
                        const filepath = `/view?filename=${encodeURIComponent(filename)}&type=output&subfolder=`;

                        // Send message to iframe (with delay to ensure iframe is loaded)
                        setTimeout(() => {
                            if (iframe.contentWindow) {
                                console.log(`[Hunyuan3D] Sending postMessage to iframe: ${filepath}`);
                                iframe.contentWindow.postMessage({
                                    type: "LOAD_EXPLODED_SCENE",
                                    filepath: filepath,
                                    numParts: numParts,
                                    globalCenter: globalCenter,
                                    maxExtent: maxExtent,
                                    timestamp: Date.now()
                                }, "*");
                            } else {
                                console.error("[Hunyuan3D] Iframe contentWindow not available");
                            }
                        }, 100);
                    } else {
                        console.log("[Hunyuan3D] No scene_file in message data. Keys:", Object.keys(message || {}));
                    }
                };

                return r;
            };
        }
    }
});

console.log("[Hunyuan3D] Exploded viewer extension registered");
