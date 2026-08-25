import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import cssInjectedByJs from "vite-plugin-css-injected-by-js";

/**
 * Vite config for nkd_sigmas_curve.
 *
 * Builds the Vue widget into web/nkd_sigmas_curve.js.
 * The ComfyUI scripts (app.js, api.js) are marked as external so they
 * are imported at runtime from the browser context.
 */
export default defineConfig({
  plugins: [vue(), cssInjectedByJs({ topExecutionPriority: false })],

  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },

  build: {
    lib: {
      entry: "./src/main.ts",
      formats: ["es"],
      fileName: "nkd_sigmas_curve",
    },
    rollupOptions: {
      // These paths are resolved by the browser at runtime
      external: [
        "../../scripts/app.js",
        "../../scripts/api.js",
      ],
      output: {
        // Output directly into web/ so ComfyUI can serve it
        dir: "web",
        entryFileNames: "nkd_sigmas_curve.js",
        assetFileNames: "assets/[name].[ext]",
      },
    },
    sourcemap: false,
    minify: false,      // Keep readable for debugging
    cssCodeSplit: false,
  },

  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
  },
});
