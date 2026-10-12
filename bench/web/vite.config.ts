import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

// https://vite.dev/config/
export default defineConfig({
  plugins: [svelte()],
  // Relative asset paths, so the built site works from any subdirectory.
  base: './',
  // `src/lib/github.ts` reads the model files, which are outside this folder.
  server: { fs: { allow: ['.', '../../WrenCore/ext/WrenCorePosteriorDBExt/models'] } },
})
