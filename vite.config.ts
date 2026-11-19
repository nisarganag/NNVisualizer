import { defineConfig, type Plugin } from 'vite';

// D3 v3's UMD IIFE assumes a classic-script global `this`. An ES module is
// strict, so explicitly bind the same global in both dev and production.
function d3GlobalContext(): Plugin {
  return {
    name: 'd3-v3-global-context',
    transform(code, id) {
      if (id.replaceAll('\\', '/').endsWith('/d3/d3.js')) {
        return { code: code.replace(/\}\(\);?\s*$/, '}.call(globalThis);'), map: null };
      }
    },
  };
}

export default defineConfig({
  plugins: [d3GlobalContext()],
  optimizeDeps: { rolldownOptions: { plugins: [d3GlobalContext()] } },
  worker: { format: 'es' },
  build: { target: 'es2022' },
});
