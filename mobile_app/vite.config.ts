import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    // ONNX wasm files must be served with correct MIME type
    // and should not be inlined
    assetsInlineLimit: 0,
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
})
