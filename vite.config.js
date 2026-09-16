import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Needed when the dev server runs inside Docker: listen on all interfaces,
    // and poll for changes because bind-mounted files don't emit inotify events.
    host: true,
    port: 5173,
    watch: { usePolling: true, interval: 500 },
  },
  build: {
    chunkSizeWarningLimit: 1200, // echarts is a single large chunk; that's expected
  },
})
