/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  // 使用 important 选择器限制 Tailwind 只作用于 opencut-root 容器
  important: '#opencut-root',
  // 禁用 preflight (base reset)，避免影响全局样式
  corePlugins: {
    preflight: false,
  },
  theme: {
    extend: {},
  },
  plugins: [],
}
