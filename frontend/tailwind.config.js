/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        constellation: {
          50: '#f0f4ff',
          100: '#e0e9ff',
          200: '#c7d6fe',
          300: '#a4bafc',
          400: '#7f94f8',
          500: '#5b6cf2',
          600: '#4549e6',
          700: '#3a39cb',
          800: '#3131a4',
          900: '#2d2f82',
          950: '#1c1c4c',
        },
      },
    },
  },
  plugins: [],
}
