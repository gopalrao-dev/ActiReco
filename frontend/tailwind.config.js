/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        netflix: "#E50914",
        darkbg: "#141414",
        cardbg: "#1f1f1f",
      },
    },
  },
  plugins: [],
}