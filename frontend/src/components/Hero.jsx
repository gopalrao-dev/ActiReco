import { useState } from "react";
import { motion } from "framer-motion";

const Hero = ({ onSearch }) => {
  const [mood, setMood] = useState("");

  const heroImage =
    "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?q=80&w=1600";

  return (
    <div
      className="relative h-screen w-full bg-cover bg-center flex items-center justify-center"
      style={{ backgroundImage: `url(${heroImage})` }}
    >
      {/* Dark Gradient Overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-black/80 via-black/60 to-darkbg"></div>

      {/* Content */}
      <div className="relative z-10 text-center px-6 max-w-3xl">

        <motion.h1
          initial={{ opacity: 0, y: -40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-6xl font-bold mb-6"
        >
          Discover Your Next Experience
        </motion.h1>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="text-gray-300 text-lg mb-8"
        >
          Tell us how you're feeling and we’ll recommend the perfect activity.
        </motion.p>

        {/* Input */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center">

          <input
            type="text"
            placeholder="How are you feeling today?"
            value={mood}
            onChange={(e) => setMood(e.target.value)}
            className="w-80 p-3 rounded bg-black/60 backdrop-blur-md border border-gray-600 text-white focus:outline-none focus:ring-2 focus:ring-netflix"
          />

          <button
            onClick={() => onSearch(mood)}
            className="px-8 py-3 bg-netflix hover:bg-red-700 rounded font-semibold transition"
          >
            Get Recommendations
          </button>

        </div>

      </div>
    </div>
  );
};

export default Hero;