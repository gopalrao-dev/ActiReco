import { motion } from "framer-motion";

const Loader = () => {
  return (
    <div className="flex justify-center items-center py-20">

      <motion.div
        animate={{ rotate: 360 }}
        transition={{ repeat: Infinity, duration: 1 }}
        className="w-10 h-10 border-4 border-netflix border-t-transparent rounded-full"
      />

    </div>
  );
};

export default Loader;