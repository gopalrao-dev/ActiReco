import { motion } from "framer-motion";

const ActivityCard = ({ activity }) => {

  const image = `https://picsum.photos/300/200?random=${activity.activity_id}`;

  return (
    <motion.div
      whileHover={{ scale: 1.1 }}
      transition={{ type: "spring", stiffness: 200 }}
      className="min-w-[260px] rounded-lg overflow-hidden bg-cardbg shadow-lg cursor-pointer"
    >

      <img
        src={image}
        alt={activity.title}
        className="w-full h-[150px] object-cover"
      />

      <div className="p-4">

        <h3 className="text-lg font-bold">
          {activity.title}
        </h3>

        <p className="text-gray-400 text-sm mt-1">
          {activity.city}
        </p>

        <p className="text-xs text-gray-500 mt-1">
          {activity.tags}
        </p>

      </div>

    </motion.div>
  );
};

export default ActivityCard;