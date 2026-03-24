import ActivityCard from "./ActivityCard";

const ActivityRow = ({ title, activities }) => {
  return (
    <div className="px-10 py-14 bg-darkbg">

      <h2 className="text-3xl font-bold mb-6">
        {title}
      </h2>

      <div className="flex gap-6 overflow-x-auto pb-4 scrollbar-hide">

        {activities.map((activity) => (
          <ActivityCard
            key={activity.activity_id}
            activity={activity}
          />
        ))}

      </div>

    </div>
  );
};

export default ActivityRow;