import { useState } from "react";
import Hero from "./components/Hero";
import ActivityRow from "./components/ActivityRow";
import Navbar from "./components/Navbar";
import Loader from "./components/Loader";
import { getRecommendationsWithMood } from "./services/api";

function App() {

  const [activities, setActivities] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async (moodText) => {

    try {

      setLoading(true);

      const res = await getRecommendationsWithMood({
        user_id: "u1",
        top_k: 6,
        mood_text: moodText,
      });

      setActivities(res.data.recommendations);

    } catch (error) {

      console.error("Error fetching recommendations:", error);

    } finally {

      setLoading(false);

    }

  };

  return (

    <>
      <Navbar />

      <Hero onSearch={handleSearch} />

      {loading && <Loader />}

      {activities.length > 0 && !loading && (
        <ActivityRow
          title="Recommended For You"
          activities={activities}
        />
      )}

    </>

  );
}

export default App;