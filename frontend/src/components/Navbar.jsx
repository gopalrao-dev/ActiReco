const Navbar = () => {
  return (
    <div className="fixed top-0 left-0 w-full bg-black/80 backdrop-blur-md z-50">
      <div className="max-w-7xl mx-auto flex items-center justify-between px-6 py-4">

        <h1 className="text-netflix text-2xl font-bold">
          ActiReco
        </h1>

        <div className="space-x-6 text-gray-300">
          <button className="hover:text-white transition">
            Home
          </button>
          <button className="hover:text-white transition">
            Activities
          </button>
          <button className="hover:text-white transition">
            My List
          </button>
        </div>

      </div>
    </div>
  );
};

export default Navbar;