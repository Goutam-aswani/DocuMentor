export const MessageDisplay = ({ message, error }) => {
  if (!message && !error) return null;
  
  return (
    <div className={`p-4 rounded-lg mb-6 ${
      message ? 'bg-green-600/20 text-green-400' : 'bg-red-600/20 text-red-400'
    }`}>
      {message || error}
    </div>
  );
};