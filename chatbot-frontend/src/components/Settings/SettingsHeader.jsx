import { Settings, ArrowLeft } from 'lucide-react';

export const SettingsHeader = ({ handleBackToChat }) => {
  return (
    <div className="flex items-center justify-between mb-8">
      <div className="flex items-center space-x-4">
        <button
          onClick={handleBackToChat}
          className="p-2 bg-gray-800 rounded-lg hover:bg-gray-700 transition-colors"
        >
          <ArrowLeft size={20} />
        </button>
        <div className="flex items-center space-x-3">
          <div className="bg-gray-600 p-3 rounded-full">
            <Settings className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-2xl font-bold">Settings</h1>
        </div>
      </div>
    </div>
  );
};